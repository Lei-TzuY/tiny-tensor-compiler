from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .inference import infer_binary, infer_reshape, infer_sum, normalize_sum_axes
from .ir import DType, Function, Module, Operation, TensorType, Value
from .verifier import verify


class AutodiffError(ValueError):
    """Raised when a module is outside the bounded reverse-mode autodiff contract."""


_SUPPORTED_BACKWARD_OPS = frozenset({"input", "const", "add", "mul", "sum", "reshape"})
_FLOAT_DTYPES = frozenset({DType.FLOAT32, DType.FLOAT64})


def differentiate_module(
    module: Module,
    *,
    output_index: int = 0,
    wrt: Sequence[int] = (0,),
) -> Module:
    """Differentiate one static scalar floating output with respect to runtime inputs.

    The returned verified module keeps the original runtime-input ABI and returns gradients
    in ``wrt`` order. The first phase intentionally supports only the pure differentiable
    subset ``add``/``mul``/``sum``/``reshape`` plus input/constant leaves. Every adjoint is
    expressed with ordinary tensor IR operations so all existing lowering/backends remain
    unchanged.
    """
    if not isinstance(module, Module):
        raise TypeError("differentiate_module requires a Module")
    verify(module)

    return_op = _terminal_return(module)
    selected_output = _select_output(return_op, output_index)
    input_ops = _input_ops_by_index(module)
    requested = _normalize_wrt(wrt, input_ops)

    _validate_static_floating_contract(module, selected_output, requested, input_ops)
    ancestors = _collect_ancestors(selected_output)
    _validate_backward_slice(ancestors)

    function = Function(f"{module.function.name}_grad")
    value_map: dict[Value, Value] = {}
    cloned_forward_ops: list[Operation] = []

    for op in module.function.ops:
        if op.opcode == "return":
            continue
        include = op.opcode == "input" or any(result in ancestors for result in op.results)
        if not include:
            continue
        cloned = _clone_op(function, op, value_map)
        cloned_forward_ops.append(cloned)

    try:
        cloned_output = value_map[selected_output]
    except KeyError as exc:  # pragma: no cover - guarded by ancestor collection
        raise RuntimeError("internal autodiff error: selected output was not cloned") from exc

    gradients: dict[Value, Value] = {
        cloned_output: _constant(function, np.array(1, dtype=cloned_output.type.dtype.to_numpy()))
    }

    for op in reversed(cloned_forward_ops):
        if not op.results:
            continue
        result = op.results[0]
        upstream = gradients.get(result)
        if upstream is None:
            continue
        _propagate_adjoint(function, op, upstream, gradients)

    outputs: list[Value] = []
    for input_index in requested:
        original_input = input_ops[input_index]
        cloned_input = value_map[original_input.results[0]]
        gradient = gradients.get(cloned_input)
        if gradient is None:
            gradient = _zeros(function, cloned_input.type)
        if gradient.type != cloned_input.type:
            raise RuntimeError(
                "internal autodiff error: input gradient type does not match input type"
            )
        outputs.append(gradient)

    function.add_op("return", operands=outputs)
    differentiated = Module(function)
    verify(differentiated)
    return differentiated


def _terminal_return(module: Module) -> Operation:
    returns = [op for op in module.function.ops if op.opcode == "return"]
    if len(returns) != 1:
        raise AutodiffError("autodiff requires exactly one terminal return operation")
    return returns[0]


def _select_output(return_op: Operation, output_index: int) -> Value:
    if not isinstance(output_index, int) or isinstance(output_index, bool):
        raise AutodiffError("output index must be an integer")
    if output_index < 0 or output_index >= len(return_op.operands):
        raise AutodiffError(
            f"output index {output_index} is out of range for {len(return_op.operands)} outputs"
        )
    output = return_op.operands[output_index]
    if output.type.shape:
        raise AutodiffError("reverse-mode autodiff currently requires a scalar selected output")
    if output.type.dtype not in _FLOAT_DTYPES:
        raise AutodiffError("reverse-mode autodiff currently requires a floating selected output")
    return output


def _input_ops_by_index(module: Module) -> dict[int, Operation]:
    inputs: dict[int, Operation] = {}
    for op in module.function.ops:
        if op.opcode != "input":
            continue
        index = op.attrs.get("index")
        if not isinstance(index, int) or isinstance(index, bool):  # verifier should reject first
            raise TypeError("verified input unexpectedly has a non-integer index")
        inputs[index] = op
    return inputs


def _normalize_wrt(wrt: Sequence[int], inputs: dict[int, Operation]) -> tuple[int, ...]:
    if isinstance(wrt, (str, bytes)):
        raise TypeError("wrt must be a sequence of runtime input indices")
    try:
        requested = tuple(wrt)
    except TypeError as exc:
        raise TypeError("wrt must be a sequence of runtime input indices") from exc
    if not requested:
        raise AutodiffError("wrt must contain at least one runtime input index")
    for index in requested:
        if not isinstance(index, int) or isinstance(index, bool):
            raise AutodiffError("wrt entries must be runtime input indices")
    if len(set(requested)) != len(requested):
        raise AutodiffError("wrt contains duplicate runtime input indices")
    missing = [index for index in requested if index not in inputs]
    if missing:
        raise AutodiffError(f"wrt references unknown runtime input index {missing[0]}")
    return requested


def _validate_static_floating_contract(
    module: Module,
    output: Value,
    requested: tuple[int, ...],
    inputs: dict[int, Operation],
) -> None:
    for op in inputs.values():
        type_ = op.results[0].type
        if not type_.is_static:
            raise AutodiffError("reverse-mode autodiff currently requires static runtime inputs")
    for index in requested:
        if inputs[index].results[0].type.dtype not in _FLOAT_DTYPES:
            raise AutodiffError("reverse-mode autodiff wrt inputs must use a floating dtype")
    if not output.type.is_static:  # scalar is static, kept explicit for contract clarity
        raise AutodiffError("reverse-mode autodiff currently requires static shapes")
    for op in module.function.ops:
        for result in op.results:
            if result in _collect_ancestors(output) and not result.type.is_static:
                raise AutodiffError("reverse-mode autodiff currently requires static shapes")


def _collect_ancestors(output: Value) -> frozenset[Value]:
    values: set[Value] = set()
    stack = [output]
    while stack:
        value = stack.pop()
        if value in values:
            continue
        values.add(value)
        producer = value.producer
        if producer is not None:
            stack.extend(producer.operands)
    return frozenset(values)


def _validate_backward_slice(ancestors: frozenset[Value]) -> None:
    producers = {value.producer for value in ancestors if value.producer is not None}
    for op in producers:
        if op.opcode not in _SUPPORTED_BACKWARD_OPS:
            raise AutodiffError(
                f"unsupported {op.opcode!r} operation on reverse-mode backward slice"
            )
        if len(op.results) != 1:
            raise AutodiffError(
                f"unsupported {op.opcode!r} multi-result operation on backward slice"
            )
        if op.results[0].type.dtype not in _FLOAT_DTYPES:
            raise AutodiffError("reverse-mode backward slice must use floating tensor values")


def _clone_op(function: Function, op: Operation, value_map: dict[Value, Value]) -> Operation:
    try:
        operands = tuple(value_map[operand] for operand in op.operands)
    except KeyError as exc:
        raise RuntimeError("internal autodiff error: forward operand was not cloned") from exc
    attrs = dict(op.attrs)
    if op.opcode == "const":
        attrs["value"] = np.array(op.attrs["value"], copy=True)
    cloned = function.add_op(
        op.opcode,
        operands=operands,
        result_types=tuple(result.type for result in op.results),
        attrs=attrs,
    )
    for original, replacement in zip(op.results, cloned.results, strict=True):
        value_map[original] = replacement
    return cloned


def _propagate_adjoint(
    function: Function,
    op: Operation,
    upstream: Value,
    gradients: dict[Value, Value],
) -> None:
    if op.opcode in {"input", "const"}:
        return
    if op.opcode == "add":
        for operand in op.operands:
            _accumulate(
                function,
                gradients,
                operand,
                _unbroadcast(function, upstream, operand.type),
            )
        return
    if op.opcode == "mul":
        lhs, rhs = op.operands
        lhs_contribution = _unbroadcast(function, _multiply(function, upstream, rhs), lhs.type)
        rhs_contribution = _unbroadcast(function, _multiply(function, upstream, lhs), rhs.type)
        _accumulate(function, gradients, lhs, lhs_contribution)
        _accumulate(function, gradients, rhs, rhs_contribution)
        return
    if op.opcode == "sum":
        (operand,) = op.operands
        contribution = _expand_sum_adjoint(
            function,
            upstream,
            operand.type,
            op.attrs.get("axis"),
        )
        _accumulate(function, gradients, operand, contribution)
        return
    if op.opcode == "reshape":
        (operand,) = op.operands
        contribution = _reshape(function, upstream, operand.type.shape)
        _accumulate(function, gradients, operand, contribution)
        return
    raise RuntimeError(f"internal autodiff error: unsupported propagated opcode {op.opcode!r}")


def _accumulate(
    function: Function,
    gradients: dict[Value, Value],
    target: Value,
    contribution: Value,
) -> None:
    if contribution.type != target.type:
        raise RuntimeError("internal autodiff error: adjoint contribution has the wrong type")
    previous = gradients.get(target)
    gradients[target] = contribution if previous is None else _add(function, previous, contribution)


def _unbroadcast(function: Function, gradient: Value, target_type: TensorType) -> Value:
    if gradient.type == target_type:
        return gradient
    output_shape = gradient.type.shape
    target_shape = target_type.shape
    if len(target_shape) > len(output_shape):
        raise RuntimeError("internal autodiff error: cannot unbroadcast to a higher-rank type")

    offset = len(output_shape) - len(target_shape)
    axes = list(range(offset))
    for target_axis, target_dim in enumerate(target_shape):
        output_dim = output_shape[offset + target_axis]
        if target_dim == 1 and output_dim != 1:
            axes.append(offset + target_axis)
        elif target_dim != output_dim:
            raise RuntimeError("internal autodiff error: incompatible broadcast adjoint shapes")

    reduced = gradient
    if axes:
        axis: int | tuple[int, ...] = axes[0] if len(axes) == 1 else tuple(axes)
        reduced = _sum(function, reduced, axis)
    if reduced.type.shape != target_shape:
        reduced = _reshape(function, reduced, target_shape)
    if reduced.type != target_type:
        raise RuntimeError("internal autodiff error: unbroadcast did not recover target type")
    return reduced


def _expand_sum_adjoint(
    function: Function,
    upstream: Value,
    input_type: TensorType,
    axis: Any,
) -> Value:
    normalized = normalize_sum_axes(input_type, axis)
    if normalized is None:
        reduced_axes = tuple(range(len(input_type.shape)))
    elif isinstance(normalized, int):
        reduced_axes = (normalized,)
    else:
        reduced_axes = normalized

    reduced_set = set(reduced_axes)
    keep_shape = tuple(
        1 if position in reduced_set else dim
        for position, dim in enumerate(input_type.shape)
    )
    expanded = upstream
    if expanded.type.shape != keep_shape:
        expanded = _reshape(function, expanded, keep_shape)
    return _multiply(function, expanded, _ones(function, input_type))


def _constant(function: Function, value: np.ndarray[Any, Any]) -> Value:
    array = np.array(value, copy=True)
    dtype = DType.from_numpy(array.dtype)
    type_ = TensorType(tuple(array.shape), dtype)
    op = function.add_op("const", result_types=(type_,), attrs={"value": array})
    return op.results[0]


def _ones(function: Function, type_: TensorType) -> Value:
    return _constant(function, np.ones(type_.shape, dtype=type_.dtype.to_numpy()))


def _zeros(function: Function, type_: TensorType) -> Value:
    return _constant(function, np.zeros(type_.shape, dtype=type_.dtype.to_numpy()))


def _add(function: Function, lhs: Value, rhs: Value) -> Value:
    result_type = infer_binary(lhs.type, rhs.type)
    op = function.add_op("add", operands=(lhs, rhs), result_types=(result_type,))
    return op.results[0]


def _multiply(function: Function, lhs: Value, rhs: Value) -> Value:
    result_type = infer_binary(lhs.type, rhs.type)
    op = function.add_op("mul", operands=(lhs, rhs), result_types=(result_type,))
    return op.results[0]


def _sum(function: Function, value: Value, axis: int | tuple[int, ...] | None) -> Value:
    result_type = infer_sum(value.type, axis)
    attrs = {} if axis is None else {"axis": axis}
    op = function.add_op("sum", operands=(value,), result_types=(result_type,), attrs=attrs)
    return op.results[0]


def _reshape(function: Function, value: Value, shape: tuple[int, ...]) -> Value:
    result_type = infer_reshape(value.type, shape)
    op = function.add_op("reshape", operands=(value,), result_types=(result_type,))
    return op.results[0]
