from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .inference import (
    infer_binary,
    infer_reshape,
    infer_reverse,
    infer_slice,
    infer_sum,
    infer_transpose,
    normalize_sum_axes,
)
from .ir import DType, Function, Module, Operation, TensorType, Value
from .verifier import verify


class AutodiffError(ValueError):
    """Raised when a module is outside the bounded reverse-mode autodiff contract."""


_SUPPORTED_BACKWARD_OPS = frozenset(
    {
        "input",
        "const",
        "add",
        "mul",
        "sum",
        "reshape",
        "view",
        "slice",
        "reverse",
        "transpose",
        "copy_into",
    }
)
_FLOAT_DTYPES = frozenset({DType.FLOAT32, DType.FLOAT64})


def differentiate_module(
    module: Module,
    *,
    output_index: int = 0,
    wrt: Sequence[int] = (0,),
) -> Module:
    """Differentiate one static scalar floating output with respect to runtime inputs.

    The returned verified module keeps the original runtime-input ABI and returns gradients
    in requested wrt order. The bounded contract supports arithmetic plus shape/alias
    transforms with verifier-backed inverse or scatter VJP semantics.
    """
    return _reverse_mode_module(
        module,
        output_index=output_index,
        wrt=wrt,
        runtime_cotangent=False,
    )


def vector_jacobian_product_module(
    module: Module,
    *,
    output_index: int = 0,
    wrt: Sequence[int] = (0,),
) -> Module:
    """Build a runtime-seeded VJP for one static floating output.

    The returned verified module preserves all original runtime inputs and appends one
    cotangent input whose type exactly matches the selected output. Results are the
    vector-Jacobian product components for requested wrt inputs in order.
    """
    return _reverse_mode_module(
        module,
        output_index=output_index,
        wrt=wrt,
        runtime_cotangent=True,
    )


def _reverse_mode_module(
    module: Module,
    *,
    output_index: int,
    wrt: Sequence[int],
    runtime_cotangent: bool,
) -> Module:
    if not isinstance(module, Module):
        raise TypeError("reverse-mode autodiff requires a Module")
    verify(module)

    return_op = _terminal_return(module)
    selected_output = _select_output(
        return_op,
        output_index,
        require_scalar=not runtime_cotangent,
    )
    input_ops = _input_ops_by_index(module)
    requested = _normalize_wrt(wrt, input_ops)
    ancestors = _collect_ancestors(selected_output)

    _validate_static_floating_contract(selected_output, requested, input_ops, ancestors)
    _validate_backward_slice(ancestors, selected_output.type.dtype)

    suffix = "vjp" if runtime_cotangent else "grad"
    function = Function(f"{module.function.name}_{suffix}")
    value_map: dict[Value, Value] = {}
    cloned_forward_ops: list[Operation] = []

    if runtime_cotangent:
        for op in module.function.ops:
            if op.opcode != "input":
                continue
            cloned_forward_ops.append(_clone_op(function, op, value_map))

        cotangent = function.add_op(
            "input",
            result_types=(selected_output.type,),
            attrs={"index": len(input_ops)},
        ).results[0]

        for op in module.function.ops:
            if op.opcode in {"input", "return"}:
                continue
            if not any(result in ancestors for result in op.results):
                continue
            cloned_forward_ops.append(_clone_op(function, op, value_map))
        seed = cotangent
    else:
        for op in module.function.ops:
            if op.opcode == "return":
                continue
            include = op.opcode == "input" or any(
                result in ancestors for result in op.results
            )
            if not include:
                continue
            cloned_forward_ops.append(_clone_op(function, op, value_map))
        seed = _constant(
            function,
            np.array(1, dtype=selected_output.type.dtype.to_numpy()),
        )

    try:
        cloned_output = value_map[selected_output]
    except KeyError as exc:  # pragma: no cover - guarded by ancestor collection
        raise RuntimeError("internal autodiff error: selected output was not cloned") from exc

    gradients: dict[Value, Value] = {cloned_output: seed}

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
    transformed = Module(function)
    verify(transformed)
    return transformed

def _terminal_return(module: Module) -> Operation:
    returns = [op for op in module.function.ops if op.opcode == "return"]
    if len(returns) != 1:
        raise AutodiffError("autodiff requires exactly one terminal return operation")
    return returns[0]


def _select_output(
    return_op: Operation,
    output_index: int,
    *,
    require_scalar: bool = True,
) -> Value:
    if not isinstance(output_index, int) or isinstance(output_index, bool):
        raise AutodiffError("output index must be an integer")
    if output_index < 0 or output_index >= len(return_op.operands):
        raise AutodiffError(
            f"output index {output_index} is out of range for {len(return_op.operands)} outputs"
        )
    output = return_op.operands[output_index]
    if require_scalar and output.type.shape:
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
    output: Value,
    requested: tuple[int, ...],
    inputs: dict[int, Operation],
    ancestors: frozenset[Value],
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
    if any(not value.type.is_static for value in ancestors):
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


def _validate_backward_slice(ancestors: frozenset[Value], output_dtype: DType) -> None:
    producers = {value.producer for value in ancestors if value.producer is not None}
    for op in producers:
        if op.opcode not in _SUPPORTED_BACKWARD_OPS:
            raise AutodiffError(
                f"unsupported {op.opcode!r} operation on reverse-mode backward slice"
            )
        if op.opcode == "copy_into":
            _direct_slice_copy_attrs(op)
            _validate_copy_into_prewrite_isolation(op, ancestors)
        if len(op.results) != 1:
            raise AutodiffError(
                f"unsupported {op.opcode!r} multi-result operation on backward slice"
            )
        result_dtype = op.results[0].type.dtype
        if result_dtype not in _FLOAT_DTYPES:
            raise AutodiffError("reverse-mode backward slice must use floating tensor values")
        if result_dtype != output_dtype:
            raise AutodiffError(
                "mixed-precision reverse-mode autodiff is not supported; "
                "the backward slice must use one exact floating dtype"
            )


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
    if op.opcode in {"reshape", "view"}:
        (operand,) = op.operands
        contribution = _reshape(function, upstream, operand.type.shape)
        _accumulate(function, gradients, operand, contribution)
        return
    if op.opcode == "transpose":
        (operand,) = op.operands
        inverse = _inverse_permutation(op.attrs["axes"])
        contribution = _transpose(function, upstream, inverse)
        _accumulate(function, gradients, operand, contribution)
        return
    if op.opcode == "reverse":
        (operand,) = op.operands
        contribution = _reverse(function, upstream, op.attrs["axis"])
        _accumulate(function, gradients, operand, contribution)
        return
    if op.opcode == "slice":
        (operand,) = op.operands
        contribution = _scatter_slice(function, upstream, operand.type, op.attrs)
        _accumulate(function, gradients, operand, contribution)
        return
    if op.opcode == "copy_into":
        root, _target, source = op.operands
        attrs = _direct_slice_copy_attrs(op)
        root_contribution = _zero_slice_region(function, upstream, root.type, attrs)
        source_contribution = _unbroadcast(
            function,
            _slice(
                function,
                upstream,
                axis=attrs["axis"],
                start=attrs["start"],
                stop=attrs["stop"],
                step=attrs["step"],
            ),
            source.type,
        )
        _accumulate(function, gradients, root, root_contribution)
        _accumulate(function, gradients, source, source_contribution)
        return
    raise RuntimeError(f"internal autodiff error: unsupported propagated opcode {op.opcode!r}")


def _validate_copy_into_prewrite_isolation(
    op: Operation,
    ancestors: frozenset[Value],
) -> None:
    root, target, _source = op.operands
    result = op.results[0]
    for value in ancestors:
        if value in {root, target, result}:
            continue
        if _value_depends_on(value, result):
            continue
        if _value_depends_on(value, root):
            raise AutodiffError(
                "copy_into backward currently requires the pre-write root to be "
                "isolated from source and other ancestor paths"
            )


def _value_depends_on(value: Value, ancestor: Value) -> bool:
    stack = [value]
    seen: set[Value] = set()
    while stack:
        current = stack.pop()
        if current is ancestor:
            return True
        if current in seen:
            continue
        seen.add(current)
        producer = current.producer
        if producer is not None:
            stack.extend(producer.operands)
    return False


def _direct_slice_copy_attrs(op: Operation) -> dict[str, Any]:
    if op.opcode != "copy_into":
        raise RuntimeError("internal autodiff error: expected copy_into operation")
    root, target, _source = op.operands
    producer = target.producer
    if (
        producer is None
        or producer.opcode != "slice"
        or len(producer.operands) != 1
        or producer.operands[0] is not root
    ):
        raise AutodiffError(
            "copy_into backward currently requires a direct slice target"
        )
    return dict(producer.attrs)


def _zero_slice_region(
    function: Function,
    upstream: Value,
    root_type: TensorType,
    attrs: dict[str, Any],
) -> Value:
    if upstream.type != root_type:
        raise RuntimeError(
            "internal autodiff error: copy_into output cotangent must match root type"
        )
    copied_root = _reshape(function, upstream, root_type.shape)
    target = _slice(
        function,
        copied_root,
        axis=attrs["axis"],
        start=attrs["start"],
        stop=attrs["stop"],
        step=attrs["step"],
    )
    op = function.add_op(
        "copy_into",
        operands=(copied_root, target, _zeros(function, target.type)),
        result_types=(root_type,),
    )
    return op.results[0]


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


def _inverse_permutation(axes: tuple[int, ...]) -> tuple[int, ...]:
    inverse = [0] * len(axes)
    for output_axis, input_axis in enumerate(axes):
        inverse[input_axis] = output_axis
    return tuple(inverse)


def _transpose(function: Function, value: Value, axes: tuple[int, ...]) -> Value:
    result_type = infer_transpose(value.type, axes)
    op = function.add_op(
        "transpose",
        operands=(value,),
        result_types=(result_type,),
        attrs={"axes": axes},
    )
    return op.results[0]


def _reverse(function: Function, value: Value, axis: int) -> Value:
    result_type = infer_reverse(value.type, axis)
    op = function.add_op(
        "reverse",
        operands=(value,),
        result_types=(result_type,),
        attrs={"axis": axis},
    )
    return op.results[0]


def _slice(
    function: Function,
    value: Value,
    *,
    axis: int,
    start: int,
    stop: int,
    step: int,
) -> Value:
    result_type = infer_slice(
        value.type,
        axis=axis,
        start=start,
        stop=stop,
        step=step,
    )
    op = function.add_op(
        "slice",
        operands=(value,),
        result_types=(result_type,),
        attrs={"axis": axis, "start": start, "stop": stop, "step": step},
    )
    return op.results[0]


def _materialized_zeros(function: Function, type_: TensorType) -> Value:
    zero = _zeros(function, type_)
    return _add(function, zero, zero)


def _scatter_slice(
    function: Function,
    upstream: Value,
    input_type: TensorType,
    attrs: dict[str, Any],
) -> Value:
    root = _materialized_zeros(function, input_type)
    target = _slice(
        function,
        root,
        axis=attrs["axis"],
        start=attrs["start"],
        stop=attrs["stop"],
        step=attrs["step"],
    )
    if target.type != upstream.type:
        raise RuntimeError(
            "internal autodiff error: slice adjoint source type does not match target"
        )
    op = function.add_op(
        "copy_into",
        operands=(root, target, upstream),
        result_types=(input_type,),
    )
    return op.results[0]
