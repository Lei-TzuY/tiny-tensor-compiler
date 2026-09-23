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
        "binary_into",
        "binary_inplace",
    }
)
_SUPPORTED_FORWARD_OPS = frozenset(
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
        "binary_into",
        "binary_inplace",
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


def jacobian_vector_product_module(
    module: Module,
    *,
    output_index: int = 0,
    wrt: Sequence[int] = (0,),
) -> Module:
    """Build a runtime-seeded forward-mode JVP for one static floating output.

    The returned verified module preserves all original runtime inputs, appends one
    exact-type tangent input for each requested wrt input in wrt order, and returns
    the tangent of the selected output.
    """
    return _forward_mode_module(
        module,
        output_index=output_index,
        wrt=wrt,
        return_primal=False,
    )


def value_and_jacobian_vector_product_module(
    module: Module,
    *,
    output_index: int = 0,
    wrt: Sequence[int] = (0,),
) -> Module:
    """Build one joint primal-plus-JVP program for a static floating output.

    The transformed verified module preserves the JVP runtime ABI but returns the
    selected primal output and its tangent from one shared forward clone.
    """
    return _forward_mode_module(
        module,
        output_index=output_index,
        wrt=wrt,
        return_primal=True,
    )


def _pushforward_linearization_modules(
    module: Module,
    *,
    output_index: int = 0,
    wrt: Sequence[int] = (0,),
) -> tuple[Module, Module, int]:
    """Split one static pure JVP into a one-shot primal tape and reusable pushforward."""
    if not isinstance(module, Module):
        raise TypeError("reusable pushforward linearization requires a Module")
    verify(module)

    return_op = _terminal_return(module)
    selected_output = _select_output(
        return_op,
        output_index,
        require_scalar=False,
    )
    input_ops = _input_ops_by_index(module)
    requested = _normalize_wrt(wrt, input_ops)
    ancestors = _collect_ancestors(selected_output)

    _validate_static_floating_contract(selected_output, requested, input_ops, ancestors)
    _validate_reusable_linearization_slice(
        ancestors,
        selected_output.type.dtype,
        context="pushforward",
    )

    tape_values = _collect_reusable_linearization_tape_values(module, ancestors)

    primal_function = Function(f"{module.function.name}_linearize_primal")
    primal_map: dict[Value, Value] = {}
    for op in module.function.ops:
        if op.opcode == "return":
            continue
        include = op.opcode == "input" or any(
            result in ancestors for result in op.results
        )
        if include:
            _clone_op(primal_function, op, primal_map)

    primal_output = primal_map.get(selected_output)
    if primal_output is None:
        raise RuntimeError(
            "internal autodiff error: reusable linearization output was not cloned"
        )
    primal_tape_outputs = [primal_output]
    for value in tape_values:
        try:
            primal_tape_outputs.append(primal_map[value])
        except KeyError as exc:
            raise RuntimeError(
                "internal autodiff error: reusable linearization tape value was not cloned"
            ) from exc
    primal_function.add_op("return", operands=primal_tape_outputs)
    primal_module = Module(primal_function)
    verify(primal_module)

    push_function = Function(f"{module.function.name}_pushforward")
    primal_values: dict[Value, Value] = {}
    tangents: dict[Value, Value] = {}

    ordered_inputs = tuple(input_ops[index] for index in sorted(input_ops))
    for op in ordered_inputs:
        original = op.results[0]
        index = op.attrs["index"]
        primal_values[original] = push_function.add_op(
            "input",
            result_types=(original.type,),
            attrs={"index": index},
        ).results[0]

    next_input_index = len(ordered_inputs)
    for offset, value in enumerate(tape_values):
        primal_values[value] = push_function.add_op(
            "input",
            result_types=(value.type,),
            attrs={"index": next_input_index + offset},
        ).results[0]

    tangent_input_index = next_input_index + len(tape_values)
    for offset, input_index in enumerate(requested):
        original_input = input_ops[input_index].results[0]
        tangents[original_input] = push_function.add_op(
            "input",
            result_types=(original_input.type,),
            attrs={"index": tangent_input_index + offset},
        ).results[0]

    for op in ordered_inputs:
        original_input = op.results[0]
        if original_input not in tangents:
            tangents[original_input] = _zeros(push_function, original_input.type)

    for op in module.function.ops:
        if op.opcode in {"input", "return"}:
            continue
        if not any(result in ancestors for result in op.results):
            continue
        result = op.results[0]
        if op.opcode == "const":
            _clone_op(push_function, op, primal_values)
            tangents[result] = _zeros(push_function, result.type)
            continue

        if op.opcode == "add":
            lhs, rhs = op.operands
            tangents[result] = _add(
                push_function,
                tangents[lhs],
                tangents[rhs],
            )
            continue

        if op.opcode == "mul":
            lhs, rhs = op.operands
            try:
                lhs_primal = primal_values[lhs]
                rhs_primal = primal_values[rhs]
            except KeyError as exc:
                raise RuntimeError(
                    "internal autodiff error: reusable pushforward is missing a primal tape value"
                ) from exc
            tangents[result] = _add(
                push_function,
                _multiply(push_function, tangents[lhs], rhs_primal),
                _multiply(push_function, lhs_primal, tangents[rhs]),
            )
            continue

        if op.opcode in {
            "sum",
            "reshape",
            "view",
            "slice",
            "reverse",
            "transpose",
        }:
            (operand,) = op.operands
            tangents[result] = push_function.add_op(
                op.opcode,
                operands=(tangents[operand],),
                result_types=(result.type,),
                attrs=dict(op.attrs),
            ).results[0]
            continue

        raise RuntimeError(
            f"internal autodiff error: unsupported reusable pushforward opcode {op.opcode!r}"
        )

    tangent_output = tangents.get(selected_output)
    if tangent_output is None:
        raise RuntimeError(
            "internal autodiff error: reusable pushforward output has no tangent"
        )
    push_function.add_op("return", operands=(tangent_output,))
    pushforward_module = Module(push_function)
    verify(pushforward_module)
    return primal_module, pushforward_module, len(tape_values)


def _validate_reusable_linearization_slice(
    ancestors: frozenset[Value],
    output_dtype: DType,
    *,
    context: str,
) -> None:
    producers = {value.producer for value in ancestors if value.producer is not None}
    for op in producers:
        if op.opcode in {"copy_into", "binary_into", "binary_inplace"}:
            raise AutodiffError(
                f"reusable {context} linearization does not yet support write effects"
            )
        if op.opcode not in {
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
        }:
            raise AutodiffError(
                f"unsupported {op.opcode!r} operation on reusable {context} slice"
            )
        if len(op.results) != 1:
            raise AutodiffError(
                f"unsupported {op.opcode!r} multi-result operation on reusable {context} slice"
            )
        result_dtype = op.results[0].type.dtype
        if result_dtype not in _FLOAT_DTYPES:
            raise AutodiffError(
                f"reusable {context} linearization must use floating tensor values"
            )
        if result_dtype != output_dtype:
            raise AutodiffError(
                f"mixed-precision reusable {context} linearization is not supported"
            )


def _collect_reusable_linearization_tape_values(
    module: Module,
    ancestors: frozenset[Value],
) -> tuple[Value, ...]:
    needed: set[Value] = set()
    producers = {value.producer for value in ancestors if value.producer is not None}
    for op in module.function.ops:
        if op not in producers or op.opcode != "mul":
            continue
        for operand in op.operands:
            producer = operand.producer
            if producer is None or producer.opcode in {"input", "const"}:
                continue
            needed.add(operand)

    ordered: list[Value] = []
    for op in module.function.ops:
        for result in op.results:
            if result in needed:
                ordered.append(result)
    return tuple(ordered)


def _pullback_linearization_modules(
    module: Module,
    *,
    output_index: int = 0,
    wrt: Sequence[int] = (0,),
) -> tuple[Module, Module, int]:
    """Split one static pure VJP into a one-shot primal tape and reusable pullback."""
    if not isinstance(module, Module):
        raise TypeError("reusable pullback linearization requires a Module")
    verify(module)

    return_op = _terminal_return(module)
    selected_output = _select_output(
        return_op,
        output_index,
        require_scalar=False,
    )
    input_ops = _input_ops_by_index(module)
    requested = _normalize_wrt(wrt, input_ops)
    ancestors = _collect_ancestors(selected_output)

    _validate_static_floating_contract(selected_output, requested, input_ops, ancestors)
    _validate_reusable_linearization_slice(
        ancestors,
        selected_output.type.dtype,
        context="pullback",
    )
    tape_values = _collect_reusable_linearization_tape_values(module, ancestors)

    primal_function = Function(f"{module.function.name}_linearize_primal")
    primal_map: dict[Value, Value] = {}
    for op in module.function.ops:
        if op.opcode == "return":
            continue
        include = op.opcode == "input" or any(
            result in ancestors for result in op.results
        )
        if include:
            _clone_op(primal_function, op, primal_map)

    primal_output = primal_map.get(selected_output)
    if primal_output is None:
        raise RuntimeError(
            "internal autodiff error: reusable pullback output was not cloned"
        )
    primal_tape_outputs = [primal_output]
    for value in tape_values:
        try:
            primal_tape_outputs.append(primal_map[value])
        except KeyError as exc:
            raise RuntimeError(
                "internal autodiff error: reusable pullback tape value was not cloned"
            ) from exc
    primal_function.add_op("return", operands=primal_tape_outputs)
    primal_module = Module(primal_function)
    verify(primal_module)

    pull_function = Function(f"{module.function.name}_pullback")
    primal_values: dict[Value, Value] = {}

    ordered_inputs = tuple(input_ops[index] for index in sorted(input_ops))
    for op in ordered_inputs:
        original = op.results[0]
        index = op.attrs["index"]
        primal_values[original] = pull_function.add_op(
            "input",
            result_types=(original.type,),
            attrs={"index": index},
        ).results[0]

    next_input_index = len(ordered_inputs)
    for offset, value in enumerate(tape_values):
        primal_values[value] = pull_function.add_op(
            "input",
            result_types=(value.type,),
            attrs={"index": next_input_index + offset},
        ).results[0]

    for op in module.function.ops:
        if op.opcode != "const":
            continue
        if not any(result in ancestors for result in op.results):
            continue
        _clone_op(pull_function, op, primal_values)

    cotangent = pull_function.add_op(
        "input",
        result_types=(selected_output.type,),
        attrs={"index": next_input_index + len(tape_values)},
    ).results[0]

    gradients: dict[Value, Value] = {selected_output: cotangent}
    for op in reversed(module.function.ops):
        if op.opcode == "return" or not op.results:
            continue
        result = op.results[0]
        if result not in ancestors:
            continue
        upstream = gradients.get(result)
        if upstream is None:
            continue
        _propagate_reusable_pullback_adjoint(
            pull_function,
            op,
            upstream,
            gradients,
            primal_values,
        )

    outputs: list[Value] = []
    for input_index in requested:
        original_input = input_ops[input_index].results[0]
        gradient = gradients.get(original_input)
        if gradient is None:
            gradient = _zeros(pull_function, original_input.type)
        if gradient.type != original_input.type:
            raise RuntimeError(
                "internal autodiff error: reusable pullback gradient type does not match input type"
            )
        outputs.append(gradient)

    pull_function.add_op("return", operands=outputs)
    pullback_module = Module(pull_function)
    verify(pullback_module)
    return primal_module, pullback_module, len(tape_values)


def _shared_linearization_modules(
    module: Module,
    *,
    output_index: int = 0,
    wrt: Sequence[int] = (0,),
) -> tuple[Module, Module, Module, int]:
    """Build one shared primal/tape layout with reusable pushforward and pullback programs."""
    if not isinstance(module, Module):
        raise TypeError("reusable shared linearization requires a Module")
    verify(module)

    return_op = _terminal_return(module)
    selected_output = _select_output(
        return_op,
        output_index,
        require_scalar=False,
    )
    input_ops = _input_ops_by_index(module)
    requested = _normalize_wrt(wrt, input_ops)
    ancestors = _collect_ancestors(selected_output)
    _validate_static_floating_contract(selected_output, requested, input_ops, ancestors)
    _validate_reusable_linearization_slice(
        ancestors,
        selected_output.type.dtype,
        context="shared",
    )

    primal_module, pushforward_module, tape_value_count = (
        _pushforward_linearization_modules(
            module,
            output_index=output_index,
            wrt=wrt,
        )
    )
    pullback_primal, pullback_module, pullback_tape_count = (
        _pullback_linearization_modules(
            module,
            output_index=output_index,
            wrt=wrt,
        )
    )
    if tape_value_count != pullback_tape_count:
        raise RuntimeError(
            "internal autodiff error: pushforward/pullback tape layouts disagree"
        )
    if primal_module.dump() != pullback_primal.dump():
        raise RuntimeError(
            "internal autodiff error: pushforward/pullback primal tape modules disagree"
        )
    return primal_module, pushforward_module, pullback_module, tape_value_count


def _propagate_reusable_pullback_adjoint(
    function: Function,
    op: Operation,
    upstream: Value,
    gradients: dict[Value, Value],
    primal_values: dict[Value, Value],
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
        try:
            lhs_primal = primal_values[lhs]
            rhs_primal = primal_values[rhs]
        except KeyError as exc:
            raise RuntimeError(
                "internal autodiff error: reusable pullback is missing a primal tape value"
            ) from exc
        _accumulate(
            function,
            gradients,
            lhs,
            _unbroadcast(
                function,
                _multiply(function, upstream, rhs_primal),
                lhs.type,
            ),
        )
        _accumulate(
            function,
            gradients,
            rhs,
            _unbroadcast(
                function,
                _multiply(function, upstream, lhs_primal),
                rhs.type,
            ),
        )
        return
    if op.opcode == "sum":
        (operand,) = op.operands
        _accumulate(
            function,
            gradients,
            operand,
            _expand_sum_adjoint(
                function,
                upstream,
                operand.type,
                op.attrs.get("axis"),
            ),
        )
        return
    if op.opcode in {"reshape", "view"}:
        (operand,) = op.operands
        _accumulate(
            function,
            gradients,
            operand,
            _reshape(function, upstream, operand.type.shape),
        )
        return
    if op.opcode == "transpose":
        (operand,) = op.operands
        _accumulate(
            function,
            gradients,
            operand,
            _transpose(function, upstream, _inverse_permutation(op.attrs["axes"])),
        )
        return
    if op.opcode == "reverse":
        (operand,) = op.operands
        _accumulate(
            function,
            gradients,
            operand,
            _reverse(function, upstream, op.attrs["axis"]),
        )
        return
    if op.opcode == "slice":
        (operand,) = op.operands
        _accumulate(
            function,
            gradients,
            operand,
            _scatter_slice(function, upstream, operand.type, op.attrs),
        )
        return
    raise RuntimeError(
        f"internal autodiff error: unsupported reusable pullback opcode {op.opcode!r}"
    )


def _forward_mode_module(
    module: Module,
    *,
    output_index: int,
    wrt: Sequence[int],
    return_primal: bool,
) -> Module:
    if not isinstance(module, Module):
        raise TypeError("forward-mode JVP requires a Module")
    verify(module)

    return_op = _terminal_return(module)
    selected_output = _select_output(
        return_op,
        output_index,
        require_scalar=False,
    )
    input_ops = _input_ops_by_index(module)
    requested = _normalize_wrt(wrt, input_ops)
    ancestors = _collect_ancestors(selected_output)

    _validate_static_floating_contract(selected_output, requested, input_ops, ancestors)
    _validate_forward_slice(ancestors, selected_output.type.dtype)

    suffix = "value_and_jvp" if return_primal else "jvp"
    function = Function(f"{module.function.name}_{suffix}")
    value_map: dict[Value, Value] = {}
    tangents: dict[Value, Value] = {}
    forward_primal_tape: dict[Value, Value] = {}

    for op in module.function.ops:
        if op.opcode != "input":
            continue
        _clone_op(function, op, value_map)

    tangent_input_index = len(input_ops)
    for offset, input_index in enumerate(requested):
        original_input = input_ops[input_index].results[0]
        tangents[original_input] = function.add_op(
            "input",
            result_types=(original_input.type,),
            attrs={"index": tangent_input_index + offset},
        ).results[0]

    for op in input_ops.values():
        original_input = op.results[0]
        if original_input not in tangents:
            tangents[original_input] = _zeros(function, original_input.type)

    for op in module.function.ops:
        if op.opcode in {"input", "return"}:
            continue
        if not any(result in ancestors for result in op.results):
            continue
        if (
            op.opcode in {"binary_into", "binary_inplace"}
            and op.attrs["operator"] == "mul"
        ):
            _capture_forward_prewrite_primal(
                function,
                op,
                value_map,
                forward_primal_tape,
            )
        cloned = _clone_op(function, op, value_map)
        original_result = op.results[0]
        tangents[original_result] = _forward_tangent(
            function,
            op,
            cloned,
            value_map,
            tangents,
            forward_primal_tape,
        )

    tangent_output = tangents.get(selected_output)
    if tangent_output is None:
        raise RuntimeError("internal autodiff error: selected JVP output has no tangent")
    if tangent_output.type != selected_output.type:
        raise RuntimeError("internal autodiff error: JVP output type does not match primal output")

    if return_primal:
        primal_output = value_map.get(selected_output)
        if primal_output is None:
            raise RuntimeError(
                "internal autodiff error: selected JVP primal output was not cloned"
            )
        function.add_op("return", operands=(primal_output, tangent_output))
    else:
        function.add_op("return", operands=(tangent_output,))

    transformed = Module(function)
    verify(transformed)
    return transformed

def _capture_forward_prewrite_primal(
    function: Function,
    op: Operation,
    value_map: dict[Value, Value],
    forward_primal_tape: dict[Value, Value],
) -> None:
    if op.opcode not in {"binary_into", "binary_inplace"} or op.attrs.get(
        "operator"
    ) != "mul":
        raise RuntimeError(
            "internal autodiff error: expected multiplicative write for forward primal tape"
        )
    primal = op.operands[1] if op.opcode == "binary_into" else op.operands[0]
    try:
        cloned_primal = value_map[primal]
    except KeyError as exc:
        raise RuntimeError(
            "internal autodiff error: pre-write primal was not cloned"
        ) from exc
    forward_primal_tape[primal] = _multiply(
        function,
        cloned_primal,
        _ones(function, cloned_primal.type),
    )


def _validate_forward_slice(
    ancestors: frozenset[Value],
    output_dtype: DType,
) -> None:
    producers = {value.producer for value in ancestors if value.producer is not None}
    for op in producers:
        if op.opcode not in _SUPPORTED_FORWARD_OPS:
            raise AutodiffError(
                f"unsupported {op.opcode!r} operation on forward-mode JVP slice"
            )
        if op.opcode in {"copy_into", "binary_into"}:
            _direct_slice_write_attrs(op, context="forward-mode JVP")
        if len(op.results) != 1:
            raise AutodiffError(
                f"unsupported {op.opcode!r} multi-result operation on forward-mode JVP slice"
            )
        result_dtype = op.results[0].type.dtype
        if result_dtype not in _FLOAT_DTYPES:
            raise AutodiffError("forward-mode JVP slice must use floating tensor values")
        if result_dtype != output_dtype:
            raise AutodiffError(
                "mixed-precision forward-mode JVP is not supported; "
                "the selected forward slice must use one exact floating dtype"
            )


def _forward_tangent(
    function: Function,
    original_op: Operation,
    cloned_op: Operation,
    value_map: dict[Value, Value],
    tangents: dict[Value, Value],
    forward_primal_tape: dict[Value, Value],
) -> Value:
    if original_op.opcode == "const":
        return _zeros(function, original_op.results[0].type)

    if original_op.opcode == "add":
        lhs, rhs = original_op.operands
        return _add(function, tangents[lhs], tangents[rhs])

    if original_op.opcode == "mul":
        lhs, rhs = original_op.operands
        lhs_primal = value_map[lhs]
        rhs_primal = value_map[rhs]
        lhs_term = _multiply(function, tangents[lhs], rhs_primal)
        rhs_term = _multiply(function, lhs_primal, tangents[rhs])
        return _add(function, lhs_term, rhs_term)

    if original_op.opcode in {
        "sum",
        "reshape",
        "view",
        "slice",
        "reverse",
        "transpose",
    }:
        (operand,) = original_op.operands
        tangent = tangents[operand]
        tangent_op = function.add_op(
            original_op.opcode,
            operands=(tangent,),
            result_types=(cloned_op.results[0].type,),
            attrs=dict(original_op.attrs),
        )
        return tangent_op.results[0]

    if original_op.opcode == "copy_into":
        root, target, source = original_op.operands
        tangent_op = function.add_op(
            "copy_into",
            operands=(tangents[root], tangents[target], tangents[source]),
            result_types=(cloned_op.results[0].type,),
        )
        return tangent_op.results[0]

    if original_op.opcode == "binary_inplace":
        root, source = original_op.operands
        if original_op.attrs["operator"] == "add":
            return _add(function, tangents[root], tangents[source])

        root_primal = forward_primal_tape.get(root)
        if root_primal is None:
            raise RuntimeError(
                "internal autodiff error: binary_inplace mul requires taped root primal"
            )
        return _add(
            function,
            _multiply(function, tangents[root], value_map[source]),
            _multiply(function, root_primal, tangents[source]),
        )

    if original_op.opcode == "binary_into":
        root, target, source = original_op.operands
        if original_op.attrs["operator"] == "add":
            tangent_op = function.add_op(
                "binary_into",
                operands=(tangents[root], tangents[target], tangents[source]),
                result_types=(cloned_op.results[0].type,),
                attrs={"operator": "add"},
            )
            return tangent_op.results[0]

        target_primal = forward_primal_tape.get(target)
        if target_primal is None:
            raise RuntimeError(
                "internal autodiff error: binary_into mul requires taped target primal"
            )
        replacement = _add(
            function,
            _multiply(function, tangents[target], value_map[source]),
            _multiply(function, target_primal, tangents[source]),
        )
        tangent_op = function.add_op(
            "copy_into",
            operands=(tangents[root], tangents[target], replacement),
            result_types=(cloned_op.results[0].type,),
        )
        return tangent_op.results[0]

    raise RuntimeError(
        f"internal autodiff error: unsupported forward tangent opcode {original_op.opcode!r}"
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
    primal_tape: dict[Value, Value] = {}
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
            if op.opcode in {"copy_into", "binary_into", "binary_inplace"}:
                _capture_prewrite_primal_tape(
                    function,
                    op,
                    ancestors,
                    value_map,
                    primal_tape,
                )
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
            if op.opcode in {"copy_into", "binary_into", "binary_inplace"}:
                _capture_prewrite_primal_tape(
                    function,
                    op,
                    ancestors,
                    value_map,
                    primal_tape,
                )
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
        _propagate_adjoint(function, op, upstream, gradients, primal_tape)

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
        if op.opcode in {"copy_into", "binary_into"}:
            _direct_slice_write_attrs(op)
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
    primal_tape: dict[Value, Value],
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
        lhs_primal = primal_tape.get(lhs, lhs)
        rhs_primal = primal_tape.get(rhs, rhs)
        lhs_contribution = _unbroadcast(
            function,
            _multiply(function, upstream, rhs_primal),
            lhs.type,
        )
        rhs_contribution = _unbroadcast(
            function,
            _multiply(function, upstream, lhs_primal),
            rhs.type,
        )
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
    if op.opcode == "binary_inplace":
        root, source = op.operands
        if op.attrs["operator"] == "add":
            root_contribution = upstream
            source_contribution = upstream
        else:
            root_primal = primal_tape.get(root)
            if root_primal is None:
                raise RuntimeError(
                    "internal autodiff error: binary_inplace mul requires taped root primal"
                )
            root_contribution = _multiply(function, upstream, source)
            source_contribution = _multiply(function, upstream, root_primal)
        _accumulate(function, gradients, root, root_contribution)
        _accumulate(function, gradients, source, source_contribution)
        return
    if op.opcode in {"copy_into", "binary_into"}:
        root, target, source = op.operands
        attrs = _direct_slice_write_attrs(op)
        region_upstream = _slice(
            function,
            upstream,
            axis=attrs["axis"],
            start=attrs["start"],
            stop=attrs["stop"],
            step=attrs["step"],
        )

        if op.opcode == "copy_into":
            root_contribution = _replace_slice_region(
                function,
                upstream,
                root.type,
                attrs,
                _zeros(function, target.type),
            )
            source_contribution = _unbroadcast(
                function,
                region_upstream,
                source.type,
            )
        elif op.attrs["operator"] == "add":
            root_contribution = upstream
            source_contribution = _unbroadcast(
                function,
                region_upstream,
                source.type,
            )
        else:
            target_primal = primal_tape.get(target)
            if target_primal is None:
                raise RuntimeError(
                    "internal autodiff error: binary_into mul requires taped target primal"
                )
            root_region = _multiply(function, region_upstream, source)
            root_contribution = _replace_slice_region(
                function,
                upstream,
                root.type,
                attrs,
                root_region,
            )
            source_contribution = _unbroadcast(
                function,
                _multiply(function, region_upstream, target_primal),
                source.type,
            )

        _accumulate(function, gradients, root, root_contribution)
        _accumulate(function, gradients, source, source_contribution)
        return
    raise RuntimeError(f"internal autodiff error: unsupported propagated opcode {op.opcode!r}")


def _capture_prewrite_primal_tape(
    function: Function,
    op: Operation,
    ancestors: frozenset[Value],
    value_map: dict[Value, Value],
    primal_tape: dict[Value, Value],
) -> None:
    root = op.operands[0]
    for original, cloned in tuple(value_map.items()):
        if original not in ancestors:
            continue
        if not _aliases_exact_root(original, root):
            continue
        if cloned in primal_tape:
            continue
        primal_tape[cloned] = _multiply(
            function,
            cloned,
            _ones(function, cloned.type),
        )


def _aliases_exact_root(value: Value, root: Value) -> bool:
    current = value
    seen: set[Value] = set()
    while True:
        if current is root:
            return True
        if current in seen:
            return False
        seen.add(current)
        producer = current.producer
        if producer is None or producer.opcode not in {
            "view",
            "slice",
            "reverse",
            "transpose",
        }:
            return False
        current = producer.operands[0]


def _direct_slice_write_attrs(
    op: Operation,
    *,
    context: str = "backward",
) -> dict[str, Any]:
    if op.opcode not in {"copy_into", "binary_into"}:
        raise RuntimeError("internal autodiff error: expected partial write operation")
    root, target, _source = op.operands
    producer = target.producer
    if (
        producer is None
        or producer.opcode != "slice"
        or len(producer.operands) != 1
        or producer.operands[0] is not root
    ):
        raise AutodiffError(
            f"{op.opcode} {context} currently requires a direct slice target"
        )
    return dict(producer.attrs)


def _replace_slice_region(
    function: Function,
    upstream: Value,
    root_type: TensorType,
    attrs: dict[str, Any],
    replacement: Value,
) -> Value:
    if upstream.type != root_type:
        raise RuntimeError(
            "internal autodiff error: write-effect output cotangent must match root type"
        )
    copied_root = _multiply(function, upstream, _ones(function, root_type))
    target = _slice(
        function,
        copied_root,
        axis=attrs["axis"],
        start=attrs["start"],
        stop=attrs["stop"],
        step=attrs["step"],
    )
    if replacement.type != target.type:
        raise RuntimeError(
            "internal autodiff error: replacement cotangent must match write target type"
        )
    op = function.add_op(
        "copy_into",
        operands=(copied_root, target, replacement),
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
