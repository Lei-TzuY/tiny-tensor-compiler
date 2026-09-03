from __future__ import annotations

import numpy as np

from .ir import DType, Module, Operation, Value
from .verifier import verify

_PURE_OPCODES = frozenset({"const", "add", "mul", "relu"})
_CSE_OPCODES = frozenset({"add", "mul", "relu"})


def constant_fold(module: Module) -> int:
    """Fold constant add/mul/relu operations in place and return the number folded."""
    verify(module)
    function = module.function
    folded = 0

    for op in list(function.ops):
        if op.opcode not in {"add", "mul", "relu"}:
            continue
        producers = [operand.producer for operand in op.operands]
        if any(producer is None or producer.opcode != "const" for producer in producers):
            continue

        inputs = [np.asarray(producer.attrs["value"]) for producer in producers if producer]
        result_type = op.results[0].type
        dtype = result_type.dtype.to_numpy()
        if op.opcode == "add":
            folded_value = np.add(inputs[0].astype(dtype), inputs[1].astype(dtype))
        elif op.opcode == "mul":
            folded_value = np.multiply(inputs[0].astype(dtype), inputs[1].astype(dtype))
        else:
            folded_value = np.maximum(inputs[0].astype(dtype), np.array(0, dtype=dtype))
        folded_value = np.asarray(folded_value, dtype=dtype)

        index = function.ops.index(op)
        replacement = function.insert_op(
            index,
            "const",
            result_types=[result_type],
            attrs={"value": np.array(folded_value, copy=True)},
        )
        op.results[0].replace_all_uses_with(replacement.results[0])
        function.erase_op(op)
        folded += 1

    verify(module)
    return folded


def algebraic_simplify(module: Module) -> int:
    """Remove exact integer add-zero and multiply-one identities in place."""
    verify(module)
    function = module.function
    simplified = 0

    for op in list(function.ops):
        replacement = _neutral_replacement(op)
        if replacement is None:
            continue

        op.results[0].replace_all_uses_with(replacement)
        function.erase_op(op)
        simplified += 1

    verify(module)
    return simplified


def dead_code_eliminate(module: Module) -> int:
    """Erase unused known-pure operations and return the number removed."""
    verify(module)
    function = module.function
    removed = 0

    while True:
        changed = False
        for op in reversed(list(function.ops)):
            if op.opcode not in _PURE_OPCODES:
                continue
            if not op.results or any(result.uses for result in op.results):
                continue

            function.erase_op(op)
            removed += 1
            changed = True

        if not changed:
            break

    verify(module)
    return removed


def common_subexpression_eliminate(module: Module) -> int:
    """Merge repeated exact pure expressions and return the number removed."""
    verify(module)
    function = module.function
    seen: dict[tuple[object, ...], Operation] = {}
    removed = 0

    for op in list(function.ops):
        if op.opcode not in _CSE_OPCODES or op.attrs or not op.results:
            continue

        key = (
            op.opcode,
            tuple(op.operands),
            tuple(result.type for result in op.results),
        )
        canonical = seen.get(key)
        if canonical is None:
            seen[key] = op
            continue

        for duplicate_result, canonical_result in zip(op.results, canonical.results):
            duplicate_result.replace_all_uses_with(canonical_result)
        function.erase_op(op)
        removed += 1

    verify(module)
    return removed


def _neutral_replacement(op: Operation) -> Value | None:
    if op.opcode not in {"add", "mul"} or len(op.results) != 1:
        return None

    result_type = op.results[0].type
    if result_type.dtype not in {DType.INT32, DType.INT64}:
        return None

    neutral = 0 if op.opcode == "add" else 1
    for neutral_index, operand in enumerate(op.operands):
        producer = operand.producer
        if producer is None or producer.opcode != "const":
            continue
        if not np.all(np.asarray(producer.attrs["value"]) == neutral):
            continue

        replacement = op.operands[1 - neutral_index]
        if replacement.type != result_type:
            continue
        return replacement

    return None
