from __future__ import annotations

import numpy as np

from .ir import Module
from .verifier import verify


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
