from __future__ import annotations

import numpy as np

from .ir import Module, Value
from .verifier import verify


def execute_reference(module: Module) -> np.ndarray:
    """Execute verified tensor IR directly; used as a semantic reference backend."""
    verify(module)
    values: dict[Value, np.ndarray] = {}

    for op in module.function.ops:
        if op.opcode == "const":
            values[op.results[0]] = np.array(op.attrs["value"], copy=True)
        elif op.opcode in {"add", "mul"}:
            dtype = op.results[0].type.dtype.to_numpy()
            lhs = values[op.operands[0]].astype(dtype, copy=False)
            rhs = values[op.operands[1]].astype(dtype, copy=False)
            fn = np.add if op.opcode == "add" else np.multiply
            values[op.results[0]] = np.asarray(fn(lhs, rhs), dtype=dtype)
        elif op.opcode == "relu":
            dtype = op.results[0].type.dtype.to_numpy()
            operand = values[op.operands[0]].astype(dtype, copy=False)
            values[op.results[0]] = np.maximum(operand, np.array(0, dtype=dtype))
        elif op.opcode == "return":
            return np.array(values[op.operands[0]], copy=True)
    raise RuntimeError("verified module unexpectedly has no return")
