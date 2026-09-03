from __future__ import annotations

import numpy as np

from ..lowering import CPUProgram


def execute(program: CPUProgram) -> np.ndarray:
    """Execute deterministic lowered CPU instructions using NumPy kernels."""
    buffers: dict[int, np.ndarray] = {}

    for inst in program.instructions:
        dtype = inst.result_type.dtype.to_numpy()
        if inst.opcode == "const":
            if inst.literal is None:
                raise RuntimeError("const instruction is missing its literal")
            buffers[inst.output] = np.array(inst.literal, dtype=dtype, copy=True)
        elif inst.opcode in {"add", "mul"}:
            lhs = buffers[inst.inputs[0]].astype(dtype, copy=False)
            rhs = buffers[inst.inputs[1]].astype(dtype, copy=False)
            fn = np.add if inst.opcode == "add" else np.multiply
            buffers[inst.output] = np.asarray(fn(lhs, rhs), dtype=dtype)
        elif inst.opcode == "relu":
            operand = buffers[inst.inputs[0]].astype(dtype, copy=False)
            buffers[inst.output] = np.maximum(operand, np.array(0, dtype=dtype))
        else:
            raise RuntimeError(f"unsupported CPU instruction: {inst.opcode}")

    return np.array(buffers[program.return_slot], copy=True)
