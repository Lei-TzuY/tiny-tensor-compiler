from __future__ import annotations

import numpy as np

from ..lowering import BufferAlloc, BufferKernel, BufferReturn, CPUProgram


def execute(program: CPUProgram) -> np.ndarray:
    """Execute verified buffer IR using NumPy kernels over explicit allocations."""
    buffers: dict[int, np.ndarray] = {}

    for op in program.operations:
        if isinstance(op, BufferAlloc):
            buffers[op.buffer] = np.empty(op.type.shape, dtype=op.type.dtype.to_numpy())
            continue

        if isinstance(op, BufferReturn):
            return np.array(buffers[op.buffer], copy=True)

        if not isinstance(op, BufferKernel):
            raise TypeError("unsupported CPU buffer operation")

        output = buffers[op.output]
        if op.opcode == "const":
            if op.literal is None:
                raise RuntimeError("verified const kernel unexpectedly has no literal")
            output[...] = op.literal
        elif op.opcode in {"add", "mul"}:
            lhs = buffers[op.inputs[0]]
            rhs = buffers[op.inputs[1]]
            fn = np.add if op.opcode == "add" else np.multiply
            fn(lhs, rhs, out=output)
        elif op.opcode == "relu":
            operand = buffers[op.inputs[0]]
            np.maximum(operand, np.array(0, dtype=output.dtype), out=output)
        else:
            raise RuntimeError(f"unsupported CPU kernel: {op.opcode}")

    raise RuntimeError("verified buffer IR unexpectedly has no return")
