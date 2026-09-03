from __future__ import annotations

import numpy as np

from ..lowering import BufferAlloc, BufferKernel, BufferReturn, CPUProgram, plan_memory


def execute(program: CPUProgram) -> np.ndarray:
    """Execute verified buffer IR using a liveness-based physical memory plan."""
    plan = plan_memory(program)
    physical_buffers = [
        np.empty(buffer_type.shape, dtype=buffer_type.dtype.to_numpy())
        for buffer_type in plan.physical_types
    ]
    slots = {assignment.virtual: assignment.physical for assignment in plan.assignments}

    for op in program.operations:
        if isinstance(op, BufferAlloc):
            continue

        if isinstance(op, BufferReturn):
            return np.array(physical_buffers[slots[op.buffer]], copy=True)

        if not isinstance(op, BufferKernel):
            raise TypeError("unsupported CPU buffer operation")

        output = physical_buffers[slots[op.output]]
        if op.opcode == "const":
            if op.literal is None:
                raise RuntimeError("verified const kernel unexpectedly has no literal")
            output[...] = op.literal
        elif op.opcode in {"add", "mul"}:
            lhs = physical_buffers[slots[op.inputs[0]]]
            rhs = physical_buffers[slots[op.inputs[1]]]
            fn = np.add if op.opcode == "add" else np.multiply
            fn(lhs, rhs, out=output)
        elif op.opcode == "relu":
            operand = physical_buffers[slots[op.inputs[0]]]
            np.maximum(operand, np.array(0, dtype=output.dtype), out=output)
        else:
            raise RuntimeError(f"unsupported CPU kernel: {op.opcode}")

    raise RuntimeError("verified buffer IR unexpectedly has no return")
