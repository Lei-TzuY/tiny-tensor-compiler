from __future__ import annotations

import numpy as np

from ..loop_ir import LoopAlloc, LoopKernel, LoopProgram, LoopReturn, lower_to_loops
from ..lowering import CPUProgram


def execute(program: CPUProgram) -> np.ndarray:
    """Lower verified buffer IR to explicit loops and execute them on the CPU."""
    return execute_loop(lower_to_loops(program))


def execute_loop(program: LoopProgram) -> np.ndarray:
    """Execute explicit loop IR over planned physical NumPy buffers."""
    buffers: dict[int, np.ndarray] = {}

    for op in program.operations:
        if isinstance(op, LoopAlloc):
            buffers[op.buffer] = np.empty(op.type.shape, dtype=op.type.dtype.to_numpy())
            continue

        if isinstance(op, LoopReturn):
            return np.array(buffers[op.buffer], copy=True)

        if not isinstance(op, LoopKernel):
            raise TypeError("unsupported CPU loop operation")

        output = buffers[op.output]
        for output_index in np.ndindex(op.iteration_shape):
            if op.opcode == "const":
                if op.literal is None:
                    raise RuntimeError("verified const loop unexpectedly has no literal")
                output[output_index] = op.literal[output_index]
                continue

            inputs = tuple(
                buffers[buffer][index_map.apply(output_index)]
                for buffer, index_map in zip(op.inputs, op.input_maps, strict=True)
            )
            if op.opcode == "add":
                output[output_index] = np.add(inputs[0], inputs[1])
            elif op.opcode == "mul":
                output[output_index] = np.multiply(inputs[0], inputs[1])
            elif op.opcode == "relu":
                zero = np.array(0, dtype=output.dtype)
                output[output_index] = np.maximum(inputs[0], zero)
            elif op.opcode in {"relu_add", "relu_mul"}:
                binary = np.add if op.opcode == "relu_add" else np.multiply
                value = output.dtype.type(binary(inputs[0], inputs[1]))
                zero = np.array(0, dtype=output.dtype)
                output[output_index] = np.maximum(value, zero)
            else:
                raise RuntimeError(f"unsupported CPU loop kernel: {op.opcode}")

    raise RuntimeError("verified loop IR unexpectedly has no return")
