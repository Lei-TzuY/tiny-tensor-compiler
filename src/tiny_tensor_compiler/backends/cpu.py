from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..input_validation import prepare_runtime_inputs
from ..loop_ir import (
    LoopAlloc,
    LoopInput,
    LoopKernel,
    LoopProgram,
    LoopReturn,
    lower_to_loops,
)
from ..lowering import CPUProgram


_BINARY_CHAIN_FUNCTIONS = {
    "chain_add_add": (np.add, np.add),
    "chain_add_mul": (np.add, np.multiply),
    "chain_mul_add": (np.multiply, np.add),
    "chain_mul_mul": (np.multiply, np.multiply),
}


def execute(program: CPUProgram, inputs: Sequence[Any] = ()) -> np.ndarray:
    """Lower verified buffer IR to explicit loops and execute them on the CPU."""
    return execute_loop(lower_to_loops(program), inputs=inputs)


def execute_loop(program: LoopProgram, inputs: Sequence[Any] = ()) -> np.ndarray:
    """Execute explicit loop IR over planned physical NumPy buffers."""
    runtime_inputs = prepare_runtime_inputs(program.input_types, inputs)
    buffers: dict[int, np.ndarray] = {}

    for op in program.operations:
        if isinstance(op, LoopAlloc):
            buffers[op.buffer] = np.empty(op.type.shape, dtype=op.type.dtype.to_numpy())
            continue

        if isinstance(op, LoopInput):
            np.copyto(buffers[op.output], runtime_inputs[op.index])
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

            values = tuple(
                buffers[buffer][index_map.apply(output_index)]
                for buffer, index_map in zip(op.inputs, op.input_maps, strict=True)
            )
            if op.opcode == "add":
                output[output_index] = np.add(values[0], values[1])
            elif op.opcode == "mul":
                output[output_index] = np.multiply(values[0], values[1])
            elif op.opcode == "relu":
                zero = np.array(0, dtype=output.dtype)
                output[output_index] = np.maximum(values[0], zero)
            elif op.opcode in {"relu_add", "relu_mul"}:
                binary = np.add if op.opcode == "relu_add" else np.multiply
                value = output.dtype.type(binary(values[0], values[1]))
                zero = np.array(0, dtype=output.dtype)
                output[output_index] = np.maximum(value, zero)
            elif op.opcode in _BINARY_CHAIN_FUNCTIONS:
                inner_fn, outer_fn = _BINARY_CHAIN_FUNCTIONS[op.opcode]
                inner = output.dtype.type(inner_fn(values[0], values[1]))
                output[output_index] = outer_fn(inner, values[2])
            else:
                raise RuntimeError(f"unsupported CPU loop kernel: {op.opcode}")

    raise RuntimeError("verified loop IR unexpectedly has no return")
