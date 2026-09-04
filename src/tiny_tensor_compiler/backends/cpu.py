from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..input_binding import (
    BorrowedLoopProgram,
    borrow_inputs as bind_borrowed_inputs,
    borrowed_slots,
)
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

ExecutionResult = np.ndarray | tuple[np.ndarray, ...]
LoopExecutionProgram = LoopProgram | BorrowedLoopProgram

_BINARY_CHAIN_FUNCTIONS = {
    "chain_add_add": (np.add, np.add),
    "chain_add_mul": (np.add, np.multiply),
    "chain_mul_add": (np.multiply, np.add),
    "chain_mul_mul": (np.multiply, np.multiply),
}
_RELU_BINARY_CHAIN_OPCODES = frozenset(f"relu_{opcode}" for opcode in _BINARY_CHAIN_FUNCTIONS)
_BINARY_TREE_FUNCTIONS = {
    f"tree_{left}_{right}_{root}": (
        np.add if left == "add" else np.multiply,
        np.add if right == "add" else np.multiply,
        np.add if root == "add" else np.multiply,
    )
    for left in ("add", "mul")
    for right in ("add", "mul")
    for root in ("add", "mul")
}
_RELU_BINARY_TREE_OPCODES = frozenset(f"relu_{opcode}" for opcode in _BINARY_TREE_FUNCTIONS)
_CHAIN_TREE_FUNCTIONS = {
    f"chain_tree_{inner}_{left}_{right}_{root}": (
        np.add if inner == "add" else np.multiply,
        np.add if left == "add" else np.multiply,
        np.add if right == "add" else np.multiply,
        np.add if root == "add" else np.multiply,
    )
    for inner in ("add", "mul")
    for left in ("add", "mul")
    for right in ("add", "mul")
    for root in ("add", "mul")
}


def execute(
    program: CPUProgram,
    inputs: Sequence[Any] = (),
    *,
    borrow_inputs: bool = False,
) -> ExecutionResult:
    """Lower verified buffer IR and optionally borrow safe external input arrays."""
    loops: LoopExecutionProgram = lower_to_loops(program)
    if borrow_inputs:
        loops = bind_borrowed_inputs(loops)
    return execute_loop(loops, inputs=inputs)


def execute_loop(
    program: LoopExecutionProgram,
    inputs: Sequence[Any] = (),
) -> ExecutionResult:
    """Execute explicit loop IR over planned physical or borrowed NumPy buffers."""
    runtime_inputs = prepare_runtime_inputs(program.input_types, inputs)
    direct_slots = borrowed_slots(program)
    buffers: dict[int, np.ndarray] = {}
    return_buffers: list[int] = []

    for op in program.operations:
        if isinstance(op, LoopAlloc):
            if op.buffer not in direct_slots:
                buffers[op.buffer] = np.empty(op.type.shape, dtype=op.type.dtype.to_numpy())
            continue

        if isinstance(op, LoopInput):
            if op.output in direct_slots:
                buffers[op.output] = runtime_inputs[op.index]
            else:
                np.copyto(buffers[op.output], runtime_inputs[op.index])
            continue

        if isinstance(op, LoopReturn):
            return_buffers.append(op.buffer)
            continue

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
            elif op.opcode in _BINARY_CHAIN_FUNCTIONS or op.opcode in _RELU_BINARY_CHAIN_OPCODES:
                relu_chain = op.opcode in _RELU_BINARY_CHAIN_OPCODES
                chain_opcode = op.opcode.removeprefix("relu_")
                inner_fn, outer_fn = _BINARY_CHAIN_FUNCTIONS[chain_opcode]
                inner = output.dtype.type(inner_fn(values[0], values[1]))
                outer = output.dtype.type(outer_fn(inner, values[2]))
                if relu_chain:
                    zero = np.array(0, dtype=output.dtype)
                    output[output_index] = np.maximum(outer, zero)
                else:
                    output[output_index] = outer
            elif op.opcode in _BINARY_TREE_FUNCTIONS or op.opcode in _RELU_BINARY_TREE_OPCODES:
                relu_tree = op.opcode in _RELU_BINARY_TREE_OPCODES
                tree_opcode = op.opcode.removeprefix("relu_")
                left_fn, right_fn, root_fn = _BINARY_TREE_FUNCTIONS[tree_opcode]
                left = output.dtype.type(left_fn(values[0], values[1]))
                right = output.dtype.type(right_fn(values[2], values[3]))
                root = output.dtype.type(root_fn(left, right))
                if relu_tree:
                    zero = np.array(0, dtype=output.dtype)
                    output[output_index] = np.maximum(root, zero)
                else:
                    output[output_index] = root
            elif op.opcode in _CHAIN_TREE_FUNCTIONS:
                inner_fn, left_fn, right_fn, root_fn = _CHAIN_TREE_FUNCTIONS[op.opcode]
                inner = output.dtype.type(inner_fn(values[0], values[1]))
                left = output.dtype.type(left_fn(inner, values[2]))
                right = output.dtype.type(right_fn(values[3], values[4]))
                output[output_index] = output.dtype.type(root_fn(left, right))
            else:
                raise RuntimeError(f"unsupported CPU loop kernel: {op.opcode}")

    if not return_buffers:
        raise RuntimeError("verified loop IR unexpectedly has no return")
    outputs = tuple(np.array(buffers[buffer], copy=True) for buffer in return_buffers)
    return outputs[0] if len(outputs) == 1 else outputs
