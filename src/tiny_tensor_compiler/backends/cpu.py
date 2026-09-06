from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..input_binding import BorrowedLoopProgram, borrowed_slots
from ..input_binding import borrow_inputs as bind_borrowed_inputs
from ..input_validation import prepare_runtime_inputs
from ..loop_ir import (
    IndexMap,
    LoopAlloc,
    LoopBinaryInto,
    LoopCopyInto,
    LoopInplaceBinary,
    LoopInput,
    LoopKernel,
    LoopProgram,
    LoopReturn,
    LoopView,
    fused_expression_for_kernel,
    lower_to_loops,
)
from ..lowering import CPUProgram
from ..reduction import ReductionOperator, ReductionPlan

ExecutionResult = np.ndarray | tuple[np.ndarray, ...]
LoopExecutionProgram = LoopProgram | BorrowedLoopProgram


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
    layouts = program.value_layouts
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

        if isinstance(op, LoopView):
            root = program.storage_root(op.output)
            root_array = buffers[root]
            layout = layouts[op.output]
            itemsize = root_array.dtype.itemsize
            viewed = np.ndarray(
                shape=op.type.shape,
                dtype=op.type.dtype.to_numpy(),
                buffer=root_array,
                offset=layout.offset * itemsize,
                strides=tuple(stride * itemsize for stride in layout.strides),
            )
            if viewed.size and not np.shares_memory(viewed, root_array):
                raise RuntimeError("verified loop view unexpectedly required a copy")
            buffers[op.output] = viewed
            continue

        if isinstance(op, LoopCopyInto):
            np.copyto(buffers[op.target], buffers[op.source])
            buffers[op.output] = buffers[op.root]
            continue

        if isinstance(op, LoopBinaryInto):
            binary = np.add if op.operator == "add" else np.multiply
            target = buffers[op.target]
            source = buffers[op.source]
            source_map = op.source_map
            if source_map is None:
                source_map = IndexMap(tuple(range(source.ndim)))
            for target_index in np.ndindex(target.shape):
                source_index = source_map.apply(target_index)
                target[target_index] = binary(
                    target[target_index],
                    source[source_index],
                    dtype=target.dtype,
                )
            buffers[op.output] = buffers[op.root]
            continue

        if isinstance(op, LoopInplaceBinary):
            binary = np.add if op.operator == "add" else np.multiply
            binary(buffers[op.root], buffers[op.source], out=buffers[op.root])
            buffers[op.output] = buffers[op.root]
            continue

        if isinstance(op, LoopReturn):
            return_buffers.append(op.buffer)
            continue

        if not isinstance(op, LoopKernel):
            raise TypeError("unsupported CPU loop operation")

        output = buffers[op.output]
        reduction = op.reduction
        if reduction is not None:
            source = buffers[op.inputs[0]]
            if reduction.operator is ReductionOperator.ARGMAX:
                _execute_argmax_reduction(reduction, source, output, op.iteration_shape)
                continue
            if reduction.axes is None:
                accumulator = reduction.operator.identity(output.dtype)
                for input_index in np.ndindex(source.shape):
                    accumulator = reduction.operator.combine(
                        output.dtype,
                        accumulator,
                        source[input_index],
                    )
                output[()] = accumulator
            else:
                reduction_shape = reduction.reduction_shape(source.shape)
                for output_index in np.ndindex(op.iteration_shape):
                    accumulator = reduction.operator.identity(output.dtype)
                    for reduction_index in np.ndindex(reduction_shape):
                        input_index = reduction.input_index(
                            source.ndim,
                            output_index,
                            reduction_index,
                        )
                        accumulator = reduction.operator.combine(
                            output.dtype,
                            accumulator,
                            source[input_index],
                        )
                    output[output_index] = accumulator
            continue

        if op.opcode == "reshape":
            source = buffers[op.inputs[0]]
            np.copyto(output.reshape(-1), source.reshape(-1))
            continue

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
            else:
                expression = fused_expression_for_kernel(op)
                if expression is None:
                    raise RuntimeError(f"unsupported CPU loop kernel: {op.opcode}")
                refs = dict(zip(expression.input_names, values, strict=True))
                for step in expression.steps:
                    if step.opcode == "relu":
                        zero = np.array(0, dtype=output.dtype)
                        refs[step.output] = np.maximum(refs[step.inputs[0]], zero)
                        continue
                    lhs, rhs = step.inputs
                    binary = np.add if step.opcode == "add" else np.multiply
                    refs[step.output] = output.dtype.type(binary(refs[lhs], refs[rhs]))
                output[output_index] = refs[expression.result]

    if not return_buffers:
        raise RuntimeError("verified loop IR unexpectedly has no return")
    outputs = tuple(np.array(buffers[buffer], copy=True) for buffer in return_buffers)
    return outputs[0] if len(outputs) == 1 else outputs


def _execute_argmax_reduction(
    plan: ReductionPlan,
    source: np.ndarray,
    output: np.ndarray,
    iteration_shape: tuple[int, ...],
) -> None:
    if plan.axes is None:
        iterator = iter(enumerate(np.ndindex(source.shape)))
        _, first_coordinate = next(iterator)
        best = source[first_coordinate]
        best_index = 0
        for logical_index, coordinate in iterator:
            candidate = source[coordinate]
            if plan.operator.candidate_wins(best, candidate):
                best = candidate
                best_index = logical_index
        output[()] = best_index
        return

    reduction_shape = plan.reduction_shape(source.shape)
    for output_index in np.ndindex(iteration_shape):
        reduction_iterator = iter(np.ndindex(reduction_shape))
        first_reduction_index = next(reduction_iterator)
        best_input_index = plan.input_index(
            source.ndim,
            output_index,
            first_reduction_index,
        )
        best = source[best_input_index]
        best_index = first_reduction_index[0]
        for reduction_index in reduction_iterator:
            input_index = plan.input_index(
                source.ndim,
                output_index,
                reduction_index,
            )
            candidate = source[input_index]
            if plan.operator.candidate_wins(best, candidate):
                best = candidate
                best_index = reduction_index[0]
        output[output_index] = best_index
