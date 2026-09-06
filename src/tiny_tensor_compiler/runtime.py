from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .input_validation import prepare_runtime_inputs
from .ir import Module, Value
from .reduction import REDUCTION_OPCODES, ReductionOperator, ReductionPlan
from .symbolic import has_symbolic_shapes, specialize_for_inputs
from .verifier import verify

ExecutionResult = np.ndarray | tuple[np.ndarray, ...]


def execute_reference(module: Module, inputs: Sequence[Any] = ()) -> ExecutionResult:
    """Execute verified tensor IR directly; symbolic shapes specialize from inputs."""
    if has_symbolic_shapes(module):
        module, _ = specialize_for_inputs(module, inputs)
    verify(module)
    input_ops = tuple(op for op in module.function.ops if op.opcode == "input")
    runtime_inputs = prepare_runtime_inputs(
        tuple(op.results[0].type for op in input_ops),
        inputs,
    )
    values: dict[Value, np.ndarray] = {}

    for op in module.function.ops:
        if op.opcode == "input":
            values[op.results[0]] = np.array(runtime_inputs[op.attrs["index"]], copy=True)
        elif op.opcode == "const":
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
        elif op.opcode in REDUCTION_OPCODES:
            operand = values[op.operands[0]]
            plan = ReductionPlan.from_opcode(op.opcode, op.attrs.get("axis"))
            if plan.operator is ReductionOperator.ARGMAX:
                values[op.results[0]] = _execute_argmax(plan, operand)
                continue

            dtype = op.results[0].type.dtype.to_numpy()
            if plan.axes is None:
                accumulator = plan.operator.identity(dtype)
                for index in np.ndindex(operand.shape):
                    accumulator = plan.operator.combine(dtype, accumulator, operand[index])
                values[op.results[0]] = np.array(accumulator, dtype=dtype)
            else:
                output = np.empty(op.results[0].type.shape, dtype=dtype)
                reduction_shape = plan.reduction_shape(operand.shape)
                for output_index in np.ndindex(output.shape):
                    accumulator = plan.operator.identity(dtype)
                    for reduction_index in np.ndindex(reduction_shape):
                        input_index = plan.input_index(
                            operand.ndim,
                            output_index,
                            reduction_index,
                        )
                        accumulator = plan.operator.combine(
                            dtype,
                            accumulator,
                            operand[input_index],
                        )
                    output[output_index] = accumulator
                values[op.results[0]] = output
        elif op.opcode == "reshape":
            operand = values[op.operands[0]]
            values[op.results[0]] = np.array(
                np.reshape(operand, op.results[0].type.shape, order="C"),
                copy=True,
            )
        elif op.opcode == "view":
            operand = values[op.operands[0]]
            viewed = np.reshape(operand, op.results[0].type.shape, order="C")
            if viewed.size and not np.shares_memory(viewed, operand):
                raise RuntimeError("verified contiguous view unexpectedly required a copy")
            values[op.results[0]] = viewed
        elif op.opcode == "slice":
            operand = values[op.operands[0]]
            index = [slice(None)] * operand.ndim
            index[op.attrs["axis"]] = slice(
                op.attrs["start"], op.attrs["stop"], op.attrs["step"]
            )
            viewed = operand[tuple(index)]
            if viewed.size and not np.shares_memory(viewed, operand):
                raise RuntimeError("verified positive-stride slice unexpectedly required a copy")
            values[op.results[0]] = viewed
        elif op.opcode == "reverse":
            operand = values[op.operands[0]]
            viewed = np.flip(operand, axis=op.attrs["axis"])
            if viewed.size and not np.shares_memory(viewed, operand):
                raise RuntimeError("verified reverse unexpectedly required a copy")
            values[op.results[0]] = viewed
        elif op.opcode == "transpose":
            operand = values[op.operands[0]]
            viewed = np.transpose(operand, axes=op.attrs["axes"])
            if viewed.size and not np.shares_memory(viewed, operand):
                raise RuntimeError("verified transpose unexpectedly required a copy")
            values[op.results[0]] = viewed
        elif op.opcode == "copy_into":
            root = values[op.operands[0]]
            target = values[op.operands[1]]
            source = values[op.operands[2]]
            np.copyto(target, source)
            values[op.results[0]] = root
        elif op.opcode == "binary_into":
            root = values[op.operands[0]]
            target = values[op.operands[1]]
            source = values[op.operands[2]]
            binary = np.add if op.attrs["operator"] == "add" else np.multiply
            binary(target, source, out=target)
            values[op.results[0]] = root
        elif op.opcode == "binary_inplace":
            root = values[op.operands[0]]
            source = values[op.operands[1]]
            binary = np.add if op.attrs["operator"] == "add" else np.multiply
            binary(root, source, out=root)
            values[op.results[0]] = root
        elif op.opcode == "return":
            outputs = tuple(np.array(values[operand], copy=True) for operand in op.operands)
            return outputs[0] if len(outputs) == 1 else outputs
    raise RuntimeError("verified module unexpectedly has no return")


def _execute_argmax(plan: ReductionPlan, operand: np.ndarray) -> np.ndarray:
    if plan.axes is None:
        best_index = 0
        iterator = iter(enumerate(np.ndindex(operand.shape)))
        _, first_coordinate = next(iterator)
        best = operand[first_coordinate]
        for logical_index, coordinate in iterator:
            candidate = operand[coordinate]
            if plan.operator.candidate_wins(best, candidate):
                best = candidate
                best_index = logical_index
        return np.array(best_index, dtype=np.int64)

    output_shape = tuple(
        dim for position, dim in enumerate(operand.shape) if position not in set(plan.axes)
    )
    output = np.empty(output_shape, dtype=np.int64)
    reduction_shape = plan.reduction_shape(operand.shape)
    for output_index in np.ndindex(output.shape):
        reduction_iterator = iter(np.ndindex(reduction_shape))
        first_reduction_index = next(reduction_iterator)
        best_input_index = plan.input_index(
            operand.ndim,
            output_index,
            first_reduction_index,
        )
        best = operand[best_input_index]
        best_index = first_reduction_index[0]
        for reduction_index in reduction_iterator:
            input_index = plan.input_index(
                operand.ndim,
                output_index,
                reduction_index,
            )
            candidate = operand[input_index]
            if plan.operator.candidate_wins(best, candidate):
                best = candidate
                best_index = reduction_index[0]
        output[output_index] = best_index
    return output
