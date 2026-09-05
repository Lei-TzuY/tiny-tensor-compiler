from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import fused_expr
from .inference import infer_binary, infer_relu
from .ir import DType, TensorType
from .lowering import BufferAlloc, BufferInput, BufferReturn, CPUProgram, plan_memory


@dataclass(frozen=True)
class IndexMap:
    """Map an output loop index to one input tensor index under broadcasting."""

    axes: tuple[int | None, ...]

    def __post_init__(self) -> None:
        for axis in self.axes:
            if axis is not None and axis < 0:
                raise ValueError("loop index-map axes must be non-negative")

    def apply(self, output_index: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(0 if axis is None else output_index[axis] for axis in self.axes)


@dataclass(frozen=True)
class LoopAlloc:
    buffer: int
    type: TensorType


@dataclass(frozen=True)
class LoopInput:
    output: int
    index: int


@dataclass(frozen=True)
class LoopKernel:
    opcode: str
    output: int
    inputs: tuple[int, ...]
    iteration_shape: tuple[int, ...]
    input_maps: tuple[IndexMap, ...]
    literal: np.ndarray[Any, Any] | None = None
    fused_expression: fused_expr.FusedExpression | None = None


@dataclass(frozen=True)
class LoopReturn:
    buffer: int


LoopOperation = LoopAlloc | LoopInput | LoopKernel | LoopReturn


@dataclass(frozen=True)
class LoopProgram:
    operations: tuple[LoopOperation, ...]

    def __post_init__(self) -> None:
        _verify_loop_ir(self.operations)

    @property
    def allocations(self) -> tuple[LoopAlloc, ...]:
        return tuple(op for op in self.operations if isinstance(op, LoopAlloc))

    @property
    def inputs(self) -> tuple[LoopInput, ...]:
        return tuple(op for op in self.operations if isinstance(op, LoopInput))

    @property
    def input_types(self) -> tuple[TensorType, ...]:
        types = {alloc.buffer: alloc.type for alloc in self.allocations}
        return tuple(types[op.output] for op in self.inputs)

    @property
    def kernels(self) -> tuple[LoopKernel, ...]:
        return tuple(op for op in self.operations if isinstance(op, LoopKernel))

    @property
    def return_slots(self) -> tuple[int, ...]:
        return tuple(op.buffer for op in self.operations if isinstance(op, LoopReturn))

    @property
    def return_slot(self) -> int:
        """Compatibility alias for single-output programs."""
        slots = self.return_slots
        if len(slots) != 1:
            raise RuntimeError(
                f"return_slot requires exactly one returned buffer, found {len(slots)}"
            )
        return slots[0]

    def dump(self) -> str:
        lines: list[str] = []
        for op in self.operations:
            if isinstance(op, LoopAlloc):
                lines.append(f"alloc p{op.buffer} : {op.type}")
                continue
            if isinstance(op, LoopInput):
                lines.append(f"p{op.output} = input {op.index}")
                continue
            if isinstance(op, LoopReturn):
                lines.append(f"return p{op.buffer}")
                continue

            output_index = _format_index(tuple(range(len(op.iteration_shape))))
            prefix = _format_loop_prefix(op.iteration_shape)
            if op.opcode == "const":
                if op.literal is None:
                    raise RuntimeError("verified const loop unexpectedly has no literal")
                literal = _format_literal(op.literal)
                literal_index = "" if op.literal.ndim == 0 else output_index
                rhs = f"const {literal}{literal_index}"
            else:
                operands = ", ".join(
                    f"p{buffer}{_format_index(index_map.axes)}"
                    for buffer, index_map in zip(op.inputs, op.input_maps, strict=True)
                )
                rhs = f"{op.opcode} {operands}"
            lines.append(f"{prefix}: p{op.output}{output_index} = {rhs}")
        return "\n".join(lines)


def fused_expression_for_kernel(op: LoopKernel) -> fused_expr.FusedExpression | None:
    """Return structured fused semantics, decoding legacy spelling only as fallback."""
    if op.fused_expression is None:
        return fused_expr.describe_fused_opcode(op.opcode)
    if fused_expr.encode_fused_opcode(op.fused_expression) != op.opcode:
        raise ValueError("loop fused expression metadata does not match opcode")
    return op.fused_expression


def lower_to_loops(program: CPUProgram) -> LoopProgram:
    """Lower verified virtual-buffer operations to explicit physical-buffer loops."""
    plan = plan_memory(program)
    virtual_types = {alloc.buffer: alloc.type for alloc in program.allocations}
    operations: list[LoopOperation] = [
        LoopAlloc(slot, buffer_type) for slot, buffer_type in enumerate(plan.physical_types)
    ]

    for op in program.operations:
        if isinstance(op, BufferAlloc):
            continue
        if isinstance(op, BufferInput):
            operations.append(LoopInput(plan.physical_for(op.output), op.index))
            continue
        if isinstance(op, BufferReturn):
            operations.append(LoopReturn(plan.physical_for(op.buffer)))
            continue

        output_type = virtual_types[op.output]
        input_types = tuple(virtual_types[buffer] for buffer in op.inputs)
        operations.append(
            LoopKernel(
                opcode=op.opcode,
                output=plan.physical_for(op.output),
                inputs=tuple(plan.physical_for(buffer) for buffer in op.inputs),
                iteration_shape=output_type.shape,
                input_maps=tuple(
                    _broadcast_index_map(input_type.shape, output_type.shape)
                    for input_type in input_types
                ),
                literal=op.literal,
            )
        )

    return LoopProgram(tuple(operations))


def fuse_elementwise(program: LoopProgram) -> LoopProgram:
    """Compatibility entry point for the sole topology-driven fusion planner."""
    from .fusion_planner import fuse_elementwise as plan_elementwise_fusion

    return plan_elementwise_fusion(program)


def _broadcast_index_map(input_shape: tuple[int, ...], output_shape: tuple[int, ...]) -> IndexMap:
    if len(input_shape) > len(output_shape):
        raise ValueError("input rank exceeds loop iteration rank")

    offset = len(output_shape) - len(input_shape)
    axes: list[int | None] = []
    for input_axis, input_dim in enumerate(input_shape):
        output_axis = offset + input_axis
        output_dim = output_shape[output_axis]
        if input_dim == output_dim:
            axes.append(output_axis)
        elif input_dim == 1:
            axes.append(None)
        else:
            raise ValueError("input shape is not broadcast-compatible with loop iteration shape")
    return IndexMap(tuple(axes))


def _verify_loop_ir(operations: tuple[LoopOperation, ...]) -> None:
    allocated: dict[int, TensorType] = {}
    written: set[int] = set()
    next_input_index = 0
    saw_kernel = False
    saw_return = False

    for index, op in enumerate(operations):
        if saw_return and not isinstance(op, LoopReturn):
            raise ValueError("loop IR operation appears after return")

        if isinstance(op, LoopAlloc):
            if saw_kernel:
                raise ValueError("physical buffer allocation appears after loop execution begins")
            if op.buffer < 0:
                raise ValueError(f"invalid negative physical buffer p{op.buffer}")
            if op.buffer in allocated:
                raise ValueError(f"physical buffer p{op.buffer} is allocated more than once")
            allocated[op.buffer] = op.type
            continue

        if isinstance(op, LoopInput):
            saw_kernel = True
            if op.output not in allocated:
                raise ValueError(f"loop input destination p{op.output} is not allocated")
            if op.index != next_input_index:
                raise ValueError(
                    f"input index {op.index} is not the next dense input index {next_input_index}"
                )
            next_input_index += 1
            written.add(op.output)
            continue

        if isinstance(op, LoopKernel):
            saw_kernel = True
            if op.output not in allocated:
                raise ValueError(f"loop output p{op.output} is not allocated")
            if op.output in op.inputs:
                raise ValueError("loop kernels do not permit in-place input/output aliasing")
            for buffer in op.inputs:
                if buffer not in allocated:
                    raise ValueError(f"loop input p{buffer} is not allocated")
                if buffer not in written:
                    raise ValueError(f"loop input p{buffer} is read before being written")

            output_type = allocated[op.output]
            if op.iteration_shape != output_type.shape:
                raise ValueError("loop iteration shape must match output buffer shape")

            if op.fused_expression is not None:
                fused_expression_for_kernel(op)

            if op.opcode == "const":
                if op.inputs or op.input_maps:
                    raise ValueError("const loop must not have inputs or index maps")
                if op.literal is None:
                    raise ValueError("const loop is missing its literal")
                literal = np.asarray(op.literal)
                if tuple(literal.shape) != output_type.shape:
                    raise ValueError("const loop literal shape does not match output buffer")
                if literal.dtype != output_type.dtype.to_numpy():
                    raise ValueError("const loop literal dtype does not match output buffer")
            elif op.opcode in {"add", "mul"}:
                if len(op.inputs) != 2 or len(op.input_maps) != 2 or op.literal is not None:
                    raise ValueError(f"{op.opcode} loop requires two inputs and two index maps")
                expected = infer_binary(allocated[op.inputs[0]], allocated[op.inputs[1]])
                if expected != output_type:
                    raise ValueError(f"{op.opcode} loop output buffer type does not match inference")
                _verify_index_maps(op, allocated)
            elif op.opcode == "relu":
                if len(op.inputs) != 1 or len(op.input_maps) != 1 or op.literal is not None:
                    raise ValueError("relu loop requires one input and one index map")
                expected = infer_relu(allocated[op.inputs[0]])
                if expected != output_type:
                    raise ValueError("relu loop output buffer type does not match inference")
                _verify_index_maps(op, allocated)
            elif op.opcode in {"relu_add", "relu_mul"}:
                if len(op.inputs) != 2 or len(op.input_maps) != 2 or op.literal is not None:
                    raise ValueError(
                        f"{op.opcode} loop requires two inputs and two index maps"
                    )
                binary_type = infer_binary(allocated[op.inputs[0]], allocated[op.inputs[1]])
                expected = infer_relu(binary_type)
                if expected != output_type:
                    raise ValueError(f"{op.opcode} loop output buffer type does not match inference")
                _verify_index_maps(op, allocated)
            else:
                expression = fused_expression_for_kernel(op)
                if expression is None:
                    raise ValueError(f"unsupported loop kernel: {op.opcode}")
                _verify_fused_expression(op, expression, allocated, output_type)

            written.add(op.output)
            continue

        if not isinstance(op, LoopReturn):
            raise TypeError(f"unsupported loop IR operation at index {index}")
        if op.buffer not in allocated:
            raise ValueError(f"returned physical buffer p{op.buffer} is not allocated")
        if op.buffer not in written:
            raise ValueError(f"returned physical buffer p{op.buffer} is not written")
        saw_return = True

    if not saw_return:
        raise ValueError("loop IR must end with a return")
    if allocated and set(allocated) != set(range(len(allocated))):
        raise ValueError("physical loop buffer ids must be dense starting at p0")


def _verify_fused_expression(
    op: LoopKernel,
    expression: fused_expr.FusedExpression,
    allocated: dict[int, TensorType],
    output_type: TensorType,
) -> None:
    arity_word = {3: "three", 4: "four", 5: "five"}[expression.input_count]
    if (
        len(op.inputs) != expression.input_count
        or len(op.input_maps) != expression.input_count
        or op.literal is not None
    ):
        raise ValueError(
            f"{op.opcode} loop requires {arity_word} inputs and {arity_word} index maps"
        )

    if output_type.dtype not in {DType.INT32, DType.INT64}:
        raise ValueError(f"{expression.display_name} loop requires an integer output dtype")
    if any(allocated[buffer].dtype != output_type.dtype for buffer in op.inputs):
        raise ValueError(f"{expression.display_name} loop requires one exact integer dtype")

    refs = {
        name: allocated[buffer]
        for name, buffer in zip(expression.input_names, op.inputs, strict=True)
    }
    for step_index, step in enumerate(expression.steps):
        if step.opcode == "relu":
            expected = infer_relu(refs[step.inputs[0]])
        else:
            lhs, rhs = step.inputs
            expected = infer_binary(refs[lhs], refs[rhs])
        refs[step.output] = expected
        if expected == output_type:
            continue

        final_step = step_index == len(expression.steps) - 1
        pre_relu_step = expression.terminal_relu and step_index == len(expression.steps) - 2
        if final_step or pre_relu_step:
            raise ValueError(f"{op.opcode} loop output buffer type does not match inference")
        intermediate = "type" if expression.family == "binary-chain" else "types"
        raise ValueError(
            f"{op.opcode} loop intermediate {intermediate} must match its output type"
        )

    _verify_index_maps(op, allocated)


def _verify_index_maps(op: LoopKernel, allocated: dict[int, TensorType]) -> None:
    for buffer, index_map in zip(op.inputs, op.input_maps, strict=True):
        expected = _broadcast_index_map(allocated[buffer].shape, op.iteration_shape)
        if index_map != expected:
            raise ValueError("loop input index map does not match broadcasting semantics")


def _format_loop_prefix(shape: tuple[int, ...]) -> str:
    if not shape:
        return "once"
    bounds = ", ".join(f"i{axis}<{dim}" for axis, dim in enumerate(shape))
    return f"for [{bounds}]"


def _format_index(axes: tuple[int | None, ...]) -> str:
    values = ", ".join("0" if axis is None else f"i{axis}" for axis in axes)
    return f"[{values}]"


def _format_literal(value: np.ndarray[Any, Any]) -> str:
    if value.ndim == 0:
        return repr(value.item())
    return repr(value.tolist())