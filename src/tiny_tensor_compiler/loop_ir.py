from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np

from . import fused_expr
from .inference import infer_binary, infer_relu, infer_reshape
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
class LoopView:
    """Read-only contiguous logical tensor sharing one allocated storage root."""

    output: int
    source: int
    type: TensorType


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


LoopOperation = LoopAlloc | LoopInput | LoopView | LoopKernel | LoopReturn


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
    def views(self) -> tuple[LoopView, ...]:
        return tuple(op for op in self.operations if isinstance(op, LoopView))

    @property
    def buffer_types(self) -> dict[int, TensorType]:
        types: dict[int, TensorType] = {alloc.buffer: alloc.type for alloc in self.allocations}
        types.update((view.output, view.type) for view in self.views)
        return types

    @property
    def input_types(self) -> tuple[TensorType, ...]:
        types = self.buffer_types
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

    def storage_root(self, buffer: int) -> int:
        roots: dict[int, int] = {alloc.buffer: alloc.buffer for alloc in self.allocations}
        for view in self.views:
            roots[view.output] = roots[view.source]
        try:
            return roots[buffer]
        except KeyError as error:
            raise KeyError(f"loop buffer p{buffer} is not declared") from error

    def dump(self) -> str:
        lines: list[str] = []
        for op in self.operations:
            if isinstance(op, LoopAlloc):
                lines.append(f"alloc p{op.buffer} : {op.type}")
                continue
            if isinstance(op, LoopInput):
                lines.append(f"p{op.output} = input {op.index}")
                continue
            if isinstance(op, LoopView):
                lines.append(f"view p{op.output} = p{op.source} : {op.type}")
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
            elif op.opcode == "reshape":
                rhs = f"reshape p{op.inputs[0]}[linear]"
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
        input_maps = (
            ()
            if op.opcode == "reshape"
            else tuple(
                _broadcast_index_map(input_type.shape, output_type.shape)
                for input_type in input_types
            )
        )
        operations.append(
            LoopKernel(
                opcode=op.opcode,
                output=plan.physical_for(op.output),
                inputs=tuple(plan.physical_for(buffer) for buffer in op.inputs),
                iteration_shape=output_type.shape,
                input_maps=input_maps,
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


def _collect_loop_buffer_metadata(
    operations: tuple[LoopOperation, ...],
) -> tuple[dict[int, TensorType], dict[int, int], dict[int, int], dict[int, int]]:
    types: dict[int, TensorType] = {}
    roots: dict[int, int] = {}
    view_positions: dict[int, int] = {}

    for index, op in enumerate(operations):
        if isinstance(op, LoopAlloc):
            if op.buffer < 0:
                raise ValueError(f"invalid negative physical buffer p{op.buffer}")
            if op.buffer in types:
                raise ValueError(f"loop buffer p{op.buffer} is declared more than once")
            types[op.buffer] = op.type
            roots[op.buffer] = op.buffer
            continue
        if not isinstance(op, LoopView):
            continue
        if op.output < 0:
            raise ValueError(f"invalid negative view buffer p{op.output}")
        if op.output in types:
            raise ValueError(f"loop buffer p{op.output} is declared more than once")
        if op.source not in types:
            raise ValueError(f"view source p{op.source} is not declared before the view")
        source_type = types[op.source]
        if source_type.dtype != op.type.dtype:
            raise ValueError("loop view dtype must match its source dtype")
        if prod(source_type.shape) != prod(op.type.shape):
            raise ValueError("loop view element count must match its source")
        expected = infer_reshape(source_type, op.type.shape)
        if expected != op.type:
            raise ValueError("loop view type does not match contiguous reshape semantics")
        types[op.output] = op.type
        roots[op.output] = roots[op.source]
        view_positions[op.output] = index

    last_uses = dict(view_positions)
    for index, op in enumerate(operations):
        referenced: tuple[int, ...]
        if isinstance(op, LoopView):
            referenced = (op.source,)
        elif isinstance(op, LoopKernel):
            referenced = op.inputs
        elif isinstance(op, LoopReturn):
            referenced = (op.buffer,)
        else:
            referenced = ()
        for buffer in referenced:
            if buffer in view_positions:
                last_uses[buffer] = index

    return types, roots, view_positions, last_uses


def _verify_storage_write_safety(
    *,
    position: int,
    output: int,
    roots: dict[int, int],
    view_positions: dict[int, int],
    last_uses: dict[int, int],
) -> None:
    root = roots[output]
    for view, declared_at in view_positions.items():
        if roots[view] != root:
            continue
        if declared_at < position <= last_uses[view]:
            raise ValueError(
                f"storage p{root} cannot be written while alias view p{view} is live"
            )


def _verify_loop_ir(operations: tuple[LoopOperation, ...]) -> None:
    buffer_types, roots, view_positions, last_uses = _collect_loop_buffer_metadata(operations)
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
            if op.buffer in allocated:
                raise ValueError(f"physical buffer p{op.buffer} is allocated more than once")
            allocated[op.buffer] = op.type
            continue

        if isinstance(op, LoopInput):
            saw_kernel = True
            if op.output not in allocated:
                raise ValueError(f"loop input destination p{op.output} is not allocated")
            _verify_storage_write_safety(
                position=index,
                output=op.output,
                roots=roots,
                view_positions=view_positions,
                last_uses=last_uses,
            )
            if op.index != next_input_index:
                raise ValueError(
                    f"input index {op.index} is not the next dense input index {next_input_index}"
                )
            next_input_index += 1
            written.add(op.output)
            continue

        if isinstance(op, LoopView):
            saw_kernel = True
            if op.source not in written:
                raise ValueError(f"view source p{op.source} is read before being written")
            written.add(op.output)
            continue

        if isinstance(op, LoopKernel):
            saw_kernel = True
            if op.output not in allocated:
                if op.output in view_positions:
                    raise ValueError("loop alias views are read-only and cannot be kernel outputs")
                raise ValueError(f"loop output p{op.output} is not allocated")
            for buffer in op.inputs:
                if buffer not in buffer_types:
                    raise ValueError(f"loop input p{buffer} is not declared")
                if buffer not in written:
                    raise ValueError(f"loop input p{buffer} is read before being written")
            _verify_storage_write_safety(
                position=index,
                output=op.output,
                roots=roots,
                view_positions=view_positions,
                last_uses=last_uses,
            )
            if any(roots[op.output] == roots[buffer] for buffer in op.inputs):
                raise ValueError("loop kernels do not permit input/output storage aliasing")

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
                expected = infer_binary(buffer_types[op.inputs[0]], buffer_types[op.inputs[1]])
                if expected != output_type:
                    raise ValueError(f"{op.opcode} loop output buffer type does not match inference")
                _verify_index_maps(op, buffer_types)
            elif op.opcode == "relu":
                if len(op.inputs) != 1 or len(op.input_maps) != 1 or op.literal is not None:
                    raise ValueError("relu loop requires one input and one index map")
                expected = infer_relu(buffer_types[op.inputs[0]])
                if expected != output_type:
                    raise ValueError("relu loop output buffer type does not match inference")
                _verify_index_maps(op, buffer_types)
            elif op.opcode == "reshape":
                if len(op.inputs) != 1 or op.input_maps or op.literal is not None:
                    raise ValueError("reshape loop requires one input, no index maps, and no literal")
                expected = infer_reshape(buffer_types[op.inputs[0]], output_type.shape)
                if expected != output_type:
                    raise ValueError("reshape loop output buffer type does not match inference")
            elif op.opcode in {"relu_add", "relu_mul"}:
                if len(op.inputs) != 2 or len(op.input_maps) != 2 or op.literal is not None:
                    raise ValueError(
                        f"{op.opcode} loop requires two inputs and two index maps"
                    )
                binary_type = infer_binary(buffer_types[op.inputs[0]], buffer_types[op.inputs[1]])
                expected = infer_relu(binary_type)
                if expected != output_type:
                    raise ValueError(f"{op.opcode} loop output buffer type does not match inference")
                _verify_index_maps(op, buffer_types)
            else:
                expression = fused_expression_for_kernel(op)
                if expression is None:
                    raise ValueError(f"unsupported loop kernel: {op.opcode}")
                _verify_fused_expression(op, expression, buffer_types, output_type)

            written.add(op.output)
            continue

        if not isinstance(op, LoopReturn):
            raise TypeError(f"unsupported loop IR operation at index {index}")
        if op.buffer not in buffer_types:
            raise ValueError(f"returned loop buffer p{op.buffer} is not declared")
        if op.buffer not in written:
            raise ValueError(f"returned loop buffer p{op.buffer} is not written")
        saw_return = True

    if not saw_return:
        raise ValueError("loop IR must end with a return")
    if allocated and set(allocated) != set(range(len(allocated))):
        raise ValueError("physical loop buffer ids must be dense starting at p0")
    if buffer_types and set(buffer_types) != set(range(len(buffer_types))):
        raise ValueError("logical loop buffer ids must be dense starting at p0")


def _verify_fused_expression(
    op: LoopKernel,
    expression: fused_expr.FusedExpression,
    buffer_types: dict[int, TensorType],
    output_type: TensorType,
) -> None:
    arity_word = {3: "three", 4: "four", 5: "five"}.get(
        expression.input_count,
        str(expression.input_count),
    )
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
    if any(buffer_types[buffer].dtype != output_type.dtype for buffer in op.inputs):
        raise ValueError(f"{expression.display_name} loop requires one exact integer dtype")

    refs = {
        name: buffer_types[buffer]
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

    _verify_index_maps(op, buffer_types)


def _verify_index_maps(op: LoopKernel, buffer_types: dict[int, TensorType]) -> None:
    for buffer, index_map in zip(op.inputs, op.input_maps, strict=True):
        expected = _broadcast_index_map(buffer_types[buffer].shape, op.iteration_shape)
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
