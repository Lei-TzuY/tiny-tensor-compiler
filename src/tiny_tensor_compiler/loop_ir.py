from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import fused_expr
from .inference import infer_binary, infer_concat, infer_reduction, infer_relu, infer_reshape
from .ir import DType, TensorType
from .layout import StorageLayout, element_count
from .lowering import (
    BufferAlloc,
    BufferBinaryInto,
    BufferCopyInto,
    BufferInplaceBinary,
    BufferInput,
    BufferReturn,
    BufferView,
    CPUProgram,
    plan_memory,
)
from .reduction import REDUCTION_OPCODES, ReductionPlan


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
    output: int
    source: int
    type: TensorType
    layout: StorageLayout | None = None


@dataclass(frozen=True)
class LoopCopyInto:
    output: int
    root: int
    target: int
    source: int
    type: TensorType
    layout: StorageLayout


@dataclass(frozen=True)
class LoopBinaryInto:
    output: int
    root: int
    target: int
    source: int
    operator: str
    type: TensorType
    layout: StorageLayout


@dataclass(frozen=True)
class LoopInplaceBinary:
    output: int
    root: int
    source: int
    operator: str
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
    reduction_axis: int | None = None
    concat_axis: int | None = None

    @property
    def reduction(self) -> ReductionPlan | None:
        if self.opcode not in REDUCTION_OPCODES:
            return None
        return ReductionPlan.from_opcode(self.opcode, self.reduction_axis)


@dataclass(frozen=True)
class LoopReturn:
    buffer: int


LoopOperation = (
    LoopAlloc
    | LoopInput
    | LoopView
    | LoopCopyInto
    | LoopBinaryInto
    | LoopInplaceBinary
    | LoopKernel
    | LoopReturn
)


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
    def copies(self) -> tuple[LoopCopyInto, ...]:
        return tuple(op for op in self.operations if isinstance(op, LoopCopyInto))

    @property
    def binary_intos(self) -> tuple[LoopBinaryInto, ...]:
        return tuple(op for op in self.operations if isinstance(op, LoopBinaryInto))

    @property
    def value_types(self) -> dict[int, TensorType]:
        types = {alloc.buffer: alloc.type for alloc in self.allocations}
        types.update(
            {
                op.output: op.type
                for op in self.operations
                if isinstance(
                    op,
                    (LoopView, LoopCopyInto, LoopBinaryInto, LoopInplaceBinary),
                )
            }
        )
        return types

    @property
    def value_layouts(self) -> dict[int, StorageLayout]:
        types = self.value_types
        layouts = {
            alloc.buffer: StorageLayout.contiguous(alloc.type.shape) for alloc in self.allocations
        }
        roots = {alloc.buffer: alloc.buffer for alloc in self.allocations}
        root_types = {alloc.buffer: alloc.type for alloc in self.allocations}
        for op in self.operations:
            if isinstance(op, LoopView):
                source_layout = layouts[op.source]
                root = roots[op.source]
                layout = (
                    source_layout.reshaped(types[op.source].shape, op.type.shape)
                    if op.layout is None
                    else op.layout
                )
                layout.validate_bounds(op.type.shape, element_count(root_types[root].shape))
                layouts[op.output] = layout
                roots[op.output] = root
            elif isinstance(op, LoopCopyInto):
                if op.root not in roots:
                    raise ValueError("copy_into root handle has no storage root")
                root = roots[op.root]
                if types[op.root] != root_types[root] or layouts[op.root] != layouts[root]:
                    raise ValueError("copy_into root handle must expose the full owning root")
                op.layout.validate_bounds(op.type.shape, element_count(root_types[root].shape))
                layouts[op.output] = op.layout
                roots[op.output] = root
            elif isinstance(op, LoopBinaryInto):
                if op.root not in roots:
                    raise ValueError("binary_into root handle has no storage root")
                root = roots[op.root]
                if types[op.root] != root_types[root] or layouts[op.root] != layouts[root]:
                    raise ValueError("binary_into root handle must expose the full owning root")
                op.layout.validate_bounds(op.type.shape, element_count(root_types[root].shape))
                layouts[op.output] = op.layout
                roots[op.output] = root
            elif isinstance(op, LoopInplaceBinary):
                if op.root not in roots:
                    raise ValueError("binary_inplace root handle has no storage root")
                root = roots[op.root]
                if types[op.root] != root_types[root] or layouts[op.root] != layouts[root]:
                    raise ValueError("binary_inplace root handle must expose the full owning root")
                layouts[op.output] = layouts[root]
                roots[op.output] = root
        return layouts

    @property
    def input_types(self) -> tuple[TensorType, ...]:
        types = self.value_types
        return tuple(types[op.output] for op in self.inputs)

    @property
    def inplace_binaries(self) -> tuple[LoopInplaceBinary, ...]:
        return tuple(op for op in self.operations if isinstance(op, LoopInplaceBinary))

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
        roots = {alloc.buffer: alloc.buffer for alloc in self.allocations}
        for op in self.operations:
            if isinstance(op, LoopView):
                if op.source not in roots:
                    raise KeyError(f"view source p{op.source} has no storage root")
                roots[op.output] = roots[op.source]
            elif isinstance(op, LoopCopyInto):
                if op.root not in roots:
                    raise KeyError(f"copy_into root handle p{op.root} has no storage root")
                roots[op.output] = roots[op.root]
            elif isinstance(op, LoopBinaryInto):
                if op.root not in roots:
                    raise KeyError(f"binary_into root handle p{op.root} has no storage root")
                roots[op.output] = roots[op.root]
            elif isinstance(op, LoopInplaceBinary):
                if op.root not in roots:
                    raise KeyError(f"binary_inplace root handle p{op.root} has no storage root")
                roots[op.output] = roots[op.root]
        try:
            return roots[buffer]
        except KeyError as exc:
            raise KeyError(f"loop value p{buffer} has no storage root") from exc

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
                suffix = (
                    ""
                    if op.layout is None
                    else f" offset={op.layout.offset} strides={op.layout.strides}"
                )
                lines.append(f"p{op.output} = view p{op.source}{suffix} : {op.type}")
                continue
            if isinstance(op, LoopCopyInto):
                lines.append(
                    f"p{op.output} = copy_into root=p{op.root} target=p{op.target} "
                    f"source=p{op.source} : {op.type}"
                )
                continue
            if isinstance(op, LoopBinaryInto):
                lines.append(
                    f"p{op.output} = binary_into[{op.operator}] root=p{op.root} "
                    f"target=p{op.target} source=p{op.source} : {op.type}"
                )
                continue
            if isinstance(op, LoopInplaceBinary):
                lines.append(
                    f"p{op.output} = binary_inplace[{op.operator}] root=p{op.root} source=p{op.source} : {op.type}"
                )
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
            elif op.opcode == "concat":
                operands = ", ".join(f"p{buffer}" for buffer in op.inputs)
                rhs = f"concat(axis={op.concat_axis}) {operands}"
            elif op.opcode == "reshape":
                rhs = f"reshape p{op.inputs[0]}[linear]"
            elif op.reduction is not None:
                plan = op.reduction
                if plan.axis is None:
                    rhs = f"{plan.opcode} p{op.inputs[0]}[logical-c-order]"
                else:
                    rhs = (
                        f"{plan.opcode}(axis={plan.axis}) "
                        f"p{op.inputs[0]}[logical-axis-order]"
                    )
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
    """Lower verified virtual-buffer operations to storage roots plus logical loop views."""
    plan = plan_memory(program)
    virtual_types = {alloc.buffer: alloc.type for alloc in program.allocations}
    virtual_handles = {
        assignment.virtual: assignment.physical for assignment in plan.assignments
    }
    next_handle = plan.physical_count
    operations: list[LoopOperation] = [
        LoopAlloc(slot, buffer_type) for slot, buffer_type in enumerate(plan.physical_types)
    ]

    for op in program.operations:
        if isinstance(op, BufferAlloc):
            continue
        if isinstance(op, BufferInput):
            operations.append(LoopInput(virtual_handles[op.output], op.index))
            continue
        if isinstance(op, BufferView):
            source = virtual_handles[op.source]
            handle = next_handle
            next_handle += 1
            virtual_handles[op.output] = handle
            alias = plan.alias_for(op.output)
            if alias is None:
                raise RuntimeError("planned buffer view unexpectedly has no alias descriptor")
            operations.append(LoopView(handle, source, virtual_types[op.output], alias.layout))
            continue
        if isinstance(op, BufferCopyInto):
            handle = next_handle
            next_handle += 1
            virtual_handles[op.output] = handle
            alias = plan.alias_for(op.output)
            if alias is None:
                raise RuntimeError("planned copy_into result unexpectedly has no alias descriptor")
            operations.append(
                LoopCopyInto(
                    output=handle,
                    root=virtual_handles[op.root],
                    target=virtual_handles[op.target],
                    source=virtual_handles[op.source],
                    type=virtual_types[op.output],
                    layout=alias.layout,
                )
            )
            continue
        if isinstance(op, BufferBinaryInto):
            handle = next_handle
            next_handle += 1
            virtual_handles[op.output] = handle
            alias = plan.alias_for(op.output)
            if alias is None:
                raise RuntimeError("planned binary_into result unexpectedly has no alias descriptor")
            operations.append(
                LoopBinaryInto(
                    output=handle,
                    root=virtual_handles[op.root],
                    target=virtual_handles[op.target],
                    source=virtual_handles[op.source],
                    operator=op.operator,
                    type=virtual_types[op.output],
                    layout=alias.layout,
                )
            )
            continue
        if isinstance(op, BufferInplaceBinary):
            handle = next_handle
            next_handle += 1
            virtual_handles[op.output] = handle
            operations.append(
                LoopInplaceBinary(
                    output=handle,
                    root=virtual_handles[op.root],
                    source=virtual_handles[op.source],
                    operator=op.operator,
                    type=virtual_types[op.output],
                )
            )
            continue
        if isinstance(op, BufferReturn):
            operations.append(LoopReturn(virtual_handles[op.buffer]))
            continue

        output_type = virtual_types[op.output]
        input_types = tuple(virtual_types[buffer] for buffer in op.inputs)
        input_maps = (
            ()
            if op.opcode in {"reshape", "concat"} or op.reduction is not None
            else tuple(
                _broadcast_index_map(input_type.shape, output_type.shape)
                for input_type in input_types
            )
        )
        operations.append(
            LoopKernel(
                opcode=op.opcode,
                output=virtual_handles[op.output],
                inputs=tuple(virtual_handles[buffer] for buffer in op.inputs),
                iteration_shape=output_type.shape,
                input_maps=input_maps,
                literal=op.literal,
                reduction_axis=op.reduction_axis,
                concat_axis=op.concat_axis,
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
    types: dict[int, TensorType] = {}
    layouts: dict[int, StorageLayout] = {}
    roots: dict[int, int] = {}
    written: set[int] = set()
    input_roots: set[int] = set()
    root_generations: dict[int, int] = {}
    value_generations: dict[int, int] = {}
    next_input_index = 0
    saw_execution = False
    saw_return = False

    for index, op in enumerate(operations):
        if saw_return and not isinstance(op, LoopReturn):
            raise ValueError("loop IR operation appears after return")

        if isinstance(op, LoopAlloc):
            if saw_execution:
                raise ValueError("physical buffer allocation appears after loop execution begins")
            if op.buffer < 0:
                raise ValueError(f"invalid negative physical buffer p{op.buffer}")
            if op.buffer in types:
                raise ValueError(f"physical buffer p{op.buffer} is allocated more than once")
            allocated[op.buffer] = op.type
            types[op.buffer] = op.type
            layouts[op.buffer] = StorageLayout.contiguous(op.type.shape)
            roots[op.buffer] = op.buffer
            root_generations[op.buffer] = 0
            continue

        if isinstance(op, LoopInput):
            saw_execution = True
            if op.output not in allocated:
                raise ValueError(f"loop input destination p{op.output} is not allocated storage")
            if op.index != next_input_index:
                raise ValueError(
                    f"input index {op.index} is not the next dense input index {next_input_index}"
                )
            next_input_index += 1
            root_generations[op.output] += 1
            value_generations[op.output] = root_generations[op.output]
            input_roots.add(op.output)
            written.add(op.output)
            continue

        if isinstance(op, LoopView):
            saw_execution = True
            if op.output < 0:
                raise ValueError(f"invalid negative loop view id p{op.output}")
            if op.output in types:
                raise ValueError(f"loop view p{op.output} collides with an existing loop value")
            if op.source not in types:
                raise ValueError(f"loop view source p{op.source} is not defined")
            if op.source not in written:
                raise ValueError(f"loop view source p{op.source} is not written")
            _verify_fresh_value(op.source, roots, root_generations, value_generations)
            if op.type.dtype != types[op.source].dtype:
                raise ValueError("loop view cannot change storage dtype")
            root = roots[op.source]
            if op.layout is None:
                expected = infer_reshape(types[op.source], op.type.shape)
                if expected != op.type:
                    raise ValueError("loop view type does not match contiguous reshape inference")
                layout = layouts[op.source].reshaped(types[op.source].shape, op.type.shape)
            else:
                layout = op.layout
            layout.validate_bounds(op.type.shape, element_count(allocated[root].shape))
            types[op.output] = op.type
            layouts[op.output] = layout
            roots[op.output] = root
            value_generations[op.output] = value_generations[op.source]
            written.add(op.output)
            continue

        if isinstance(op, LoopCopyInto):
            saw_execution = True
            if op.output < 0:
                raise ValueError(f"invalid negative copy_into result id p{op.output}")
            if op.output in types:
                raise ValueError(f"copy_into result p{op.output} collides with an existing loop value")
            for buffer in (op.root, op.target, op.source):
                if buffer not in types:
                    raise ValueError(f"copy_into input p{buffer} is not defined")
                if buffer not in written:
                    raise ValueError(f"copy_into input p{buffer} is not written")
                _verify_fresh_value(buffer, roots, root_generations, value_generations)
            root = roots[op.root]
            if root not in allocated:
                raise ValueError("copy_into root handle has no owning storage")
            if root in input_roots:
                raise ValueError("copy_into cannot mutate borrowed or copied runtime input storage")
            if types[op.root] != allocated[root] or layouts[op.root] != layouts[root]:
                raise ValueError("copy_into root must be a fresh full-root handle")
            if roots[op.target] != root:
                raise ValueError("copy_into target must alias its owning root")
            if roots[op.source] == root:
                raise ValueError("copy_into source must use a different storage root")
            if types[op.target] != types[op.source]:
                raise ValueError("copy_into target and source types must exactly match")
            if op.type != allocated[root]:
                raise ValueError("copy_into fresh result type must match its owning root")
            if op.layout != layouts[root]:
                raise ValueError("copy_into fresh result must expose the full owning root layout")
            op.layout.validate_bounds(op.type.shape, element_count(allocated[root].shape))

            root_generations[root] += 1
            types[op.output] = op.type
            layouts[op.output] = op.layout
            roots[op.output] = root
            value_generations[op.output] = root_generations[root]
            written.add(op.output)
            continue

        if isinstance(op, LoopBinaryInto):
            saw_execution = True
            if op.output < 0:
                raise ValueError(f"invalid negative binary_into result id p{op.output}")
            if op.output in types:
                raise ValueError(f"binary_into result p{op.output} collides with an existing loop value")
            for buffer in (op.root, op.target, op.source):
                if buffer not in types:
                    raise ValueError(f"binary_into input p{buffer} is not defined")
                if buffer not in written:
                    raise ValueError(f"binary_into input p{buffer} is not written")
                _verify_fresh_value(buffer, roots, root_generations, value_generations)
            root = roots[op.root]
            if root not in allocated:
                raise ValueError("binary_into root handle has no owning storage")
            if root in input_roots:
                raise ValueError("binary_into cannot mutate borrowed or copied runtime input storage")
            if types[op.root] != allocated[root] or layouts[op.root] != layouts[root]:
                raise ValueError("binary_into root must be a fresh full-root handle")
            if roots[op.target] != root:
                raise ValueError("binary_into target must alias its owning root")
            if roots[op.source] == root:
                raise ValueError("binary_into source must use a different storage root")
            if op.operator not in {"add", "mul"}:
                raise ValueError("binary_into operator must be add or mul")
            if types[op.target] != types[op.source]:
                raise ValueError("binary_into target and source types must exactly match")
            if not _layout_is_non_overlapping(types[op.target].shape, layouts[op.target]):
                raise ValueError("binary_into target layout must not overlap itself")
            if op.type != allocated[root]:
                raise ValueError("binary_into fresh result type must match its owning root")
            if op.layout != layouts[root]:
                raise ValueError("binary_into fresh result must expose the full owning root layout")
            op.layout.validate_bounds(op.type.shape, element_count(allocated[root].shape))

            root_generations[root] += 1
            types[op.output] = op.type
            layouts[op.output] = op.layout
            roots[op.output] = root
            value_generations[op.output] = root_generations[root]
            written.add(op.output)
            continue

        if isinstance(op, LoopInplaceBinary):
            saw_execution = True
            if op.output < 0:
                raise ValueError(f"invalid negative binary_inplace result id p{op.output}")
            if op.output in types:
                raise ValueError(f"binary_inplace result p{op.output} collides with an existing loop value")
            for buffer in (op.root, op.source):
                if buffer not in types:
                    raise ValueError(f"binary_inplace input p{buffer} is not defined")
                if buffer not in written:
                    raise ValueError(f"binary_inplace input p{buffer} is not written")
                _verify_fresh_value(buffer, roots, root_generations, value_generations)
            root = roots[op.root]
            if root not in allocated:
                raise ValueError("binary_inplace root handle has no owning storage")
            if root in input_roots:
                raise ValueError("binary_inplace cannot mutate borrowed or copied runtime input storage")
            if types[op.root] != allocated[root] or layouts[op.root] != layouts[root]:
                raise ValueError("binary_inplace root must be a fresh full-root handle")
            if roots[op.source] == root:
                raise ValueError("binary_inplace source must use a different storage root")
            if op.operator not in {"add", "mul"}:
                raise ValueError("binary_inplace operator must be add or mul")
            if types[op.root] != types[op.source] or op.type != types[op.root]:
                raise ValueError("binary_inplace root, source, and result types must exactly match")

            root_generations[root] += 1
            types[op.output] = op.type
            layouts[op.output] = layouts[root]
            roots[op.output] = root
            value_generations[op.output] = root_generations[root]
            written.add(op.output)
            continue

        if isinstance(op, LoopKernel):
            saw_execution = True
            if op.output not in allocated:
                raise ValueError(f"loop output p{op.output} is not allocated storage")
            for buffer in op.inputs:
                if buffer not in types:
                    raise ValueError(f"loop input p{buffer} is not defined")
                if buffer not in written:
                    raise ValueError(f"loop input p{buffer} is read before being written")
                _verify_fresh_value(buffer, roots, root_generations, value_generations)
            output_root = roots[op.output]
            if any(roots[buffer] == output_root for buffer in op.inputs):
                raise ValueError("loop kernels do not permit output/input storage aliasing")

            if op.opcode not in REDUCTION_OPCODES and op.reduction_axis is not None:
                raise ValueError("only reduction loop kernels may carry a reduction axis")
            if op.opcode != "concat" and op.concat_axis is not None:
                raise ValueError("only concatenate loop kernels may carry a concatenate axis")
            output_type = types[op.output]
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
                expected = infer_binary(types[op.inputs[0]], types[op.inputs[1]])
                if expected != output_type:
                    raise ValueError(f"{op.opcode} loop output buffer type does not match inference")
                _verify_index_maps(op, types)
            elif op.opcode == "relu":
                if len(op.inputs) != 1 or len(op.input_maps) != 1 or op.literal is not None:
                    raise ValueError("relu loop requires one input and one index map")
                expected = infer_relu(types[op.inputs[0]])
                if expected != output_type:
                    raise ValueError("relu loop output buffer type does not match inference")
                _verify_index_maps(op, types)
            elif op.opcode == "concat":
                if (
                    len(op.inputs) < 2
                    or op.input_maps
                    or op.literal is not None
                    or op.concat_axis is None
                ):
                    raise ValueError(
                        "concat loop requires at least two inputs, one axis, no index maps, and no literal"
                    )
                expected = infer_concat(tuple(types[buffer] for buffer in op.inputs), op.concat_axis)
                if expected != output_type:
                    raise ValueError("concat loop output buffer type does not match inference")
            elif op.opcode in REDUCTION_OPCODES:
                if len(op.inputs) != 1 or op.input_maps or op.literal is not None:
                    raise ValueError(
                        f"{op.opcode} loop requires one input, no index maps, and no literal"
                    )
                reduction = op.reduction
                if reduction is None:
                    raise RuntimeError("verified reduction loop unexpectedly has no plan")
                expected = infer_reduction(
                    types[op.inputs[0]],
                    reduction.operator,
                    reduction.axis,
                )
                if expected != output_type:
                    raise ValueError(
                        f"{op.opcode} loop output buffer type does not match inference"
                    )
            elif op.opcode == "reshape":
                if len(op.inputs) != 1 or op.input_maps or op.literal is not None:
                    raise ValueError("reshape loop requires one input, no index maps, and no literal")
                expected = infer_reshape(types[op.inputs[0]], output_type.shape)
                if expected != output_type:
                    raise ValueError("reshape loop output buffer type does not match inference")
            elif op.opcode in {"relu_add", "relu_mul"}:
                if len(op.inputs) != 2 or len(op.input_maps) != 2 or op.literal is not None:
                    raise ValueError(
                        f"{op.opcode} loop requires two inputs and two index maps"
                    )
                binary_type = infer_binary(types[op.inputs[0]], types[op.inputs[1]])
                expected = infer_relu(binary_type)
                if expected != output_type:
                    raise ValueError(f"{op.opcode} loop output buffer type does not match inference")
                _verify_index_maps(op, types)
            else:
                expression = fused_expression_for_kernel(op)
                if expression is None:
                    raise ValueError(f"unsupported loop kernel: {op.opcode}")
                _verify_fused_expression(op, expression, types, output_type)

            root_generations[output_root] += 1
            value_generations[op.output] = root_generations[output_root]
            written.add(op.output)
            continue

        if not isinstance(op, LoopReturn):
            raise TypeError(f"unsupported loop IR operation at index {index}")
        if op.buffer not in types:
            raise ValueError(f"returned loop value p{op.buffer} is not defined")
        if op.buffer not in written:
            raise ValueError(f"returned loop value p{op.buffer} is not written")
        _verify_fresh_value(op.buffer, roots, root_generations, value_generations)
        saw_return = True

    if not saw_return:
        raise ValueError("loop IR must end with a return")
    if allocated and set(allocated) != set(range(len(allocated))):
        raise ValueError("physical loop buffer ids must be dense starting at p0")


def _verify_fresh_value(
    buffer: int,
    roots: dict[int, int],
    root_generations: dict[int, int],
    value_generations: dict[int, int],
) -> None:
    root = roots[buffer]
    if value_generations.get(buffer) != root_generations[root]:
        raise ValueError(
            f"stale loop view/alias p{buffer} refers to an older generation of storage p{root}"
        )


def _layout_is_non_overlapping(shape: tuple[int, ...], layout: StorageLayout) -> bool:
    """Conservatively prove that each logical index reaches a distinct storage element."""
    if any(dim == 0 for dim in shape):
        return True
    axes = sorted(
        (
            (abs(stride), dim)
            for dim, stride in zip(shape, layout.strides, strict=True)
            if dim > 1
        ),
        key=lambda item: item[0],
    )
    required_span = 1
    for stride, dim in axes:
        if stride < required_span:
            return False
        required_span *= dim
    return True


def _verify_fused_expression(
    op: LoopKernel,
    expression: fused_expr.FusedExpression,
    types: dict[int, TensorType],
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
    if any(types[buffer].dtype != output_type.dtype for buffer in op.inputs):
        raise ValueError(f"{expression.display_name} loop requires one exact integer dtype")

    refs = {
        name: types[buffer]
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

    _verify_index_maps(op, types)


def _verify_index_maps(op: LoopKernel, types: dict[int, TensorType]) -> None:
    for buffer, index_map in zip(op.inputs, op.input_maps, strict=True):
        expected = _broadcast_index_map(types[buffer].shape, op.iteration_shape)
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
