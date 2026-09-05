from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .inference import (
    infer_binary,
    infer_relu,
    infer_reshape,
    infer_reverse,
    infer_slice,
    infer_sum,
    infer_transpose,
)
from .ir import Module, TensorType, Value
from .layout import StorageLayout, element_count
from .verifier import verify


@dataclass(frozen=True)
class BufferAlloc:
    buffer: int
    type: TensorType


@dataclass(frozen=True)
class BufferInput:
    output: int
    index: int


@dataclass(frozen=True)
class BufferView:
    output: int
    source: int
    slice_axis: int | None = None
    start: int = 0
    stop: int = 0
    step: int = 1
    reverse_axis: int | None = None
    permutation: tuple[int, ...] | None = None


@dataclass(frozen=True)
class BufferCopyInto:
    output: int
    root: int
    target: int
    source: int


@dataclass(frozen=True)
class BufferKernel:
    opcode: str
    output: int
    inputs: tuple[int, ...]
    literal: np.ndarray[Any, Any] | None = None
    reduction_axis: int | None = None


@dataclass(frozen=True)
class BufferReturn:
    buffer: int


BufferOperation = BufferAlloc | BufferInput | BufferView | BufferCopyInto | BufferKernel | BufferReturn


@dataclass(frozen=True)
class BufferAssignment:
    virtual: int
    physical: int
    type: TensorType


@dataclass(frozen=True)
class BufferAlias:
    virtual: int
    source: int
    physical: int
    type: TensorType
    layout: StorageLayout


@dataclass(frozen=True)
class MemoryPlan:
    assignments: tuple[BufferAssignment, ...]
    aliases: tuple[BufferAlias, ...] = ()

    def __post_init__(self) -> None:
        virtuals: set[int] = set()
        physical_types: dict[int, TensorType] = {}
        for assignment in self.assignments:
            if assignment.virtual in virtuals:
                raise ValueError(f"virtual buffer b{assignment.virtual} is assigned more than once")
            if assignment.physical < 0:
                raise ValueError(f"invalid negative physical buffer p{assignment.physical}")
            previous_type = physical_types.get(assignment.physical)
            if previous_type is not None and previous_type != assignment.type:
                raise ValueError(
                    f"physical buffer p{assignment.physical} is assigned incompatible tensor types"
                )
            virtuals.add(assignment.virtual)
            physical_types[assignment.physical] = assignment.type

        if physical_types and set(physical_types) != set(range(len(physical_types))):
            raise ValueError("physical buffer ids must be dense starting at p0")

        alias_virtuals: set[int] = set()
        for alias in self.aliases:
            if alias.virtual in virtuals or alias.virtual in alias_virtuals:
                raise ValueError(f"virtual buffer b{alias.virtual} has multiple memory bindings")
            root_type = physical_types.get(alias.physical)
            if root_type is None:
                raise ValueError(
                    f"view alias b{alias.virtual} refers to missing physical buffer p{alias.physical}"
                )
            if root_type.dtype != alias.type.dtype:
                raise ValueError(f"view alias b{alias.virtual} changes backing storage dtype")
            alias.layout.validate_bounds(alias.type.shape, element_count(root_type.shape))
            alias_virtuals.add(alias.virtual)

    @property
    def physical_count(self) -> int:
        return len({assignment.physical for assignment in self.assignments})

    @property
    def physical_types(self) -> tuple[TensorType, ...]:
        types: dict[int, TensorType] = {}
        for assignment in self.assignments:
            types.setdefault(assignment.physical, assignment.type)
        return tuple(types[slot] for slot in range(len(types)))

    def physical_for(self, virtual: int) -> int:
        for assignment in self.assignments:
            if assignment.virtual == virtual:
                return assignment.physical
        for alias in self.aliases:
            if alias.virtual == virtual:
                return alias.physical
        raise KeyError(f"virtual buffer b{virtual} has no physical assignment")

    def alias_for(self, virtual: int) -> BufferAlias | None:
        return next((alias for alias in self.aliases if alias.virtual == virtual), None)

    def is_alias(self, virtual: int) -> bool:
        return self.alias_for(virtual) is not None

    def dump(self) -> str:
        lines = [
            f"b{assignment.virtual} -> p{assignment.physical} : {assignment.type}"
            for assignment in self.assignments
        ]
        lines.extend(
            f"b{alias.virtual} => p{alias.physical} view(b{alias.source}) "
            f"offset={alias.layout.offset} strides={alias.layout.strides} : {alias.type}"
            for alias in self.aliases
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class CPUProgram:
    operations: tuple[BufferOperation, ...]

    def __post_init__(self) -> None:
        _verify_buffer_ir(self.operations)

    @property
    def allocations(self) -> tuple[BufferAlloc, ...]:
        return tuple(op for op in self.operations if isinstance(op, BufferAlloc))

    @property
    def inputs(self) -> tuple[BufferInput, ...]:
        return tuple(op for op in self.operations if isinstance(op, BufferInput))

    @property
    def views(self) -> tuple[BufferView, ...]:
        return tuple(op for op in self.operations if isinstance(op, BufferView))

    @property
    def copies(self) -> tuple[BufferCopyInto, ...]:
        return tuple(op for op in self.operations if isinstance(op, BufferCopyInto))

    @property
    def instructions(self) -> tuple[BufferKernel, ...]:
        """Compatibility view containing only executable kernel operations."""
        return tuple(op for op in self.operations if isinstance(op, BufferKernel))

    @property
    def return_slots(self) -> tuple[int, ...]:
        return tuple(op.buffer for op in self.operations if isinstance(op, BufferReturn))

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
            if isinstance(op, BufferAlloc):
                lines.append(f"alloc b{op.buffer} : {op.type}")
            elif isinstance(op, BufferInput):
                lines.append(f"b{op.output} = input {op.index}")
            elif isinstance(op, BufferView):
                if op.permutation is not None:
                    lines.append(
                        f"b{op.output} = transpose b{op.source} axes={op.permutation}"
                    )
                elif op.reverse_axis is not None:
                    lines.append(
                        f"b{op.output} = reverse b{op.source} axis={op.reverse_axis}"
                    )
                elif op.slice_axis is None:
                    lines.append(f"b{op.output} = view b{op.source}")
                else:
                    lines.append(
                        f"b{op.output} = slice b{op.source} axis={op.slice_axis} "
                        f"[{op.start}:{op.stop}:{op.step}]"
                    )
            elif isinstance(op, BufferCopyInto):
                lines.append(
                    f"b{op.output} = copy_into root=b{op.root} target=b{op.target} source=b{op.source}"
                )
            elif isinstance(op, BufferKernel):
                if op.opcode == "const":
                    if op.literal is None:
                        raise RuntimeError("verified const kernel unexpectedly has no literal")
                    lines.append(f"b{op.output} = const {_format_literal(op.literal)}")
                else:
                    operands = ", ".join(f"b{buffer}" for buffer in op.inputs)
                    suffix = (
                        ""
                        if op.opcode != "sum" or op.reduction_axis is None
                        else f" axis={op.reduction_axis}"
                    )
                    lines.append(f"b{op.output} = {op.opcode} {operands}{suffix}")
            else:
                lines.append(f"return b{op.buffer}")
        return "\n".join(lines)


def lower_to_cpu(module: Module) -> CPUProgram:
    verify(module)
    buffers: dict[Value, int] = {}
    operations: list[BufferOperation] = []
    next_buffer = 0

    for op in module.function.ops:
        if op.opcode == "return":
            operations.extend(BufferReturn(buffers[operand]) for operand in op.operands)
            continue

        result = op.results[0]
        buffer = next_buffer
        next_buffer += 1
        buffers[result] = buffer
        operations.append(BufferAlloc(buffer, result.type))

        if op.opcode == "input":
            operations.append(BufferInput(output=buffer, index=op.attrs["index"]))
            continue
        if op.opcode == "view":
            operations.append(BufferView(output=buffer, source=buffers[op.operands[0]]))
            continue
        if op.opcode == "slice":
            operations.append(
                BufferView(
                    output=buffer,
                    source=buffers[op.operands[0]],
                    slice_axis=op.attrs["axis"],
                    start=op.attrs["start"],
                    stop=op.attrs["stop"],
                    step=op.attrs["step"],
                )
            )
            continue
        if op.opcode == "reverse":
            operations.append(
                BufferView(
                    output=buffer,
                    source=buffers[op.operands[0]],
                    reverse_axis=op.attrs["axis"],
                )
            )
            continue
        if op.opcode == "transpose":
            operations.append(
                BufferView(
                    output=buffer,
                    source=buffers[op.operands[0]],
                    permutation=op.attrs["axes"],
                )
            )
            continue
        if op.opcode == "copy_into":
            operations.append(
                BufferCopyInto(
                    output=buffer,
                    root=buffers[op.operands[0]],
                    target=buffers[op.operands[1]],
                    source=buffers[op.operands[2]],
                )
            )
            continue

        literal = None
        if op.opcode == "const":
            literal = np.array(op.attrs["value"], copy=True)
            literal.setflags(write=False)
        operations.append(
            BufferKernel(
                opcode=op.opcode,
                output=buffer,
                inputs=tuple(buffers[operand] for operand in op.operands),
                literal=literal,
                reduction_axis=op.attrs.get("axis") if op.opcode == "sum" else None,
            )
        )

    return CPUProgram(tuple(operations))


def plan_memory(program: CPUProgram) -> MemoryPlan:
    """Reuse storage while preserving transitive signed-stride alias lifetimes."""
    allocation_positions: dict[int, int] = {}
    types: dict[int, TensorType] = {}
    last_uses: dict[int, int] = {}
    alias_sources: dict[int, int] = {}

    for index, op in enumerate(program.operations):
        if isinstance(op, BufferAlloc):
            allocation_positions[op.buffer] = index
            types[op.buffer] = op.type
        elif isinstance(op, BufferInput):
            last_uses[op.output] = max(last_uses.get(op.output, -1), index)
        elif isinstance(op, BufferView):
            alias_sources[op.output] = op.source
            last_uses[op.output] = max(last_uses.get(op.output, -1), index)
            last_uses[op.source] = max(last_uses.get(op.source, -1), index)
        elif isinstance(op, BufferCopyInto):
            alias_sources[op.output] = op.root
            for buffer in (op.output, op.root, op.target, op.source):
                last_uses[buffer] = max(last_uses.get(buffer, -1), index)
        elif isinstance(op, BufferKernel):
            last_uses[op.output] = max(last_uses.get(op.output, -1), index)
            for buffer in op.inputs:
                last_uses[buffer] = max(last_uses.get(buffer, -1), index)
        else:
            last_uses[op.buffer] = max(last_uses.get(op.buffer, -1), index)

    def root_virtual(buffer: int) -> int:
        seen: set[int] = set()
        current = buffer
        while current in alias_sources:
            if current in seen:
                raise ValueError("buffer view alias cycle detected")
            seen.add(current)
            current = alias_sources[current]
        return current

    for buffer, last_use in tuple(last_uses.items()):
        root = root_virtual(buffer)
        last_uses[root] = max(last_uses.get(root, -1), last_use)

    physical_state: list[tuple[TensorType, int]] = []
    assignments: list[BufferAssignment] = []

    for buffer, start in allocation_positions.items():
        if buffer in alias_sources:
            continue
        buffer_type = types[buffer]
        end = max(start, last_uses.get(buffer, start))
        physical: int | None = None

        for slot, (slot_type, previous_end) in enumerate(physical_state):
            if slot_type == buffer_type and previous_end < start:
                physical = slot
                break

        if physical is None:
            physical = len(physical_state)
            physical_state.append((buffer_type, end))
        else:
            physical_state[physical] = (buffer_type, end)

        assignments.append(BufferAssignment(buffer, physical, buffer_type))

    layouts: dict[int, StorageLayout] = {}
    alias_outputs = set(alias_sources)
    for op in program.operations:
        if isinstance(op, BufferAlloc) and op.buffer not in alias_outputs:
            layouts[op.buffer] = StorageLayout.contiguous(op.type.shape)
        elif isinstance(op, BufferView):
            source_layout = layouts[op.source]
            source_type = types[op.source]
            output_type = types[op.output]
            if op.permutation is not None:
                layout, inferred_shape = source_layout.permuted(
                    source_type.shape, op.permutation
                )
                if inferred_shape != output_type.shape:
                    raise ValueError("buffer transpose layout shape does not match inferred output")
            elif op.reverse_axis is not None:
                layout = source_layout.reversed(source_type.shape, op.reverse_axis)
                if output_type != source_type:
                    raise ValueError("buffer reverse layout type does not match inferred output")
            elif op.slice_axis is None:
                layout = source_layout.reshaped(source_type.shape, output_type.shape)
            else:
                layout, inferred_shape = source_layout.sliced(
                    source_type.shape,
                    axis=op.slice_axis,
                    start=op.start,
                    stop=op.stop,
                    step=op.step,
                )
                if inferred_shape != output_type.shape:
                    raise ValueError("buffer slice layout shape does not match inferred output")
            layouts[op.output] = layout
        elif isinstance(op, BufferCopyInto):
            layouts[op.output] = layouts[op.root]

    assignment_by_virtual = {assignment.virtual: assignment for assignment in assignments}
    aliases: list[BufferAlias] = []
    for op in program.operations:
        if isinstance(op, BufferView):
            source = op.source
            output = op.output
        elif isinstance(op, BufferCopyInto):
            source = op.root
            output = op.output
        else:
            continue
        root = root_virtual(output)
        root_assignment = assignment_by_virtual.get(root)
        if root_assignment is None:
            raise ValueError(f"view root b{root} has no physical storage assignment")
        aliases.append(
            BufferAlias(
                virtual=output,
                source=source,
                physical=root_assignment.physical,
                type=types[output],
                layout=layouts[output],
            )
        )

    return MemoryPlan(tuple(assignments), tuple(aliases))


def _verify_buffer_ir(operations: tuple[BufferOperation, ...]) -> None:
    allocated: dict[int, TensorType] = {}
    written: set[int] = set()
    alias_sources: dict[int, int] = {}
    roots: dict[int, int] = {}
    root_generations: dict[int, int] = {}
    value_generations: dict[int, int] = {}
    full_root_handles: set[int] = set()
    input_roots: set[int] = set()
    next_input_index = 0
    saw_return = False

    def root_virtual(buffer: int) -> int:
        seen: set[int] = set()
        current = buffer
        while current in alias_sources:
            if current in seen:
                raise ValueError("buffer alias cycle detected")
            seen.add(current)
            current = alias_sources[current]
        return current

    def require_fresh(buffer: int) -> None:
        root = roots.get(buffer)
        if root is None:
            raise ValueError(f"buffer b{buffer} has no storage-generation metadata")
        if value_generations.get(buffer) != root_generations[root]:
            raise ValueError(
                f"stale buffer view/alias b{buffer} refers to an older storage generation"
            )

    for index, op in enumerate(operations):
        if saw_return and not isinstance(op, BufferReturn):
            raise ValueError("buffer IR operation appears after return")

        if isinstance(op, BufferAlloc):
            if op.buffer < 0:
                raise ValueError(f"invalid negative buffer id b{op.buffer}")
            if op.buffer in allocated:
                raise ValueError(f"buffer b{op.buffer} is allocated more than once")
            allocated[op.buffer] = op.type
            continue

        if isinstance(op, BufferInput):
            if op.output not in allocated:
                raise ValueError(f"buffer b{op.output} is not allocated")
            if op.output in written:
                raise ValueError(f"buffer b{op.output} is written more than once")
            if op.index != next_input_index:
                raise ValueError(
                    f"input index {op.index} is not the next dense input index {next_input_index}"
                )
            next_input_index += 1
            roots[op.output] = op.output
            root_generations[op.output] = 1
            value_generations[op.output] = 1
            full_root_handles.add(op.output)
            input_roots.add(op.output)
            written.add(op.output)
            continue

        if isinstance(op, BufferView):
            if op.output not in allocated or op.source not in allocated:
                raise ValueError("buffer view requires allocated source and output values")
            if op.output in written:
                raise ValueError(f"buffer b{op.output} is written more than once")
            if op.source not in written:
                raise ValueError(f"buffer b{op.source} is viewed before being written")
            require_fresh(op.source)
            mode_count = sum(
                transform is not None
                for transform in (op.slice_axis, op.reverse_axis, op.permutation)
            )
            if mode_count > 1:
                raise ValueError("buffer view cannot combine slice, reverse, and transpose transforms")
            output_type = allocated[op.output]
            if op.permutation is not None:
                expected = infer_transpose(allocated[op.source], op.permutation)
            elif op.reverse_axis is not None:
                expected = infer_reverse(allocated[op.source], op.reverse_axis)
            elif op.slice_axis is None:
                expected = infer_reshape(allocated[op.source], output_type.shape)
            else:
                expected = infer_slice(
                    allocated[op.source],
                    axis=op.slice_axis,
                    start=op.start,
                    stop=op.stop,
                    step=op.step,
                )
            if expected != output_type:
                raise ValueError("buffer view output type does not match inference")
            alias_sources[op.output] = op.source
            roots[op.output] = roots[op.source]
            value_generations[op.output] = value_generations[op.source]
            written.add(op.output)
            continue

        if isinstance(op, BufferCopyInto):
            for buffer in (op.output, op.root, op.target, op.source):
                if buffer not in allocated:
                    raise ValueError("copy_into requires allocated logical buffer values")
            if op.output in written:
                raise ValueError(f"buffer b{op.output} is written more than once")
            for buffer in (op.root, op.target, op.source):
                if buffer not in written:
                    raise ValueError(f"copy_into reads b{buffer} before it is written")
                require_fresh(buffer)
            owner = roots[op.root]
            if op.root not in full_root_handles:
                raise ValueError("copy_into root must be a fresh full-root buffer handle")
            if owner in input_roots:
                raise ValueError("copy_into root must use internal computed storage")
            if roots[op.target] != owner:
                raise ValueError("copy_into target must alias its owning root storage")
            if roots[op.source] == owner:
                raise ValueError("copy_into source must use a different storage root")
            if allocated[op.root] != allocated[owner]:
                raise ValueError("copy_into root handle type must match owning storage")
            if allocated[op.output] != allocated[op.root]:
                raise ValueError("copy_into result type must match its root handle type")
            if allocated[op.target] != allocated[op.source]:
                raise ValueError("copy_into target and source types must exactly match")
            alias_sources[op.output] = op.root
            root_generations[owner] += 1
            roots[op.output] = owner
            value_generations[op.output] = root_generations[owner]
            full_root_handles.add(op.output)
            written.add(op.output)
            continue

        if isinstance(op, BufferKernel):
            if op.output not in allocated:
                raise ValueError(f"buffer b{op.output} is not allocated")
            if op.output in written:
                raise ValueError(f"buffer b{op.output} is written more than once")
            for buffer in op.inputs:
                if buffer not in allocated:
                    raise ValueError(f"buffer b{buffer} is not allocated")
                if buffer not in written:
                    raise ValueError(f"buffer b{buffer} is read before being written")
                require_fresh(buffer)

            if op.opcode != "sum" and op.reduction_axis is not None:
                raise ValueError("only sum kernels may carry a reduction axis")
            output_type = allocated[op.output]
            if op.opcode == "const":
                if op.inputs:
                    raise ValueError("const kernel must not have input buffers")
                if op.literal is None:
                    raise ValueError("const kernel is missing its literal")
                literal = np.asarray(op.literal)
                if tuple(literal.shape) != output_type.shape:
                    raise ValueError("const kernel literal shape does not match output buffer")
                if literal.dtype != output_type.dtype.to_numpy():
                    raise ValueError("const kernel literal dtype does not match output buffer")
            elif op.opcode in {"add", "mul"}:
                if len(op.inputs) != 2 or op.literal is not None:
                    raise ValueError(f"{op.opcode} kernel requires two inputs and no literal")
                expected = infer_binary(allocated[op.inputs[0]], allocated[op.inputs[1]])
                if expected != output_type:
                    raise ValueError(f"{op.opcode} kernel output buffer type does not match inference")
            elif op.opcode == "relu":
                if len(op.inputs) != 1 or op.literal is not None:
                    raise ValueError("relu kernel requires one input and no literal")
                expected = infer_relu(allocated[op.inputs[0]])
                if expected != output_type:
                    raise ValueError("relu kernel output buffer type does not match inference")
            elif op.opcode == "sum":
                if len(op.inputs) != 1 or op.literal is not None:
                    raise ValueError("sum kernel requires one input and no literal")
                expected = infer_sum(allocated[op.inputs[0]], op.reduction_axis)
                if expected != output_type:
                    raise ValueError("sum kernel output buffer type does not match inference")
            elif op.opcode == "reshape":
                if len(op.inputs) != 1 or op.literal is not None:
                    raise ValueError("reshape kernel requires one input and no literal")
                expected = infer_reshape(allocated[op.inputs[0]], output_type.shape)
                if expected != output_type:
                    raise ValueError("reshape kernel output buffer type does not match inference")
            else:
                raise ValueError(f"unsupported buffer kernel: {op.opcode}")

            roots[op.output] = op.output
            root_generations[op.output] = 1
            value_generations[op.output] = 1
            full_root_handles.add(op.output)
            written.add(op.output)
            continue

        if not isinstance(op, BufferReturn):
            raise TypeError(f"unsupported buffer IR operation at index {index}")
        if op.buffer not in allocated:
            raise ValueError(f"buffer b{op.buffer} is not allocated")
        if op.buffer not in written:
            raise ValueError(f"buffer b{op.buffer} is returned before being written")
        require_fresh(op.buffer)
        saw_return = True

    if not saw_return:
        raise ValueError("buffer IR must end with a return")


def _format_literal(value: np.ndarray[Any, Any]) -> str:
    if value.ndim == 0:
        return repr(value.item())
    return repr(value.tolist())