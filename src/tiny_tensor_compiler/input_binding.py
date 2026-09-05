from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from .ir import TensorType
from .loop_ir import LoopAlloc, LoopInput, LoopKernel, LoopProgram, LoopReturn, LoopView


@dataclass(frozen=True)
class BorrowedInput:
    """One runtime input bound directly to a dedicated read-only physical slot."""

    index: int
    buffer: int
    type: TensorType


@dataclass(frozen=True)
class InputTypeContract(Sequence[TensorType]):
    """Runtime input types plus a per-input zero-copy validation contract."""

    types: tuple[TensorType, ...]
    borrow_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.types) != len(self.borrow_mask):
            raise ValueError("input type contract and borrow mask must have equal length")

    def __len__(self) -> int:
        return len(self.types)

    def __getitem__(self, index):
        return self.types[index]

    def __iter__(self) -> Iterator[TensorType]:
        return iter(self.types)


@dataclass(frozen=True)
class BorrowedLoopProgram:
    """Verified Loop IR whose external-input epochs are safe to bind without copies."""

    program: LoopProgram
    borrowed_inputs: tuple[BorrowedInput, ...]

    def __post_init__(self) -> None:
        input_by_index = {op.index: op for op in self.program.inputs}
        types = {alloc.buffer: alloc.type for alloc in self.program.allocations}
        kernel_outputs = {kernel.output for kernel in self.program.kernels}

        seen_indices: set[int] = set()
        seen_buffers: set[int] = set()
        for binding in self.borrowed_inputs:
            if binding.index in seen_indices:
                raise ValueError(f"runtime input {binding.index} is borrowed more than once")
            if binding.buffer in seen_buffers:
                raise ValueError(f"physical buffer p{binding.buffer} is borrowed more than once")
            op = input_by_index.get(binding.index)
            if op is None or op.output != binding.buffer:
                raise ValueError(
                    f"borrowed runtime input {binding.index} does not match its LoopInput slot"
                )
            if binding.buffer in kernel_outputs:
                raise ValueError(
                    f"borrowed physical buffer p{binding.buffer} is written by a kernel"
                )
            if types[binding.buffer] != binding.type:
                raise ValueError(
                    f"borrowed runtime input {binding.index} type does not match its slot"
                )
            seen_indices.add(binding.index)
            seen_buffers.add(binding.buffer)

        expected_indices = set(range(len(self.program.inputs)))
        if seen_indices != expected_indices:
            raise ValueError("borrowed loop programs must bind every runtime input")

    @property
    def operations(self):
        return self.program.operations

    @property
    def allocations(self):
        return self.program.allocations

    @property
    def inputs(self):
        return self.program.inputs

    @property
    def views(self):
        return self.program.views

    @property
    def value_types(self):
        return self.program.value_types

    @property
    def input_types(self) -> InputTypeContract:
        types = tuple(self.program.input_types)
        return InputTypeContract(types=types, borrow_mask=(True,) * len(types))

    @property
    def kernels(self):
        return self.program.kernels

    @property
    def return_slots(self):
        return self.program.return_slots

    @property
    def return_slot(self):
        return self.program.return_slot

    def storage_root(self, buffer: int) -> int:
        return self.program.storage_root(buffer)

    @property
    def borrowed_input_indices(self) -> frozenset[int]:
        return frozenset(binding.index for binding in self.borrowed_inputs)

    @property
    def borrowed_input_slots(self) -> frozenset[int]:
        return frozenset(binding.buffer for binding in self.borrowed_inputs)

    def dump(self) -> str:
        return self.program.dump()


def borrow_inputs(program: LoopProgram) -> BorrowedLoopProgram:
    """Split reused input lifetimes while preserving logical view handles."""
    types = {alloc.buffer: alloc.type for alloc in program.allocations}
    storage_count = len(types)
    operations = program.operations
    split_positions = {
        position
        for position, op in enumerate(operations)
        if isinstance(op, LoopInput) and _has_other_write(operations, position, op.output)
    }
    split_count = len(split_positions)
    next_buffer = storage_count
    extra_allocations: list[LoopAlloc] = []
    transformed_operations = []
    active_aliases: dict[int, int] = {}
    bindings: list[BorrowedInput] = []

    def remap_handle(buffer: int) -> int:
        if buffer >= storage_count:
            return buffer + split_count
        return active_aliases.get(buffer, buffer)

    for position, op in enumerate(operations):
        if isinstance(op, LoopAlloc):
            continue

        if isinstance(op, LoopInput):
            active_aliases.pop(op.output, None)
            destination = op.output
            if position in split_positions:
                destination = next_buffer
                next_buffer += 1
                extra_allocations.append(LoopAlloc(destination, types[op.output]))
                active_aliases[op.output] = destination

            transformed_operations.append(LoopInput(destination, op.index))
            bindings.append(
                BorrowedInput(
                    index=op.index,
                    buffer=destination,
                    type=types[op.output],
                )
            )
            continue

        if isinstance(op, LoopView):
            transformed_operations.append(
                LoopView(
                    output=op.output + split_count,
                    source=remap_handle(op.source),
                    type=op.type,
                )
            )
            continue

        if isinstance(op, LoopKernel):
            transformed_operations.append(
                LoopKernel(
                    opcode=op.opcode,
                    output=op.output,
                    inputs=tuple(remap_handle(buffer) for buffer in op.inputs),
                    iteration_shape=op.iteration_shape,
                    input_maps=op.input_maps,
                    literal=op.literal,
                    fused_expression=op.fused_expression,
                )
            )
            active_aliases.pop(op.output, None)
            continue

        if isinstance(op, LoopReturn):
            transformed_operations.append(LoopReturn(remap_handle(op.buffer)))
            continue

        raise TypeError("unsupported Loop IR operation during input borrowing")

    transformed = LoopProgram(
        (*program.allocations, *extra_allocations, *transformed_operations)
    )
    return BorrowedLoopProgram(program=transformed, borrowed_inputs=tuple(bindings))


def borrowed_slots(program: LoopProgram | BorrowedLoopProgram) -> frozenset[int]:
    if isinstance(program, BorrowedLoopProgram):
        return program.borrowed_input_slots
    return frozenset()


def _has_other_write(operations, position: int, buffer: int) -> bool:
    for other_position, other in enumerate(operations):
        if other_position == position:
            continue
        if isinstance(other, LoopInput) and other.output == buffer:
            return True
        if isinstance(other, LoopKernel) and other.output == buffer:
            return True
    return False
