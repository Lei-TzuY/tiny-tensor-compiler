from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from .ir import TensorType
from .loop_ir import LoopInput, LoopProgram


@dataclass(frozen=True)
class BorrowedInput:
    """One runtime input proven safe to bind directly to a physical read slot."""

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
    """A verified LoopProgram plus external inputs that may bypass materialization copies."""

    program: LoopProgram
    borrowed_inputs: tuple[BorrowedInput, ...]

    def __post_init__(self) -> None:
        input_ops = tuple(self.program.inputs)
        input_by_index = {op.index: op for op in input_ops}
        slot_counts = Counter(op.output for op in input_ops)
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
            if slot_counts[binding.buffer] != 1:
                raise ValueError(
                    f"borrowed physical buffer p{binding.buffer} is shared by multiple inputs"
                )
            if binding.buffer in kernel_outputs:
                raise ValueError(
                    f"borrowed physical buffer p{binding.buffer} is later written by a kernel"
                )
            if types[binding.buffer] != binding.type:
                raise ValueError(
                    f"borrowed runtime input {binding.index} type does not match its slot"
                )
            seen_indices.add(binding.index)
            seen_buffers.add(binding.buffer)

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
    def input_types(self) -> InputTypeContract:
        types = tuple(self.program.input_types)
        borrowed = self.borrowed_input_indices
        return InputTypeContract(
            types=types,
            borrow_mask=tuple(index in borrowed for index in range(len(types))),
        )

    @property
    def kernels(self):
        return self.program.kernels

    @property
    def return_slots(self):
        return self.program.return_slots

    @property
    def return_slot(self):
        return self.program.return_slot

    @property
    def borrowed_input_indices(self) -> frozenset[int]:
        return frozenset(binding.index for binding in self.borrowed_inputs)

    @property
    def borrowed_input_slots(self) -> frozenset[int]:
        return frozenset(binding.buffer for binding in self.borrowed_inputs)

    def dump(self) -> str:
        return self.program.dump()


def borrow_inputs(program: LoopProgram) -> BorrowedLoopProgram:
    """Borrow every runtime input whose planned physical slot is never reused for a write."""
    input_ops = tuple(program.inputs)
    slot_counts = Counter(op.output for op in input_ops)
    kernel_outputs = {kernel.output for kernel in program.kernels}
    types = {alloc.buffer: alloc.type for alloc in program.allocations}

    bindings = tuple(
        BorrowedInput(index=op.index, buffer=op.output, type=types[op.output])
        for op in input_ops
        if slot_counts[op.output] == 1 and op.output not in kernel_outputs
    )
    return BorrowedLoopProgram(program=program, borrowed_inputs=bindings)


def unwrap_loop_program(program: LoopProgram | BorrowedLoopProgram) -> LoopProgram:
    return program.program if isinstance(program, BorrowedLoopProgram) else program


def borrowed_slots(program: LoopProgram | BorrowedLoopProgram) -> frozenset[int]:
    if isinstance(program, BorrowedLoopProgram):
        return program.borrowed_input_slots
    return frozenset()
