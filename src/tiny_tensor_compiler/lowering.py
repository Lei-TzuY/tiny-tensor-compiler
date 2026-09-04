from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .inference import infer_binary, infer_relu
from .ir import Module, TensorType, Value
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
class BufferKernel:
    opcode: str
    output: int
    inputs: tuple[int, ...]
    literal: np.ndarray[Any, Any] | None = None


@dataclass(frozen=True)
class BufferReturn:
    buffer: int


BufferOperation = BufferAlloc | BufferInput | BufferKernel | BufferReturn


@dataclass(frozen=True)
class BufferAssignment:
    virtual: int
    physical: int
    type: TensorType


@dataclass(frozen=True)
class MemoryPlan:
    assignments: tuple[BufferAssignment, ...]

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
        raise KeyError(f"virtual buffer b{virtual} has no physical assignment")

    def dump(self) -> str:
        return "\n".join(
            f"b{assignment.virtual} -> p{assignment.physical} : {assignment.type}"
            for assignment in self.assignments
        )


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
            elif isinstance(op, BufferKernel):
                if op.opcode == "const":
                    if op.literal is None:
                        raise RuntimeError("verified const kernel unexpectedly has no literal")
                    lines.append(f"b{op.output} = const {_format_literal(op.literal)}")
                else:
                    operands = ", ".join(f"b{buffer}" for buffer in op.inputs)
                    lines.append(f"b{op.output} = {op.opcode} {operands}")
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
            )
        )

    return CPUProgram(tuple(operations))


def plan_memory(program: CPUProgram) -> MemoryPlan:
    """Greedily reuse same-typed physical slots after virtual-buffer lifetimes end."""
    allocation_positions: dict[int, int] = {}
    types: dict[int, TensorType] = {}
    last_uses: dict[int, int] = {}

    for index, op in enumerate(program.operations):
        if isinstance(op, BufferAlloc):
            allocation_positions[op.buffer] = index
            types[op.buffer] = op.type
        elif isinstance(op, BufferInput):
            last_uses[op.output] = max(last_uses.get(op.output, -1), index)
        elif isinstance(op, BufferKernel):
            last_uses[op.output] = max(last_uses.get(op.output, -1), index)
            for buffer in op.inputs:
                last_uses[buffer] = max(last_uses.get(buffer, -1), index)
        else:
            last_uses[op.buffer] = max(last_uses.get(op.buffer, -1), index)

    physical_state: list[tuple[TensorType, int]] = []
    assignments: list[BufferAssignment] = []

    for buffer, start in allocation_positions.items():
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

    return MemoryPlan(tuple(assignments))


def _verify_buffer_ir(operations: tuple[BufferOperation, ...]) -> None:
    allocated: dict[int, TensorType] = {}
    written: set[int] = set()
    next_input_index = 0
    saw_return = False

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
            else:
                raise ValueError(f"unsupported buffer kernel: {op.opcode}")

            written.add(op.output)
            continue

        if not isinstance(op, BufferReturn):
            raise TypeError(f"unsupported buffer IR operation at index {index}")
        if op.buffer not in allocated:
            raise ValueError(f"buffer b{op.buffer} is not allocated")
        if op.buffer not in written:
            raise ValueError(f"buffer b{op.buffer} is returned before being written")
        saw_return = True

    if not saw_return:
        raise ValueError("buffer IR must end with a return")


def _format_literal(value: np.ndarray[Any, Any]) -> str:
    if value.ndim == 0:
        return repr(value.item())
    return repr(value.tolist())
