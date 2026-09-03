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
class BufferKernel:
    opcode: str
    output: int
    inputs: tuple[int, ...]
    literal: np.ndarray[Any, Any] | None = None


@dataclass(frozen=True)
class BufferReturn:
    buffer: int


BufferOperation = BufferAlloc | BufferKernel | BufferReturn


@dataclass(frozen=True)
class CPUProgram:
    operations: tuple[BufferOperation, ...]

    def __post_init__(self) -> None:
        _verify_buffer_ir(self.operations)

    @property
    def instructions(self) -> tuple[BufferKernel, ...]:
        """Compatibility view containing only executable kernel operations."""
        return tuple(op for op in self.operations if isinstance(op, BufferKernel))

    @property
    def return_slot(self) -> int:
        """Compatibility alias for the returned virtual buffer."""
        for op in reversed(self.operations):
            if isinstance(op, BufferReturn):
                return op.buffer
        raise RuntimeError("verified buffer IR unexpectedly has no return")

    def dump(self) -> str:
        lines: list[str] = []
        for op in self.operations:
            if isinstance(op, BufferAlloc):
                lines.append(f"alloc b{op.buffer} : {op.type}")
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
            operations.append(BufferReturn(buffers[op.operands[0]]))
            continue

        result = op.results[0]
        buffer = next_buffer
        next_buffer += 1
        buffers[result] = buffer
        operations.append(BufferAlloc(buffer, result.type))

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


def _verify_buffer_ir(operations: tuple[BufferOperation, ...]) -> None:
    allocated: dict[int, TensorType] = {}
    written: set[int] = set()
    saw_return = False

    for index, op in enumerate(operations):
        if saw_return:
            raise ValueError("buffer IR operation appears after return")

        if isinstance(op, BufferAlloc):
            if op.buffer < 0:
                raise ValueError(f"invalid negative buffer id b{op.buffer}")
            if op.buffer in allocated:
                raise ValueError(f"buffer b{op.buffer} is allocated more than once")
            allocated[op.buffer] = op.type
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
