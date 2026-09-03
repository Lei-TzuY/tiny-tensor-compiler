from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .ir import Module, TensorType, Value
from .verifier import verify


@dataclass(frozen=True)
class Instruction:
    opcode: str
    output: int
    inputs: tuple[int, ...]
    result_type: TensorType
    literal: np.ndarray[Any, Any] | None = None


@dataclass(frozen=True)
class CPUProgram:
    instructions: tuple[Instruction, ...]
    return_slot: int

    def dump(self) -> str:
        lines: list[str] = []
        for inst in self.instructions:
            if inst.opcode == "const":
                literal = (
                    repr(inst.literal.item())
                    if inst.literal.ndim == 0
                    else repr(inst.literal.tolist())
                )
                lines.append(f"b{inst.output} = const {literal} : {inst.result_type}")
            else:
                operands = ", ".join(f"b{slot}" for slot in inst.inputs)
                lines.append(f"b{inst.output} = {inst.opcode} {operands} : {inst.result_type}")
        lines.append(f"return b{self.return_slot}")
        return "\n".join(lines)


def lower_to_cpu(module: Module) -> CPUProgram:
    verify(module)
    slots: dict[Value, int] = {}
    instructions: list[Instruction] = []
    next_slot = 0
    return_slot: int | None = None

    for op in module.function.ops:
        if op.opcode == "return":
            return_slot = slots[op.operands[0]]
            continue
        result = op.results[0]
        slot = next_slot
        next_slot += 1
        slots[result] = slot
        literal = None
        if op.opcode == "const":
            literal = np.array(op.attrs["value"], copy=True)
            literal.setflags(write=False)
        instructions.append(
            Instruction(
                opcode=op.opcode,
                output=slot,
                inputs=tuple(slots[operand] for operand in op.operands),
                result_type=result.type,
                literal=literal,
            )
        )

    if return_slot is None:
        raise RuntimeError("verified module unexpectedly has no return")
    return CPUProgram(tuple(instructions), return_slot)
