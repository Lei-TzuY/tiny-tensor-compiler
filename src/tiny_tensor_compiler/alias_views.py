from __future__ import annotations

from .input_binding import BorrowedLoopProgram
from .loop_ir import LoopInput, LoopKernel, LoopProgram, LoopReturn, LoopView


def alias_contiguous_reshapes(
    program: LoopProgram | BorrowedLoopProgram,
) -> LoopProgram | BorrowedLoopProgram:
    """Replace safe contiguous reshape copies with verified read-only alias views."""
    if isinstance(program, BorrowedLoopProgram):
        transformed = _alias_loop_program(program.program)
        if transformed is program.program:
            return program
        return BorrowedLoopProgram(
            program=transformed,
            borrowed_inputs=program.borrowed_inputs,
        )
    return _alias_loop_program(program)


def _alias_loop_program(program: LoopProgram) -> LoopProgram:
    if program.views:
        return program

    operations = program.operations
    types = program.buffer_types
    roots = {buffer: program.storage_root(buffer) for buffer in types}
    next_buffer = max(types, default=-1) + 1
    active_aliases: dict[int, int] = {}
    transformed = []
    changed = False

    for position, op in enumerate(operations):
        if isinstance(op, LoopInput):
            active_aliases.pop(op.output, None)
            transformed.append(op)
            continue

        if isinstance(op, LoopKernel):
            rewritten_inputs = tuple(active_aliases.get(buffer, buffer) for buffer in op.inputs)
            if op.opcode == "reshape":
                source = rewritten_inputs[0]
                last_use = _last_epoch_use(operations, position, op.output)
                source_root = roots[source]
                if not _has_storage_write(
                    operations,
                    start=position + 1,
                    end=last_use,
                    storage_root=source_root,
                ):
                    view = LoopView(
                        output=next_buffer,
                        source=source,
                        type=types[op.output],
                    )
                    next_buffer += 1
                    roots[view.output] = source_root
                    types[view.output] = view.type
                    active_aliases[op.output] = view.output
                    transformed.append(view)
                    changed = True
                    continue

            active_aliases.pop(op.output, None)
            transformed.append(
                LoopKernel(
                    opcode=op.opcode,
                    output=op.output,
                    inputs=rewritten_inputs,
                    iteration_shape=op.iteration_shape,
                    input_maps=op.input_maps,
                    literal=op.literal,
                    fused_expression=op.fused_expression,
                )
            )
            continue

        if isinstance(op, LoopReturn):
            transformed.append(LoopReturn(active_aliases.get(op.buffer, op.buffer)))
            continue

        transformed.append(op)

    if not changed:
        return program
    return LoopProgram(tuple(transformed))


def _last_epoch_use(operations, definition: int, buffer: int) -> int:
    last_use = definition
    for position in range(definition + 1, len(operations)):
        op = operations[position]
        if _writes(op, buffer):
            break
        if _reads(op, buffer):
            last_use = position
    return last_use


def _has_storage_write(
    operations,
    *,
    start: int,
    end: int,
    storage_root: int,
) -> bool:
    for position in range(start, end + 1):
        op = operations[position]
        if _writes(op, storage_root):
            return True
    return False


def _writes(op, buffer: int) -> bool:
    if isinstance(op, LoopInput):
        return op.output == buffer
    if isinstance(op, LoopKernel):
        return op.output == buffer
    return False


def _reads(op, buffer: int) -> bool:
    if isinstance(op, LoopKernel):
        return buffer in op.inputs
    if isinstance(op, LoopReturn):
        return op.buffer == buffer
    if isinstance(op, LoopView):
        return op.source == buffer
    return False
