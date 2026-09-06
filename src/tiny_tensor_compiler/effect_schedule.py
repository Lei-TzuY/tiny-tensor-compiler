from __future__ import annotations

from dataclasses import dataclass

from .loop_ir import LoopBinaryInto, LoopCopyInto, LoopInplaceBinary, LoopProgram

EffectOperation = LoopCopyInto | LoopBinaryInto | LoopInplaceBinary


@dataclass(frozen=True)
class EffectAccess:
    """Storage-root reads and writes performed by one verified mutation effect."""

    reads: frozenset[int]
    writes: frozenset[int]


@dataclass(frozen=True)
class ParallelEffectGroup:
    """One consecutive effect group that may execute under a shared barrier."""

    operation_indices: tuple[int, ...]
    effects: tuple[EffectOperation, ...]

    def __post_init__(self) -> None:
        if not self.operation_indices or len(self.operation_indices) != len(self.effects):
            raise ValueError("parallel effect groups require matching non-empty indices/effects")
        if self.operation_indices != tuple(
            range(self.operation_indices[0], self.operation_indices[0] + len(self.effects))
        ):
            raise ValueError("parallel effect groups must contain consecutive operations")

    @property
    def start(self) -> int:
        return self.operation_indices[0]


def plan_parallel_effect_groups(program: LoopProgram) -> tuple[ParallelEffectGroup, ...]:
    """Partition consecutive mutation effects into deterministic hazard-free groups.

    This planner never moves an effect across a non-effect operation. Within one consecutive
    run it greedily keeps effects together while every member is pairwise independent at the
    storage-root level. A conflict closes the current group and starts the next barrier level.
    """

    groups: list[ParallelEffectGroup] = []
    indices: list[int] = []
    effects: list[EffectOperation] = []
    accesses: list[EffectAccess] = []

    def flush() -> None:
        if not effects:
            return
        groups.append(ParallelEffectGroup(tuple(indices), tuple(effects)))
        indices.clear()
        effects.clear()
        accesses.clear()

    for index, op in enumerate(program.operations):
        if not isinstance(op, (LoopCopyInto, LoopBinaryInto, LoopInplaceBinary)):
            flush()
            continue

        access = effect_access(program, op)
        if any(effect_accesses_conflict(access, previous) for previous in accesses):
            flush()
        indices.append(index)
        effects.append(op)
        accesses.append(access)

    flush()
    return tuple(groups)


def effect_access(program: LoopProgram, op: EffectOperation) -> EffectAccess:
    """Return the conservative root-level dependence footprint of one effect."""

    write_root = program.storage_root(op.root)
    source_root = program.storage_root(op.source)
    return EffectAccess(
        reads=frozenset((write_root, source_root)),
        writes=frozenset((write_root,)),
    )


def effect_accesses_conflict(lhs: EffectAccess, rhs: EffectAccess) -> bool:
    """Return whether two effects have a RAW, WAR, or WAW storage-root hazard."""

    return bool(
        lhs.writes & (rhs.reads | rhs.writes)
        or rhs.writes & (lhs.reads | lhs.writes)
    )
