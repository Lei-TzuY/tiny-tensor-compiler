from __future__ import annotations

from dataclasses import dataclass
from math import gcd, prod

from .layout import StorageLayout
from .loop_ir import LoopBinaryInto, LoopCopyInto, LoopInplaceBinary, LoopProgram, LoopView

EffectOperation = LoopCopyInto | LoopBinaryInto | LoopInplaceBinary


@dataclass(frozen=True)
class StorageRegion:
    """One concrete logical region addressed relative to a physical storage root."""

    root: int
    shape: tuple[int, ...]
    layout: StorageLayout


@dataclass(frozen=True)
class EffectAccess:
    """Concrete storage regions read and written by one verified mutation effect."""

    reads: tuple[StorageRegion, ...]
    writes: tuple[StorageRegion, ...]


@dataclass(frozen=True)
class ParallelEffectGroup:
    """One effect group that may execute under a shared barrier.

    Effects remain in original program order. Alias-only ``LoopView`` operations may appear
    between grouped effects and are replayed after the sections barrier; no executable kernel
    or other operation is crossed.
    """

    operation_indices: tuple[int, ...]
    effects: tuple[EffectOperation, ...]
    transparent_indices: tuple[int, ...] = ()
    transparent_views: tuple[LoopView, ...] = ()

    def __post_init__(self) -> None:
        if not self.operation_indices or len(self.operation_indices) != len(self.effects):
            raise ValueError("parallel effect groups require matching non-empty indices/effects")
        if any(
            later <= earlier
            for earlier, later in zip(
                self.operation_indices,
                self.operation_indices[1:],
                strict=False,
            )
        ):
            raise ValueError("parallel effect indices must be strictly increasing")
        if len(self.transparent_indices) != len(self.transparent_views):
            raise ValueError("parallel effect groups require matching transparent views")
        if any(
            index <= self.operation_indices[0] or index >= self.operation_indices[-1]
            for index in self.transparent_indices
        ):
            raise ValueError("transparent views must appear between grouped effects")
        if tuple(sorted(self.transparent_indices)) != self.transparent_indices:
            raise ValueError("transparent view indices must be ordered")
        if set(self.operation_indices) & set(self.transparent_indices):
            raise ValueError("effect and transparent-view indices must be disjoint")

    @property
    def start(self) -> int:
        return self.operation_indices[0]

    @property
    def consumed_indices(self) -> tuple[int, ...]:
        """Program indices emitted by this group instead of the ordinary operation loop."""
        return tuple(sorted((*self.operation_indices, *self.transparent_indices)))


def plan_parallel_effect_groups(program: LoopProgram) -> tuple[ParallelEffectGroup, ...]:
    """Partition verified mutation effects into deterministic hazard-free groups.

    Only alias-only ``LoopView`` operations may be crossed. A normal kernel, input, return, or
    allocation remains a hard scheduling boundary. Within one eligible run, effects stay in
    one group only while their concrete read/write regions are pairwise independent.
    """

    groups: list[ParallelEffectGroup] = []
    indices: list[int] = []
    effects: list[EffectOperation] = []
    accesses: list[EffectAccess] = []
    transparent_indices: list[int] = []
    transparent_views: list[LoopView] = []
    pending_indices: list[int] = []
    pending_views: list[LoopView] = []

    def clear_pending() -> None:
        pending_indices.clear()
        pending_views.clear()

    def flush() -> None:
        if effects:
            groups.append(
                ParallelEffectGroup(
                    operation_indices=tuple(indices),
                    effects=tuple(effects),
                    transparent_indices=tuple(transparent_indices),
                    transparent_views=tuple(transparent_views),
                )
            )
        indices.clear()
        effects.clear()
        accesses.clear()
        transparent_indices.clear()
        transparent_views.clear()
        clear_pending()

    for index, op in enumerate(program.operations):
        if isinstance(op, LoopView):
            if effects:
                pending_indices.append(index)
                pending_views.append(op)
            continue

        if not isinstance(op, (LoopCopyInto, LoopBinaryInto, LoopInplaceBinary)):
            flush()
            continue

        access = effect_access(program, op)
        if effects and any(effect_accesses_conflict(access, previous) for previous in accesses):
            flush()

        if effects:
            transparent_indices.extend(pending_indices)
            transparent_views.extend(pending_views)
        clear_pending()
        indices.append(index)
        effects.append(op)
        accesses.append(access)

    flush()
    return tuple(groups)


def effect_access(program: LoopProgram, op: EffectOperation) -> EffectAccess:
    """Return the concrete dependence footprint of one verified mutation effect."""

    source = _value_region(program, op.source)
    if isinstance(op, LoopCopyInto):
        target = _value_region(program, op.target)
        return EffectAccess(reads=(source,), writes=(target,))
    if isinstance(op, LoopBinaryInto):
        target = _value_region(program, op.target)
        return EffectAccess(reads=(target, source), writes=(target,))
    if isinstance(op, LoopInplaceBinary):
        root = _root_region(program, op.root)
        return EffectAccess(reads=(root, source), writes=(root,))
    raise TypeError("unsupported write effect for dependence analysis")


def effect_accesses_conflict(lhs: EffectAccess, rhs: EffectAccess) -> bool:
    """Return whether two effects have a conservatively proven RAW, WAR, or WAW hazard."""

    return any(
        _regions_may_overlap(write, other)
        for write in lhs.writes
        for other in (*rhs.reads, *rhs.writes)
    ) or any(
        _regions_may_overlap(write, other)
        for write in rhs.writes
        for other in (*lhs.reads, *lhs.writes)
    )


def regions_provably_disjoint(lhs: StorageRegion, rhs: StorageRegion) -> bool:
    """Prove two regions disjoint without enumerating storage elements.

    Different roots are trivially disjoint. On one root, non-overlapping bounding intervals
    are accepted immediately. Overlapping intervals are accepted only when both exact reachable
    offset sets are representable as finite arithmetic progressions whose intersection is empty.
    Other multidimensional strided layouts deliberately fail closed.
    """

    if lhs.root != rhs.root:
        return True
    if any(dim == 0 for dim in lhs.shape) or any(dim == 0 for dim in rhs.shape):
        return True

    lhs_min, lhs_max = _reachable_span(lhs.layout, lhs.shape)
    rhs_min, rhs_max = _reachable_span(rhs.layout, rhs.shape)
    if lhs_max < rhs_min or rhs_max < lhs_min:
        return True

    lhs_progression = _finite_progression(lhs.layout, lhs.shape)
    rhs_progression = _finite_progression(rhs.layout, rhs.shape)
    if lhs_progression is None or rhs_progression is None:
        return False
    return not _progressions_intersect(lhs_progression, rhs_progression)


def _regions_may_overlap(lhs: StorageRegion, rhs: StorageRegion) -> bool:
    return not regions_provably_disjoint(lhs, rhs)


def _value_region(program: LoopProgram, buffer: int) -> StorageRegion:
    types = program.value_types
    layouts = program.value_layouts
    return StorageRegion(
        root=program.storage_root(buffer),
        shape=types[buffer].shape,
        layout=layouts[buffer],
    )


def _root_region(program: LoopProgram, buffer: int) -> StorageRegion:
    root = program.storage_root(buffer)
    types = program.value_types
    layouts = program.value_layouts
    return StorageRegion(root=root, shape=types[root].shape, layout=layouts[root])


@dataclass(frozen=True)
class _FiniteProgression:
    first: int
    step: int
    count: int

    def __post_init__(self) -> None:
        if self.step <= 0 or self.count <= 0:
            raise ValueError("finite storage progression requires positive step/count")

    @property
    def last(self) -> int:
        return self.first + self.step * (self.count - 1)


def _reachable_span(layout: StorageLayout, shape: tuple[int, ...]) -> tuple[int, int]:
    minimum = layout.offset
    maximum = layout.offset
    for dim, stride in zip(shape, layout.strides, strict=True):
        span = (dim - 1) * stride
        minimum += min(0, span)
        maximum += max(0, span)
    return minimum, maximum


def _finite_progression(
    layout: StorageLayout,
    shape: tuple[int, ...],
) -> _FiniteProgression | None:
    count = prod(shape)
    if count <= 0:
        return None
    minimum, maximum = _reachable_span(layout, shape)
    if count == 1:
        return _FiniteProgression(first=minimum, step=1, count=1)

    varying_axes = sorted(
        (abs(stride), dim)
        for dim, stride in zip(shape, layout.strides, strict=True)
        if dim > 1
    )
    if not varying_axes:
        return _FiniteProgression(first=minimum, step=1, count=1)

    base_step = varying_axes[0][0]
    covered = 1
    for stride, dim in varying_axes:
        if stride != base_step * covered:
            return None
        covered *= dim

    if covered != count or maximum - minimum != base_step * (count - 1):
        return None
    return _FiniteProgression(first=minimum, step=base_step, count=count)


def _progressions_intersect(lhs: _FiniteProgression, rhs: _FiniteProgression) -> bool:
    lower = max(lhs.first, rhs.first)
    upper = min(lhs.last, rhs.last)
    if lower > upper:
        return False
    if lhs.count == 1:
        return _progression_contains(rhs, lhs.first)
    if rhs.count == 1:
        return _progression_contains(lhs, rhs.first)

    divisor = gcd(lhs.step, rhs.step)
    delta = rhs.first - lhs.first
    if delta % divisor:
        return False

    lhs_reduced = lhs.step // divisor
    rhs_reduced = rhs.step // divisor
    multiplier = (
        0
        if rhs_reduced == 1
        else ((delta // divisor) * pow(lhs_reduced, -1, rhs_reduced)) % rhs_reduced
    )
    solution = lhs.first + lhs.step * multiplier
    period = lhs.step * rhs_reduced
    solution += _ceil_div(lower - solution, period) * period
    return solution <= upper


def _progression_contains(progression: _FiniteProgression, value: int) -> bool:
    if value < progression.first or value > progression.last:
        return False
    if progression.count == 1:
        return value == progression.first
    return (value - progression.first) % progression.step == 0


def _ceil_div(numerator: int, denominator: int) -> int:
    return -((-numerator) // denominator)
