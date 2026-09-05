from __future__ import annotations

from dataclasses import dataclass
from math import gcd, prod

from .ir import Value
from .layout import StorageLayout

_ALIAS_OPCODES = frozenset({"view", "slice", "reverse", "transpose"})


@dataclass(frozen=True)
class _FiniteProgression:
    first: int
    step: int
    count: int

    def __post_init__(self) -> None:
        if self.step <= 0:
            raise ValueError("finite progression step must be positive")
        if self.count <= 0:
            raise ValueError("finite progression count must be positive")

    @property
    def last(self) -> int:
        return self.first + self.step * (self.count - 1)


def provably_disjoint_storage_spans(lhs: Value, rhs: Value) -> bool:
    """Prove concrete same-root regions disjoint without enumerating storage elements.

    Non-overlapping bounding intervals are accepted immediately. If the intervals overlap,
    the proof recognizes layouts whose complete reachable offset set is one finite arithmetic
    progression and uses exact modular arithmetic to decide whether the two progressions
    intersect. Other multidimensional strided sets remain deliberately fail-closed.
    """
    if _storage_root(lhs) is not _storage_root(rhs):
        return False

    lhs_layout = _concrete_storage_layout(lhs)
    rhs_layout = _concrete_storage_layout(rhs)
    lhs_shape = _concrete_shape(lhs)
    rhs_shape = _concrete_shape(rhs)
    if lhs_layout is None or rhs_layout is None or lhs_shape is None or rhs_shape is None:
        return False
    if any(dim == 0 for dim in lhs_shape) or any(dim == 0 for dim in rhs_shape):
        return True

    lhs_min, lhs_max = _reachable_span(lhs_layout, lhs_shape)
    rhs_min, rhs_max = _reachable_span(rhs_layout, rhs_shape)
    if lhs_max < rhs_min or rhs_max < lhs_min:
        return True

    lhs_progression = _finite_progression(lhs_layout, lhs_shape)
    rhs_progression = _finite_progression(rhs_layout, rhs_shape)
    if lhs_progression is None or rhs_progression is None:
        return False
    return not _progressions_intersect(lhs_progression, rhs_progression)


def _finite_progression(
    layout: StorageLayout,
    shape: tuple[int, ...],
) -> _FiniteProgression | None:
    """Return the exact reachable set when one packed finite progression represents it."""
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

    if covered != count:
        return None
    if maximum - minimum != base_step * (count - 1):
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
    if rhs_reduced == 1:
        multiplier = 0
    else:
        multiplier = (
            (delta // divisor) * pow(lhs_reduced, -1, rhs_reduced)
        ) % rhs_reduced

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


def _concrete_storage_layout(value: Value) -> StorageLayout | None:
    shape = _concrete_shape(value)
    if shape is None:
        return None

    producer = value.producer
    if producer is None or producer.opcode not in _ALIAS_OPCODES | {"copy_into"}:
        return StorageLayout.contiguous(shape)
    if producer.opcode == "copy_into":
        return _concrete_storage_layout(producer.operands[0])

    source = producer.operands[0]
    source_shape = _concrete_shape(source)
    source_layout = _concrete_storage_layout(source)
    if source_shape is None or source_layout is None:
        return None

    if producer.opcode == "view":
        return source_layout.reshaped(source_shape, shape)
    if producer.opcode == "slice":
        layout, inferred_shape = source_layout.sliced(
            source_shape,
            axis=producer.attrs["axis"],
            start=producer.attrs["start"],
            stop=producer.attrs["stop"],
            step=producer.attrs["step"],
        )
        return layout if inferred_shape == shape else None
    if producer.opcode == "reverse":
        return source_layout.reversed(source_shape, producer.attrs["axis"])
    if producer.opcode == "transpose":
        layout, inferred_shape = source_layout.permuted(source_shape, producer.attrs["axes"])
        return layout if inferred_shape == shape else None
    return None


def _concrete_shape(value: Value) -> tuple[int, ...] | None:
    shape = value.type.shape
    if any(not isinstance(dim, int) or isinstance(dim, bool) for dim in shape):
        return None
    return shape


def _reachable_span(layout: StorageLayout, shape: tuple[int, ...]) -> tuple[int, int]:
    minimum = layout.offset
    maximum = layout.offset
    for dim, stride in zip(shape, layout.strides, strict=True):
        span = (dim - 1) * stride
        minimum += min(0, span)
        maximum += max(0, span)
    return minimum, maximum


def _storage_root(value: Value) -> Value:
    current = value
    seen: set[Value] = set()
    while True:
        if current in seen:
            raise ValueError("tensor alias cycle detected")
        seen.add(current)
        producer = current.producer
        if producer is None:
            return current
        if producer.opcode in _ALIAS_OPCODES:
            current = producer.operands[0]
            continue
        if producer.opcode == "copy_into":
            current = producer.operands[0]
            continue
        return current
