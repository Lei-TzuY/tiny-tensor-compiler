from __future__ import annotations

from .ir import Value
from .layout import StorageLayout

_ALIAS_OPCODES = frozenset({"view", "slice", "reverse", "transpose"})


def provably_disjoint_storage_spans(lhs: Value, rhs: Value) -> bool:
    """Prove disjoint same-root storage only when concrete bounding spans do not overlap.

    This is intentionally sufficient-but-not-necessary. Interleaved strided regions whose
    bounding intervals overlap remain unproven and therefore fail closed.
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
    return lhs_max < rhs_min or rhs_max < lhs_min


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
