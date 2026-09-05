from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul


@dataclass(frozen=True)
class StorageLayout:
    """Logical tensor layout in element offsets relative to one storage root."""

    offset: int
    strides: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.offset, int) or isinstance(self.offset, bool) or self.offset < 0:
            raise ValueError("storage layout offset must be a non-negative integer")
        if any(
            not isinstance(stride, int) or isinstance(stride, bool) or stride == 0
            for stride in self.strides
        ):
            raise ValueError("storage layout strides must be non-zero integers")

    @classmethod
    def contiguous(cls, shape: tuple[int, ...], *, offset: int = 0) -> StorageLayout:
        return cls(offset=offset, strides=contiguous_strides(shape))

    def is_contiguous(self, shape: tuple[int, ...]) -> bool:
        canonical = contiguous_strides(shape)
        if len(self.strides) != len(shape):
            return False
        if any(dim == 0 for dim in shape):
            return True
        return all(
            dim == 1 or stride == expected
            for dim, stride, expected in zip(shape, self.strides, canonical, strict=True)
        )

    def validate_bounds(self, shape: tuple[int, ...], storage_elements: int) -> None:
        if len(shape) != len(self.strides):
            raise ValueError("storage layout rank does not match logical tensor rank")
        if storage_elements < 0:
            raise ValueError("storage element count must be non-negative")
        if any(dim < 0 for dim in shape):
            raise ValueError("storage layout requires a concrete non-negative shape")
        if any(dim == 0 for dim in shape):
            if self.offset > storage_elements:
                raise ValueError("empty storage view offset exceeds storage bounds")
            return

        minimum = self.offset
        maximum = self.offset
        for dim, stride in zip(shape, self.strides, strict=True):
            span = (dim - 1) * stride
            minimum += min(0, span)
            maximum += max(0, span)
        if minimum < 0 or maximum >= storage_elements:
            raise ValueError("storage layout exceeds backing storage bounds")

    def reshaped(self, source_shape: tuple[int, ...], target_shape: tuple[int, ...]) -> StorageLayout:
        if element_count(source_shape) != element_count(target_shape):
            raise ValueError("contiguous view reshape requires equal element counts")
        if not self.is_contiguous(source_shape):
            raise ValueError("cannot reshape a non-contiguous storage view without copying")
        offset = 0 if any(dim == 0 for dim in target_shape) else self.offset
        return StorageLayout.contiguous(target_shape, offset=offset)

    def sliced(
        self,
        source_shape: tuple[int, ...],
        *,
        axis: int,
        start: int,
        stop: int,
        step: int = 1,
    ) -> tuple[StorageLayout, tuple[int, ...]]:
        _validate_slice(source_shape, axis=axis, start=start, stop=stop, step=step)
        length = (stop - start + step - 1) // step
        shape = list(source_shape)
        shape[axis] = length
        strides = list(self.strides)
        offset = self.offset + start * strides[axis]
        strides[axis] *= step
        output_shape = tuple(shape)
        if any(dim == 0 for dim in output_shape):
            offset = 0
        return StorageLayout(offset=offset, strides=tuple(strides)), output_shape

    def reversed(self, source_shape: tuple[int, ...], axis: int) -> StorageLayout:
        _validate_axis(source_shape, axis, operation="reverse")
        extent = source_shape[axis]
        strides = list(self.strides)
        offset = self.offset
        if extent:
            offset += (extent - 1) * strides[axis]
        strides[axis] *= -1
        if any(dim == 0 for dim in source_shape):
            offset = 0
        return StorageLayout(offset=offset, strides=tuple(strides))

    def permuted(
        self,
        source_shape: tuple[int, ...],
        axes: tuple[int, ...],
    ) -> tuple[StorageLayout, tuple[int, ...]]:
        permutation = normalize_permutation(len(source_shape), axes)
        shape = tuple(source_shape[axis] for axis in permutation)
        strides = tuple(self.strides[axis] for axis in permutation)
        offset = 0 if any(dim == 0 for dim in shape) else self.offset
        return StorageLayout(offset=offset, strides=strides), shape


def normalize_permutation(rank: int, axes: tuple[int, ...]) -> tuple[int, ...]:
    if len(axes) != rank:
        raise ValueError("transpose axes must form a complete permutation of tensor rank")
    if any(not isinstance(axis, int) or isinstance(axis, bool) for axis in axes):
        raise TypeError("transpose axes must be integers")
    if set(axes) != set(range(rank)):
        raise ValueError("transpose axes must form a complete permutation of tensor rank")
    return axes


def contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    if any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape):
        raise ValueError("contiguous strides require a concrete non-negative shape")
    strides = [1] * len(shape)
    running = 1
    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = running
        running *= max(shape[axis], 1)
    return tuple(strides)


def element_count(shape: tuple[int, ...]) -> int:
    return reduce(mul, shape, 1)


def validate_slice_bounds(
    shape: tuple[int, ...],
    *,
    axis: int,
    start: int,
    stop: int,
    step: int,
) -> None:
    _validate_slice(shape, axis=axis, start=start, stop=stop, step=step)


def _validate_axis(shape: tuple[int, ...], axis: int, *, operation: str) -> None:
    if not isinstance(axis, int) or isinstance(axis, bool) or axis < 0 or axis >= len(shape):
        raise ValueError(f"{operation} axis is out of range")
    extent = shape[axis]
    if not isinstance(extent, int) or isinstance(extent, bool):
        raise TypeError(f"{operation} axis extent must be concrete")


def _validate_slice(
    shape: tuple[int, ...],
    *,
    axis: int,
    start: int,
    stop: int,
    step: int,
) -> None:
    _validate_axis(shape, axis, operation="slice")
    for name, value in (("start", start), ("stop", stop), ("step", step)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"slice {name} must be an integer")
    if step <= 0:
        raise ValueError("slice step must be a positive integer")
    extent = shape[axis]
    if start < 0 or stop < 0 or start > stop or stop > extent:
        raise ValueError(
            f"slice bounds must satisfy 0 <= start <= stop <= extent ({extent})"
        )