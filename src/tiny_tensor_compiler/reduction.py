from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .ir import DType


class ReductionOperator(str, Enum):
    """Supported deterministic reduction combiners."""

    SUM = "sum"
    PRODUCT = "prod"

    @classmethod
    def from_opcode(cls, opcode: str) -> ReductionOperator:
        try:
            return cls(opcode)
        except ValueError as exc:
            raise ValueError(f"unsupported reduction opcode: {opcode!r}") from exc

    @property
    def c_operator(self) -> str:
        return "+" if self is ReductionOperator.SUM else "*"

    @property
    def identity_number(self) -> int:
        return 0 if self is ReductionOperator.SUM else 1

    def identity(self, dtype: DType | np.dtype[Any]) -> np.generic:
        np_dtype = dtype.to_numpy() if isinstance(dtype, DType) else np.dtype(dtype)
        return np_dtype.type(self.identity_number)

    def combine(
        self,
        dtype: DType | np.dtype[Any],
        lhs: Any,
        rhs: Any,
    ) -> np.generic:
        np_dtype = dtype.to_numpy() if isinstance(dtype, DType) else np.dtype(dtype)
        operation = np.add if self is ReductionOperator.SUM else np.multiply
        return np_dtype.type(operation(lhs, rhs))


REDUCTION_OPCODES = frozenset(operator.value for operator in ReductionOperator)
ReductionAxis = int | tuple[int, ...] | None


@dataclass(frozen=True)
class ReductionPlan:
    """One deterministic reduction operator over all elements or canonical logical axes."""

    operator: ReductionOperator
    axis: ReductionAxis = None

    def __post_init__(self) -> None:
        if not isinstance(self.operator, ReductionOperator):
            raise TypeError("reduction plan operator must be a ReductionOperator")
        if self.axis is None:
            return
        if isinstance(self.axis, int) and not isinstance(self.axis, bool):
            if self.axis < 0:
                raise ValueError("reduction plan axis must be non-negative")
            return
        if not isinstance(self.axis, tuple) or not self.axis:
            raise ValueError(
                "reduction plan axis must be a non-negative integer, non-empty canonical tuple, or None"
            )
        previous = -1
        for axis in self.axis:
            if not isinstance(axis, int) or isinstance(axis, bool) or axis < 0:
                raise ValueError("reduction plan axes must be non-negative integers")
            if axis <= previous:
                raise ValueError("reduction plan axis tuple must be strictly increasing")
            previous = axis

    @classmethod
    def from_opcode(cls, opcode: str, axis: ReductionAxis = None) -> ReductionPlan:
        return cls(ReductionOperator.from_opcode(opcode), axis)

    @property
    def opcode(self) -> str:
        return self.operator.value

    @property
    def axes(self) -> tuple[int, ...] | None:
        """Canonical reduced axes, or None for the historical full-tensor domain."""
        if self.axis is None:
            return None
        if isinstance(self.axis, int):
            return (self.axis,)
        return self.axis

    def reduction_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        axes = self.axes
        if axes is None:
            return input_shape
        return tuple(input_shape[axis] for axis in axes)

    def input_index(
        self,
        rank: int,
        output_index: tuple[int, ...],
        reduction_index: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Compose one logical source coordinate for a non-full reduction domain."""
        axes = self.axes
        if axes is None:
            raise ValueError("full-tensor reduction does not use split output/reduction indices")
        if len(reduction_index) != len(axes):
            raise ValueError("reduction index rank does not match reduction domain")
        if len(output_index) != rank - len(axes):
            raise ValueError("output index rank does not match unreduced domain")

        reduced_positions = {axis: position for position, axis in enumerate(axes)}
        source: list[int] = []
        output_position = 0
        for axis in range(rank):
            reduction_position = reduced_positions.get(axis)
            if reduction_position is None:
                source.append(output_index[output_position])
                output_position += 1
            else:
                source.append(reduction_index[reduction_position])
        return tuple(source)
