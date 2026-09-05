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
