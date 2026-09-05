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


@dataclass(frozen=True)
class ReductionPlan:
    """One deterministic reduction operator over all elements or one logical axis."""

    operator: ReductionOperator
    axis: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operator, ReductionOperator):
            raise TypeError("reduction plan operator must be a ReductionOperator")
        if self.axis is not None and (
            not isinstance(self.axis, int) or isinstance(self.axis, bool) or self.axis < 0
        ):
            raise ValueError("reduction plan axis must be a non-negative integer or None")

    @classmethod
    def from_opcode(cls, opcode: str, axis: int | None = None) -> ReductionPlan:
        return cls(ReductionOperator.from_opcode(opcode), axis)

    @property
    def opcode(self) -> str:
        return self.operator.value
