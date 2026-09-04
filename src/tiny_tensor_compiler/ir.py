from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias

import numpy as np


class DType(str, Enum):
    INT32 = "i32"
    INT64 = "i64"
    FLOAT32 = "f32"
    FLOAT64 = "f64"

    @classmethod
    def from_numpy(cls, dtype: np.dtype[Any] | str) -> DType:
        normalized = np.dtype(dtype)
        mapping = {
            np.dtype("int32"): cls.INT32,
            np.dtype("int64"): cls.INT64,
            np.dtype("float32"): cls.FLOAT32,
            np.dtype("float64"): cls.FLOAT64,
        }
        try:
            return mapping[normalized]
        except KeyError as exc:
            raise TypeError(f"unsupported tensor dtype: {normalized}") from exc

    def to_numpy(self) -> np.dtype[Any]:
        return {
            DType.INT32: np.dtype("int32"),
            DType.INT64: np.dtype("int64"),
            DType.FLOAT32: np.dtype("float32"),
            DType.FLOAT64: np.dtype("float64"),
        }[self]


@dataclass(frozen=True, order=True)
class SymbolicDim:
    """Named tensor dimension that must be specialized before physical lowering."""

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError(f"invalid symbolic dimension name: {self.name!r}")

    def __str__(self) -> str:
        return self.name


ShapeDim: TypeAlias = int | SymbolicDim


@dataclass(frozen=True)
class TensorType:
    shape: tuple[ShapeDim, ...]
    dtype: DType

    def __post_init__(self) -> None:
        for dim in self.shape:
            if isinstance(dim, SymbolicDim):
                continue
            if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
                raise ValueError(f"invalid tensor shape: {self.shape}")

    @property
    def is_static(self) -> bool:
        return all(isinstance(dim, int) for dim in self.shape)

    @property
    def symbolic_dims(self) -> frozenset[SymbolicDim]:
        return frozenset(dim for dim in self.shape if isinstance(dim, SymbolicDim))

    def __str__(self) -> str:
        dims = "x".join(str(dim) for dim in self.shape)
        prefix = f"{dims}x" if dims else ""
        return f"tensor<{prefix}{self.dtype.value}>"


@dataclass(frozen=True)
class Use:
    user: Operation
    operand_index: int


class Value:
    def __init__(
        self,
        value_id: int,
        type_: TensorType,
        producer: Operation | None,
        result_index: int,
    ) -> None:
        self.id = value_id
        self.type = type_
        self.producer = producer
        self.result_index = result_index
        self.uses: list[Use] = []

    def replace_all_uses_with(self, replacement: Value) -> None:
        if replacement is self:
            return
        old_uses = list(self.uses)
        for use in old_uses:
            use.user.replace_operand(use.operand_index, replacement)
        if self.uses:
            raise RuntimeError("internal error: value still has uses after replacement")


class Operation:
    def __init__(
        self,
        opcode: str,
        operands: Iterable[Value],
        result_types: Iterable[TensorType],
        attrs: dict[str, Any] | None = None,
    ) -> None:
        self.opcode = opcode
        self.operands = list(operands)
        self.results: list[Value] = []
        self.attrs = dict(attrs or {})
        self.parent: Function | None = None
        self._result_types = list(result_types)

    def attach(self, function: Function) -> None:
        if self.parent is not None:
            raise ValueError("operation is already attached to a function")
        self.parent = function
        for index, operand in enumerate(self.operands):
            operand.uses.append(Use(self, index))
        for result_index, result_type in enumerate(self._result_types):
            self.results.append(
                Value(function.allocate_value_id(), result_type, self, result_index)
            )

    def replace_operand(self, index: int, replacement: Value) -> None:
        current = self.operands[index]
        try:
            current.uses.remove(Use(self, index))
        except ValueError as exc:
            raise RuntimeError("internal error: missing use-def edge") from exc
        self.operands[index] = replacement
        replacement.uses.append(Use(self, index))


class Function:
    def __init__(self, name: str = "main") -> None:
        self.name = name
        self.ops: list[Operation] = []
        self._next_value_id = 0

    def allocate_value_id(self) -> int:
        value_id = self._next_value_id
        self._next_value_id += 1
        return value_id

    def add_op(
        self,
        opcode: str,
        operands: Iterable[Value] = (),
        result_types: Iterable[TensorType] = (),
        attrs: dict[str, Any] | None = None,
    ) -> Operation:
        return self.insert_op(len(self.ops), opcode, operands, result_types, attrs)

    def insert_op(
        self,
        index: int,
        opcode: str,
        operands: Iterable[Value] = (),
        result_types: Iterable[TensorType] = (),
        attrs: dict[str, Any] | None = None,
    ) -> Operation:
        op = Operation(opcode, operands, result_types, attrs)
        op.attach(self)
        self.ops.insert(index, op)
        return op

    def erase_op(self, op: Operation) -> None:
        if op.parent is not self:
            raise ValueError("operation does not belong to this function")
        if any(result.uses for result in op.results):
            raise ValueError("cannot erase operation with live result uses")
        for index, operand in enumerate(op.operands):
            try:
                operand.uses.remove(Use(op, index))
            except ValueError as exc:
                raise RuntimeError("internal error: missing use-def edge") from exc
        self.ops.remove(op)
        op.parent = None


class Module:
    def __init__(self, function: Function) -> None:
        self.function = function

    def dump(self) -> str:
        value_names: dict[Value, str] = {}
        next_name = 0
        lines = [f"func @{self.function.name}() {{"]
        for op in self.function.ops:
            result_names: list[str] = []
            for result in op.results:
                name = f"%{next_name}"
                next_name += 1
                value_names[result] = name
                result_names.append(name)

            if op.opcode == "input":
                lines.append(
                    f"  {result_names[0]} = input {op.attrs['index']} : {op.results[0].type}"
                )
            elif op.opcode == "const":
                literal = _format_literal(op.attrs["value"])
                lines.append(
                    f"  {result_names[0]} = const {literal} : {op.results[0].type}"
                )
            elif op.opcode in {"add", "mul"}:
                operands = ", ".join(value_names[value] for value in op.operands)
                lines.append(
                    f"  {result_names[0]} = {op.opcode} {operands} : {op.results[0].type}"
                )
            elif op.opcode == "relu":
                lines.append(
                    f"  {result_names[0]} = relu {value_names[op.operands[0]]} : "
                    f"{op.results[0].type}"
                )
            elif op.opcode == "return":
                operands = ", ".join(value_names[value] for value in op.operands)
                lines.append(f"  return {operands}")
            else:
                operands = ", ".join(
                    value_names.get(value, f"%?{value.id}") for value in op.operands
                )
                lhs = f"{', '.join(result_names)} = " if result_names else ""
                lines.append(f"  {lhs}{op.opcode} {operands}".rstrip())
        lines.append("}")
        return "\n".join(lines)


def _format_literal(value: Any) -> str:
    array = np.asarray(value)
    if array.ndim == 0:
        return repr(array.item())
    return repr(array.tolist())
