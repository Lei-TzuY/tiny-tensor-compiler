from __future__ import annotations

from collections.abc import Iterable, Mapping
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

    def __mul__(self, scale: int) -> SymbolicDim | AffineDim:
        _validate_positive_scale(scale)
        if scale == 1:
            return self
        return AffineDim(self, scale=scale)

    def __rmul__(self, scale: int) -> SymbolicDim | AffineDim:
        return self * scale

    def __add__(
        self,
        other: int | SymbolicDim | AffineDim | LinearDim,
    ) -> SymbolicDim | AffineDim | LinearDim:
        return _add_shape_expression(self, other)

    def __radd__(
        self,
        other: int | SymbolicDim | AffineDim | LinearDim,
    ) -> SymbolicDim | AffineDim | LinearDim:
        return _add_shape_expression(other, self)


@dataclass(frozen=True, order=True)
class AffineDim:
    """One-variable positive affine tensor dimension ``scale * symbol + offset``."""

    symbol: SymbolicDim
    scale: int = 1
    offset: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, SymbolicDim):
            raise TypeError("affine dimension requires a SymbolicDim")
        _validate_positive_scale(self.scale)
        _validate_non_negative_offset(self.offset)

    def __str__(self) -> str:
        base = str(self.symbol) if self.scale == 1 else f"{self.scale}*{self.symbol}"
        return f"{base}+{self.offset}" if self.offset else base

    def __mul__(self, factor: int) -> AffineDim:
        _validate_positive_scale(factor)
        if factor == 1:
            return self
        return AffineDim(
            self.symbol,
            scale=self.scale * factor,
            offset=self.offset * factor,
        )

    def __rmul__(self, factor: int) -> AffineDim:
        return self * factor

    def __add__(
        self,
        other: int | SymbolicDim | AffineDim | LinearDim,
    ) -> SymbolicDim | AffineDim | LinearDim:
        return _add_shape_expression(self, other)

    def __radd__(
        self,
        other: int | SymbolicDim | AffineDim | LinearDim,
    ) -> SymbolicDim | AffineDim | LinearDim:
        return _add_shape_expression(other, self)

    def evaluate(self, binding: int) -> int:
        if not isinstance(binding, int) or isinstance(binding, bool) or binding < 0:
            raise ValueError("affine dimension binding must be a non-negative integer")
        return self.scale * binding + self.offset

    def solve(self, extent: int) -> int:
        if not isinstance(extent, int) or isinstance(extent, bool) or extent < 0:
            raise ValueError("affine runtime extent must be a non-negative integer")
        if extent < self.offset:
            raise ValueError(
                f"runtime extent {extent} is smaller than affine offset {self.offset} "
                f"for {self}"
            )
        residual = extent - self.offset
        if residual % self.scale:
            raise ValueError(
                f"runtime extent {extent} minus offset {self.offset} is not divisible "
                f"by scale {self.scale} for {self}"
            )
        return residual // self.scale


@dataclass(frozen=True)
class LinearDim:
    """Canonical positive linear form over at least two named dimensions."""

    terms: tuple[tuple[SymbolicDim, int], ...]
    offset: int = 0

    def __post_init__(self) -> None:
        _validate_non_negative_offset(self.offset)
        combined: dict[SymbolicDim, int] = {}
        for term in self.terms:
            if not isinstance(term, tuple) or len(term) != 2:
                raise TypeError("linear dimension terms must be (SymbolicDim, coefficient) pairs")
            symbol, coefficient = term
            if not isinstance(symbol, SymbolicDim):
                raise TypeError("linear dimension terms require SymbolicDim keys")
            _validate_positive_coefficient(coefficient)
            combined[symbol] = combined.get(symbol, 0) + coefficient

        canonical = tuple(sorted(combined.items(), key=lambda item: item[0].name))
        if len(canonical) < 2:
            raise ValueError("linear dimension requires at least two distinct symbols")
        object.__setattr__(self, "terms", canonical)

    @property
    def symbolic_dims(self) -> frozenset[SymbolicDim]:
        return frozenset(symbol for symbol, _ in self.terms)

    def __str__(self) -> str:
        pieces = [
            str(symbol) if coefficient == 1 else f"{coefficient}*{symbol}"
            for symbol, coefficient in self.terms
        ]
        if self.offset:
            pieces.append(str(self.offset))
        return "+".join(pieces)

    def __mul__(self, factor: int) -> LinearDim:
        _validate_positive_scale(factor)
        if factor == 1:
            return self
        return LinearDim(
            tuple((symbol, coefficient * factor) for symbol, coefficient in self.terms),
            offset=self.offset * factor,
        )

    def __rmul__(self, factor: int) -> LinearDim:
        return self * factor

    def __add__(
        self,
        other: int | SymbolicDim | AffineDim | LinearDim,
    ) -> SymbolicDim | AffineDim | LinearDim:
        return _add_shape_expression(self, other)

    def __radd__(
        self,
        other: int | SymbolicDim | AffineDim | LinearDim,
    ) -> SymbolicDim | AffineDim | LinearDim:
        return _add_shape_expression(other, self)

    def evaluate(self, bindings: Mapping[SymbolicDim, int]) -> int:
        total = self.offset
        for symbol, coefficient in self.terms:
            try:
                binding = bindings[symbol]
            except KeyError as exc:
                raise ValueError(f"missing binding for linear dimension symbol {symbol}") from exc
            if not isinstance(binding, int) or isinstance(binding, bool) or binding < 0:
                raise ValueError(
                    f"linear dimension symbol {symbol} requires a non-negative integer binding"
                )
            total += coefficient * binding
        return total


ShapeDim: TypeAlias = int | SymbolicDim | AffineDim | LinearDim


@dataclass(frozen=True)
class TensorType:
    shape: tuple[ShapeDim, ...]
    dtype: DType

    def __post_init__(self) -> None:
        for dim in self.shape:
            if isinstance(dim, (SymbolicDim, AffineDim, LinearDim)):
                continue
            if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
                raise ValueError(f"invalid tensor shape: {self.shape}")

    @property
    def is_static(self) -> bool:
        return all(isinstance(dim, int) for dim in self.shape)

    @property
    def symbolic_dims(self) -> frozenset[SymbolicDim]:
        symbols: set[SymbolicDim] = set()
        for dim in self.shape:
            if isinstance(dim, SymbolicDim):
                symbols.add(dim)
            elif isinstance(dim, AffineDim):
                symbols.add(dim.symbol)
            elif isinstance(dim, LinearDim):
                symbols.update(dim.symbolic_dims)
        return frozenset(symbols)

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


def _add_shape_expression(
    lhs: int | SymbolicDim | AffineDim | LinearDim,
    rhs: int | SymbolicDim | AffineDim | LinearDim,
) -> SymbolicDim | AffineDim | LinearDim:
    lhs_terms, lhs_offset = _shape_expression_parts(lhs)
    rhs_terms, rhs_offset = _shape_expression_parts(rhs)
    coefficients = dict(lhs_terms)
    for symbol, coefficient in rhs_terms.items():
        coefficients[symbol] = coefficients.get(symbol, 0) + coefficient
    return _build_shape_expression(coefficients, lhs_offset + rhs_offset)


def _shape_expression_parts(
    dim: int | SymbolicDim | AffineDim | LinearDim,
) -> tuple[dict[SymbolicDim, int], int]:
    if isinstance(dim, bool):
        raise ValueError("symbolic shape offsets must be non-negative integers")
    if isinstance(dim, int):
        _validate_non_negative_offset(dim)
        return {}, dim
    if isinstance(dim, SymbolicDim):
        return {dim: 1}, 0
    if isinstance(dim, AffineDim):
        return {dim.symbol: dim.scale}, dim.offset
    if isinstance(dim, LinearDim):
        return dict(dim.terms), dim.offset
    raise TypeError(f"unsupported symbolic shape expression operand: {type(dim).__name__}")


def _build_shape_expression(
    coefficients: Mapping[SymbolicDim, int],
    offset: int,
) -> SymbolicDim | AffineDim | LinearDim:
    _validate_non_negative_offset(offset)
    canonical = tuple(sorted(coefficients.items(), key=lambda item: item[0].name))
    if not canonical:
        raise TypeError("symbolic shape expression must contain at least one symbol")
    if len(canonical) == 1:
        symbol, scale = canonical[0]
        _validate_positive_scale(scale)
        if scale == 1 and offset == 0:
            return symbol
        return AffineDim(symbol, scale=scale, offset=offset)
    return LinearDim(canonical, offset=offset)


def _validate_positive_scale(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("affine dimension requires a positive integer scale")


def _validate_positive_coefficient(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("linear dimension requires a positive integer coefficient")


def _validate_non_negative_offset(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("affine dimension requires a non-negative integer offset")


def _format_literal(value: Any) -> str:
    array = np.asarray(value)
    if array.ndim == 0:
        return repr(array.item())
    return repr(array.tolist())
