from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .ir import Function, Module, SymbolicDim, TensorType, Value
from .verifier import verify


class SymbolicShapeError(ValueError):
    """Raised when the bounded runtime-symbolic shape contract is violated."""


def has_symbolic_shapes(module: Module) -> bool:
    return bool(symbolic_dims(module))


def symbolic_dims(module: Module) -> frozenset[SymbolicDim]:
    return frozenset(
        dim
        for op in module.function.ops
        for result in op.results
        for dim in result.type.symbolic_dims
    )


def validate_dynamic_batch_module(module: Module) -> SymbolicDim:
    """Verify the current dynamic contract: one shared symbolic leading dimension."""
    verify(module)
    symbols = symbolic_dims(module)
    if len(symbols) != 1:
        raise SymbolicShapeError(
            "dynamic batch compilation requires exactly one symbolic dimension"
        )
    symbol = next(iter(symbols))

    for op in module.function.ops:
        for result in op.results:
            for axis, dim in enumerate(result.type.shape):
                if isinstance(dim, SymbolicDim) and axis != 0:
                    raise SymbolicShapeError(
                        f"symbolic dimension {dim} must be the leading axis; "
                        f"got {result.type}"
                    )
                if isinstance(dim, SymbolicDim) and dim != symbol:
                    raise SymbolicShapeError(
                        "dynamic batch compilation supports one shared symbolic dimension"
                    )

    input_types = _input_types(module)
    if not any(symbol in type_.symbolic_dims for type_ in input_types):
        raise SymbolicShapeError(
            f"symbolic dimension {symbol} must be bound by at least one runtime input"
        )
    return symbol


def bind_dynamic_batch(
    module: Module,
    inputs: Sequence[Any] = (),
) -> tuple[SymbolicDim, int]:
    """Resolve and validate the one shared leading runtime batch dimension."""
    symbol = validate_dynamic_batch_module(module)
    expected_types = _input_types(module)
    provided = tuple(inputs)
    if len(provided) != len(expected_types):
        raise ValueError(
            f"expected {len(expected_types)} runtime inputs, got {len(provided)}"
        )

    batch_size: int | None = None
    for index, (value, expected_type) in enumerate(
        zip(provided, expected_types, strict=True)
    ):
        array = np.asarray(value)
        actual_shape = tuple(array.shape)
        if len(actual_shape) != len(expected_type.shape):
            raise ValueError(
                f"input {index} rank {len(actual_shape)} does not match expected "
                f"rank {len(expected_type.shape)} for {expected_type}"
            )
        expected_dtype = expected_type.dtype.to_numpy()
        if array.dtype != expected_dtype:
            raise ValueError(
                f"input {index} dtype {array.dtype} does not match expected {expected_dtype}"
            )

        for axis, (expected_dim, actual_dim) in enumerate(
            zip(expected_type.shape, actual_shape, strict=True)
        ):
            if isinstance(expected_dim, SymbolicDim):
                if expected_dim != symbol:
                    raise SymbolicShapeError(
                        f"unexpected symbolic dimension {expected_dim}; expected {symbol}"
                    )
                if batch_size is None:
                    batch_size = actual_dim
                elif batch_size != actual_dim:
                    raise ValueError(
                        f"input {index} binds symbolic dimension {symbol} to {actual_dim}, "
                        f"but the existing binding is {batch_size}"
                    )
                continue
            if actual_dim != expected_dim:
                raise ValueError(
                    f"input {index} shape {actual_shape} does not match symbolic "
                    f"contract {expected_type.shape}: axis {axis} requires {expected_dim}"
                )

    if batch_size is None:
        raise SymbolicShapeError(
            f"symbolic dimension {symbol} was not bound by the runtime inputs"
        )
    return symbol, batch_size


def clone_module(module: Module) -> Module:
    """Deep-clone verified tensor IR so reusable executables own immutable-by-convention templates."""
    verify(module)
    cloned = _clone_module(module, lambda type_: type_)
    verify(cloned)
    return cloned


def specialize_for_inputs(
    module: Module,
    inputs: Sequence[Any] = (),
) -> tuple[Module, int]:
    """Bind the runtime batch size, clone the module, and reverify the concrete IR."""
    symbol, batch_size = bind_dynamic_batch(module, inputs)
    return specialize_module(module, {symbol: batch_size}), batch_size


def specialize_module(
    module: Module,
    bindings: Mapping[SymbolicDim | str, int],
) -> Module:
    """Clone a symbolic module with explicit dimensions replaced by concrete integers."""
    verify(module)
    symbols = symbolic_dims(module)
    normalized = _normalize_bindings(symbols, bindings)
    specialized = _clone_module(
        module,
        lambda type_: _specialize_type(type_, normalized),
    )
    verify(specialized)
    if has_symbolic_shapes(specialized):
        raise SymbolicShapeError("specialization left unresolved symbolic dimensions")
    return specialized


def _clone_module(
    module: Module,
    transform_type: Callable[[TensorType], TensorType],
) -> Module:
    function = Function(module.function.name)
    values: dict[Value, Value] = {}
    for op in module.function.ops:
        operands = tuple(values[operand] for operand in op.operands)
        result_types = tuple(transform_type(result.type) for result in op.results)
        attrs = dict(op.attrs)
        if op.opcode == "const":
            attrs["value"] = np.array(op.attrs["value"], copy=True)
        cloned = function.add_op(
            op.opcode,
            operands=operands,
            result_types=result_types,
            attrs=attrs,
        )
        for original, replacement in zip(op.results, cloned.results, strict=True):
            values[original] = replacement
    return Module(function)


def _input_types(module: Module) -> tuple[TensorType, ...]:
    return tuple(
        op.results[0].type for op in module.function.ops if op.opcode == "input"
    )


def _normalize_bindings(
    symbols: frozenset[SymbolicDim],
    bindings: Mapping[SymbolicDim | str, int],
) -> dict[SymbolicDim, int]:
    by_name = {symbol.name: symbol for symbol in symbols}
    normalized: dict[SymbolicDim, int] = {}

    for key, value in bindings.items():
        if isinstance(key, SymbolicDim):
            symbol = by_name.get(key.name)
        elif isinstance(key, str):
            symbol = by_name.get(key)
        else:
            raise TypeError("symbolic binding keys must be SymbolicDim or str")
        if symbol is None:
            raise SymbolicShapeError(f"binding references unknown symbolic dimension {key!r}")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise SymbolicShapeError(
                f"symbolic dimension {symbol} requires a non-negative integer binding"
            )
        previous = normalized.get(symbol)
        if previous is not None and previous != value:
            raise SymbolicShapeError(
                f"symbolic dimension {symbol} has conflicting bindings {previous} and {value}"
            )
        normalized[symbol] = value

    missing = symbols - normalized.keys()
    if missing:
        names = ", ".join(sorted(symbol.name for symbol in missing))
        raise SymbolicShapeError(f"missing bindings for symbolic dimensions: {names}")
    return normalized


def _specialize_type(
    type_: TensorType,
    bindings: Mapping[SymbolicDim, int],
) -> TensorType:
    shape = tuple(
        bindings[dim] if isinstance(dim, SymbolicDim) else dim
        for dim in type_.shape
    )
    return TensorType(shape, type_.dtype)
