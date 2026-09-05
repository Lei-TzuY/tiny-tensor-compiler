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


def validate_dynamic_module(module: Module) -> tuple[SymbolicDim, ...]:
    """Verify that every runtime symbol can be bound by the module inputs."""
    verify(module)
    symbols = tuple(sorted(symbolic_dims(module)))
    if not symbols:
        raise SymbolicShapeError(
            "dynamic compilation requires at least one symbolic dimension"
        )

    input_types = _input_types(module)
    for symbol in symbols:
        if not any(symbol in type_.symbolic_dims for type_ in input_types):
            raise SymbolicShapeError(
                f"symbolic dimension {symbol} must be bound by at least one runtime input"
            )
    return symbols


def validate_dynamic_batch_module(module: Module) -> SymbolicDim:
    """Compatibility helper for callers that require exactly one runtime symbol."""
    symbols = validate_dynamic_module(module)
    if len(symbols) != 1:
        raise SymbolicShapeError(
            "dynamic batch compilation requires exactly one symbolic dimension"
        )
    return symbols[0]


def bind_dynamic_shapes(
    module: Module,
    inputs: Sequence[Any] = (),
) -> dict[SymbolicDim, int]:
    """Resolve every named runtime dimension from exact runtime input shapes."""
    symbols = validate_dynamic_module(module)
    expected_types = _input_types(module)
    provided = tuple(inputs)
    if len(provided) != len(expected_types):
        raise ValueError(
            f"expected {len(expected_types)} runtime inputs, got {len(provided)}"
        )

    bindings: dict[SymbolicDim, int] = {}
    symbol_set = frozenset(symbols)
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
                if expected_dim not in symbol_set:
                    raise SymbolicShapeError(
                        f"unexpected symbolic dimension {expected_dim}"
                    )
                previous = bindings.get(expected_dim)
                if previous is None:
                    bindings[expected_dim] = actual_dim
                elif previous != actual_dim:
                    raise ValueError(
                        f"input {index} binds symbolic dimension {expected_dim} to "
                        f"{actual_dim}, but the existing binding is {previous}"
                    )
                continue
            if actual_dim != expected_dim:
                raise ValueError(
                    f"input {index} shape {actual_shape} does not match symbolic "
                    f"contract {expected_type.shape}: axis {axis} requires {expected_dim}"
                )

    missing = symbol_set - bindings.keys()
    if missing:
        names = ", ".join(sorted(symbol.name for symbol in missing))
        raise SymbolicShapeError(
            f"symbolic dimensions were not bound by runtime inputs: {names}"
        )
    return {symbol: bindings[symbol] for symbol in symbols}


def bind_dynamic_batch(
    module: Module,
    inputs: Sequence[Any] = (),
) -> tuple[SymbolicDim, int]:
    """Compatibility helper for one runtime symbolic dimension."""
    symbol = validate_dynamic_batch_module(module)
    bindings = bind_dynamic_shapes(module, inputs)
    return symbol, bindings[symbol]


def clone_module(module: Module) -> Module:
    """Deep-clone verified tensor IR so reusable executables own immutable-by-convention templates."""
    verify(module)
    cloned = _clone_module(module, lambda type_: type_)
    verify(cloned)
    return cloned


def specialize_for_inputs(
    module: Module,
    inputs: Sequence[Any] = (),
) -> tuple[Module, dict[SymbolicDim, int]]:
    """Bind runtime symbols, clone the module, and reverify the concrete IR."""
    bindings = bind_dynamic_shapes(module, inputs)
    return specialize_module(module, bindings), bindings


def normalize_symbolic_bindings(
    module: Module,
    bindings: Mapping[SymbolicDim | str, int],
) -> dict[SymbolicDim, int]:
    """Normalize explicit bindings against the symbols declared by a module."""
    return _normalize_bindings(symbolic_dims(module), bindings)


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
