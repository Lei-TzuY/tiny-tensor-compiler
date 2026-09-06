from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .ir import AffineDim, DType, LinearDim, ShapeDim, SymbolicDim, TensorType
from .layout import normalize_permutation
from .reduction import ReductionAxis, ReductionOperator


class TypeInferenceError(ValueError):
    pass


def infer_binary(lhs: TensorType, rhs: TensorType) -> TensorType:
    shape = _broadcast_shapes(lhs.shape, rhs.shape)
    try:
        dtype = DType.from_numpy(np.result_type(lhs.dtype.to_numpy(), rhs.dtype.to_numpy()))
    except TypeError as exc:
        raise TypeInferenceError(str(exc)) from exc
    return TensorType(shape, dtype)


def infer_relu(input_type: TensorType) -> TensorType:
    if input_type.dtype not in {DType.INT32, DType.INT64, DType.FLOAT32, DType.FLOAT64}:
        raise TypeInferenceError(f"relu requires a numeric tensor, got {input_type.dtype.value}")
    return input_type


def normalize_reduction_axis(
    input_type: TensorType,
    axis: int,
    operator: ReductionOperator,
) -> int:
    """Normalize one Python-style reduction axis to canonical non-negative form."""
    if not isinstance(axis, int) or isinstance(axis, bool):
        raise TypeInferenceError(f"{operator.value} axis must be an integer")
    rank = len(input_type.shape)
    if axis < -rank or axis >= rank:
        raise TypeInferenceError(
            f"{operator.value} axis {axis} is out of range for rank {rank}"
        )
    return axis + rank if axis < 0 else axis


def normalize_reduction_axes(
    input_type: TensorType,
    axis: int | Iterable[int] | None,
    operator: ReductionOperator,
) -> ReductionAxis:
    """Canonicalize a full, single-axis, or non-empty multi-axis reduction domain."""
    if axis is None:
        return None
    if isinstance(axis, int) and not isinstance(axis, bool):
        return normalize_reduction_axis(input_type, axis, operator)
    if operator is ReductionOperator.ARGMAX:
        raise TypeInferenceError("argmax axis must be an integer or None")
    if isinstance(axis, (str, bytes)):
        raise TypeInferenceError(f"{operator.value} axis must be an integer or axis collection")
    try:
        raw_axes = tuple(axis)
    except TypeError as exc:
        raise TypeInferenceError(
            f"{operator.value} axis must be an integer or axis collection"
        ) from exc
    if not raw_axes:
        raise TypeInferenceError(f"{operator.value} axis collection must not be empty")
    normalized = tuple(
        normalize_reduction_axis(input_type, candidate, operator) for candidate in raw_axes
    )
    if len(set(normalized)) != len(normalized):
        raise TypeInferenceError(f"{operator.value} axis collection contains duplicates")
    canonical = tuple(sorted(normalized))
    return canonical[0] if len(canonical) == 1 else canonical


def infer_reduction(
    input_type: TensorType,
    operator: ReductionOperator,
    axis: int | Iterable[int] | None = None,
) -> TensorType:
    """Infer one deterministic logical reduction domain."""
    if input_type.dtype not in {DType.INT32, DType.INT64, DType.FLOAT32, DType.FLOAT64}:
        raise TypeInferenceError(
            f"{operator.value} requires a numeric tensor, got {input_type.dtype.value}"
        )
    normalized = normalize_reduction_axes(input_type, axis, operator)
    if normalized is None:
        shape: tuple[ShapeDim, ...] = ()
        reduced_axes = tuple(range(len(input_type.shape)))
    else:
        axes = (normalized,) if isinstance(normalized, int) else normalized
        reduced = set(axes)
        reduced_axes = axes
        shape = tuple(
            dim for position, dim in enumerate(input_type.shape) if position not in reduced
        )

    if operator is ReductionOperator.ARGMAX:
        if any(input_type.shape[position] == 0 for position in reduced_axes):
            raise TypeInferenceError("argmax reduction domain must not be empty")
        return TensorType(shape, DType.INT64)
    return TensorType(shape, input_type.dtype)


def normalize_sum_axis(input_type: TensorType, axis: int) -> int:
    """Compatibility wrapper for the historical single sum-axis API."""
    return normalize_reduction_axis(input_type, axis, ReductionOperator.SUM)


def normalize_prod_axis(input_type: TensorType, axis: int) -> int:
    """Compatibility wrapper for the historical single product-axis API."""
    return normalize_reduction_axis(input_type, axis, ReductionOperator.PRODUCT)


def normalize_sum_axes(
    input_type: TensorType, axis: int | Iterable[int] | None
) -> ReductionAxis:
    return normalize_reduction_axes(input_type, axis, ReductionOperator.SUM)


def normalize_prod_axes(
    input_type: TensorType, axis: int | Iterable[int] | None
) -> ReductionAxis:
    return normalize_reduction_axes(input_type, axis, ReductionOperator.PRODUCT)


def normalize_argmax_axis(input_type: TensorType, axis: int | None) -> int | None:
    normalized = normalize_reduction_axes(input_type, axis, ReductionOperator.ARGMAX)
    if isinstance(normalized, tuple):
        raise TypeInferenceError("argmax accepts only one reduction axis")
    return normalized


def infer_sum(
    input_type: TensorType, axis: int | Iterable[int] | None = None
) -> TensorType:
    """Compatibility wrapper for deterministic sum inference."""
    return infer_reduction(input_type, ReductionOperator.SUM, axis)


def infer_prod(
    input_type: TensorType, axis: int | Iterable[int] | None = None
) -> TensorType:
    """Infer a deterministic full/single/multi-axis same-dtype product."""
    return infer_reduction(input_type, ReductionOperator.PRODUCT, axis)


def infer_argmax(input_type: TensorType, axis: int | None = None) -> TensorType:
    """Infer deterministic first-index argmax with an int64 result."""
    return infer_reduction(input_type, ReductionOperator.ARGMAX, axis)


def infer_reshape(input_type: TensorType, shape: Iterable[ShapeDim]) -> TensorType:
    """Infer a row-major reshape only when element-count equality is provable exactly."""
    try:
        result_type = TensorType(tuple(shape), input_type.dtype)
    except (TypeError, ValueError) as exc:
        raise TypeInferenceError(str(exc)) from exc

    new_symbols = result_type.symbolic_dims - input_type.symbolic_dims
    if new_symbols:
        names = ", ".join(sorted(symbol.name for symbol in new_symbols))
        raise TypeInferenceError(
            f"reshape target introduces new symbolic dimension(s): {names}"
        )

    if _shape_element_polynomial(input_type.shape) != _shape_element_polynomial(
        result_type.shape
    ):
        raise TypeInferenceError(
            f"reshape element count cannot be proven equal for "
            f"{input_type.shape} -> {result_type.shape}"
        )
    return result_type


def infer_slice(
    input_type: TensorType,
    *,
    axis: int,
    start: int,
    stop: int,
    step: int,
) -> TensorType:
    """Infer one bounded positive-stride slice along a concrete source axis."""
    if (
        not isinstance(axis, int)
        or isinstance(axis, bool)
        or axis < 0
        or axis >= len(input_type.shape)
    ):
        raise TypeInferenceError("slice axis is out of range")
    for name, value in (("start", start), ("stop", stop), ("step", step)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeInferenceError(f"slice {name} must be an integer")
    if step <= 0:
        raise TypeInferenceError("slice step must be a positive integer")

    extent = input_type.shape[axis]
    if not isinstance(extent, int) or isinstance(extent, bool):
        raise TypeInferenceError("slice axis extent must be concrete before slicing")
    if start < 0 or stop < 0 or start > stop or stop > extent:
        raise TypeInferenceError(
            f"slice bounds must satisfy 0 <= start <= stop <= extent ({extent})"
        )

    shape = list(input_type.shape)
    shape[axis] = (stop - start + step - 1) // step
    return TensorType(tuple(shape), input_type.dtype)


def infer_reverse(input_type: TensorType, axis: int) -> TensorType:
    """Infer one read-only axis reversal without changing logical tensor type."""
    if (
        not isinstance(axis, int)
        or isinstance(axis, bool)
        or axis < 0
        or axis >= len(input_type.shape)
    ):
        raise TypeInferenceError("reverse axis is out of range")
    return input_type


def infer_transpose(input_type: TensorType, axes: Iterable[int]) -> TensorType:
    """Infer one full compile-time axis permutation without changing storage dtype."""
    try:
        permutation = normalize_permutation(len(input_type.shape), tuple(axes))
    except (TypeError, ValueError) as exc:
        raise TypeInferenceError(str(exc)) from exc
    return TensorType(tuple(input_type.shape[axis] for axis in permutation), input_type.dtype)


Monomial = tuple[tuple[SymbolicDim, int], ...]
Polynomial = dict[Monomial, int]


def _shape_element_polynomial(shape: tuple[ShapeDim, ...]) -> Polynomial:
    polynomial: Polynomial = {(): 1}
    for dim in shape:
        polynomial = _multiply_polynomials(polynomial, _dim_polynomial(dim))
    return polynomial


def _dim_polynomial(dim: ShapeDim) -> Polynomial:
    if isinstance(dim, bool):
        raise TypeInferenceError("tensor dimensions must not be bool")
    if isinstance(dim, int):
        return {} if dim == 0 else {(): dim}
    if isinstance(dim, SymbolicDim):
        return {((dim, 1),): 1}
    if isinstance(dim, AffineDim):
        polynomial: Polynomial = {((dim.symbol, 1),): dim.scale}
        if dim.offset:
            polynomial[()] = dim.offset
        return polynomial
    if isinstance(dim, LinearDim):
        polynomial = {
            ((symbol, 1),): coefficient for symbol, coefficient in dim.terms
        }
        if dim.offset:
            polynomial[()] = dim.offset
        return polynomial
    raise TypeInferenceError(f"unsupported tensor dimension: {dim!r}")


def _multiply_polynomials(lhs: Polynomial, rhs: Polynomial) -> Polynomial:
    if not lhs or not rhs:
        return {}
    product: Polynomial = {}
    for lhs_monomial, lhs_coefficient in lhs.items():
        for rhs_monomial, rhs_coefficient in rhs.items():
            powers: dict[SymbolicDim, int] = {}
            for symbol, exponent in (*lhs_monomial, *rhs_monomial):
                powers[symbol] = powers.get(symbol, 0) + exponent
            monomial = tuple(sorted(powers.items(), key=lambda item: item[0].name))
            coefficient = lhs_coefficient * rhs_coefficient
            product[monomial] = product.get(monomial, 0) + coefficient
    return {
        monomial: coefficient for monomial, coefficient in product.items() if coefficient
    }


def _broadcast_shapes(
    lhs: tuple[ShapeDim, ...],
    rhs: tuple[ShapeDim, ...],
) -> tuple[ShapeDim, ...]:
    rank = max(len(lhs), len(rhs))
    lhs_padded: tuple[ShapeDim, ...] = (1,) * (rank - len(lhs)) + lhs
    rhs_padded: tuple[ShapeDim, ...] = (1,) * (rank - len(rhs)) + rhs
    result: list[ShapeDim] = []

    for lhs_dim, rhs_dim in zip(lhs_padded, rhs_padded, strict=True):
        if lhs_dim == rhs_dim:
            result.append(lhs_dim)
            continue
        if lhs_dim == 1:
            result.append(rhs_dim)
            continue
        if rhs_dim == 1:
            result.append(lhs_dim)
            continue
        if isinstance(lhs_dim, (SymbolicDim, AffineDim, LinearDim)) or isinstance(
            rhs_dim, (SymbolicDim, AffineDim, LinearDim)
        ):
            raise TypeInferenceError(
                f"cannot broadcast symbolic dimensions {lhs_dim} and {rhs_dim} "
                f"in shapes {lhs} and {rhs}"
            )
        raise TypeInferenceError(f"cannot broadcast shapes {lhs} and {rhs}")

    return tuple(result)
