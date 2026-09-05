from __future__ import annotations

import numpy as np

from .ir import AffineDim, DType, LinearDim, ShapeDim, SymbolicDim, TensorType


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
