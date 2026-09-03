from __future__ import annotations

import numpy as np

from .ir import DType, TensorType


class TypeInferenceError(ValueError):
    pass


def infer_binary(lhs: TensorType, rhs: TensorType) -> TensorType:
    try:
        shape = tuple(np.broadcast_shapes(lhs.shape, rhs.shape))
    except ValueError as exc:
        raise TypeInferenceError(
            f"cannot broadcast shapes {lhs.shape} and {rhs.shape}"
        ) from exc
    try:
        dtype = DType.from_numpy(np.result_type(lhs.dtype.to_numpy(), rhs.dtype.to_numpy()))
    except TypeError as exc:
        raise TypeInferenceError(str(exc)) from exc
    return TensorType(shape, dtype)


def infer_relu(input_type: TensorType) -> TensorType:
    if input_type.dtype not in {DType.INT32, DType.INT64, DType.FLOAT32, DType.FLOAT64}:
        raise TypeInferenceError(f"relu requires a numeric tensor, got {input_type.dtype.value}")
    return input_type
