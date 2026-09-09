from __future__ import annotations

from typing import Any

import numpy as np

from .frontend import Tensor
from .inference import TypeInferenceError
from .ir import DType

# Conversions accepted here must preserve every value representable by the source dtype.
# Keep this table deliberately smaller than NumPy's general casting surface.
_LOSSLESS_CASTS = frozenset(
    {
        (DType.INT32, DType.INT64),
        (DType.INT32, DType.FLOAT64),
        (DType.FLOAT32, DType.FLOAT64),
    }
)


def lossless_cast(
    tensor: Tensor,
    dtype: str | np.dtype[Any] | DType,
) -> Tensor:
    """Convert one tensor only when the target exactly represents every source value.

    The conversion is canonicalized to the compiler's already-verified binary multiply
    semantics with a scalar one in the target dtype. This keeps conversion semantics on
    the existing Tensor IR / Buffer IR / Loop IR / C/native execution path rather than
    introducing a second backend-specific cast implementation.
    """
    if not isinstance(tensor, Tensor):
        raise TypeError("lossless_cast requires a Tensor")

    try:
        target = dtype if isinstance(dtype, DType) else DType.from_numpy(np.dtype(dtype))
    except (TypeError, ValueError) as exc:
        raise TypeInferenceError(str(exc)) from exc

    source = tensor.type.dtype
    if source == target:
        return tensor
    if (source, target) not in _LOSSLESS_CASTS:
        raise TypeInferenceError(
            f"lossless cast is not defined for {source.value} -> {target.value}"
        )

    one = tensor._builder.tensor(np.array(1, dtype=target.to_numpy()))
    result = tensor * one
    if result.type.shape != tensor.type.shape or result.type.dtype != target:
        raise RuntimeError("internal error: lossless cast lowering produced the wrong tensor type")
    return result
