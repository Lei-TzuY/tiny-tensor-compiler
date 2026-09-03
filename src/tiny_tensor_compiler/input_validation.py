from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .ir import TensorType


def prepare_runtime_inputs(
    expected_types: Sequence[TensorType],
    inputs: Sequence[Any] = (),
) -> tuple[np.ndarray, ...]:
    """Validate exact static runtime-input contracts and return contiguous arrays."""
    provided = tuple(inputs)
    if len(provided) != len(expected_types):
        raise ValueError(
            f"expected {len(expected_types)} runtime inputs, got {len(provided)}"
        )

    prepared: list[np.ndarray] = []
    for index, (value, expected_type) in enumerate(
        zip(provided, expected_types, strict=True)
    ):
        array = np.asarray(value)
        if tuple(array.shape) != expected_type.shape:
            raise ValueError(
                f"input {index} shape {tuple(array.shape)} does not match expected "
                f"{expected_type.shape}"
            )
        expected_dtype = expected_type.dtype.to_numpy()
        if array.dtype != expected_dtype:
            raise ValueError(
                f"input {index} dtype {array.dtype} does not match expected {expected_dtype}"
            )
        prepared.append(np.ascontiguousarray(array))

    return tuple(prepared)
