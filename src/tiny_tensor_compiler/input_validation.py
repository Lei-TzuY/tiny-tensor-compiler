from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .ir import TensorType


def prepare_runtime_inputs(
    expected_types: Sequence[TensorType],
    inputs: Sequence[Any] = (),
) -> tuple[np.ndarray, ...]:
    """Validate exact static runtime-input contracts and prepare safe arrays."""
    provided = tuple(inputs)
    if len(provided) != len(expected_types):
        raise ValueError(
            f"expected {len(expected_types)} runtime inputs, got {len(provided)}"
        )

    borrow_mask = getattr(expected_types, "borrow_mask", None)
    if borrow_mask is None:
        borrow_mask = (False,) * len(expected_types)
    else:
        borrow_mask = tuple(borrow_mask)
        if len(borrow_mask) != len(expected_types):
            raise ValueError("runtime input borrow mask has the wrong length")

    prepared: list[np.ndarray] = []
    for index, (value, expected_type, borrowed) in enumerate(
        zip(provided, expected_types, borrow_mask, strict=True)
    ):
        if borrowed and not isinstance(value, np.ndarray):
            raise TypeError(
                f"borrowed input {index} must be a numpy.ndarray to preserve zero-copy binding"
            )

        array = value if borrowed else np.asarray(value)
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

        if borrowed:
            if not array.flags.c_contiguous:
                raise ValueError(
                    f"borrowed input {index} must be C-contiguous; implicit copies are disabled"
                )
            if not array.flags.aligned:
                raise ValueError(
                    f"borrowed input {index} must be aligned for its dtype"
                )
            prepared.append(array)
        else:
            prepared.append(np.ascontiguousarray(array))

    return tuple(prepared)
