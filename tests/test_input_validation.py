from __future__ import annotations

import numpy as np

from tiny_tensor_compiler.input_validation import prepare_runtime_inputs
from tiny_tensor_compiler.ir import DType, TensorType


def test_prepare_runtime_inputs_preserves_scalar_shape_after_copy_normalization():
    expected = (TensorType((), DType.FLOAT64),)
    value = np.array(-1.5, dtype=np.float64)

    (prepared,) = prepare_runtime_inputs(expected, (value,))

    assert prepared.shape == ()
    assert prepared.dtype == np.dtype(np.float64)
    assert prepared.flags.c_contiguous
    assert prepared.flags.aligned
    assert prepared is not value
    np.testing.assert_array_equal(prepared, value)
