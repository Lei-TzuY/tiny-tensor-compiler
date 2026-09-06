from __future__ import annotations

import numpy as np
import pytest

from tiny_tensor_compiler import GraphBuilder, compile_module, execute_reference
from tiny_tensor_compiler.autodiff import AutodiffError, differentiate_module
from tiny_tensor_compiler.ir import DType


def _scalar_loss_module():
    builder = GraphBuilder()
    x = builder.input((2, 3), DType.FLOAT32)
    y = builder.input((1, 3), DType.FLOAT32)
    bias = builder.input((), DType.FLOAT32)
    product = x * y
    shifted = product + bias
    loss = shifted.sum()
    return builder.finish(loss)


def test_reverse_mode_matches_closed_form_with_broadcast_unbroadcast():
    module = _scalar_loss_module()
    differentiated = differentiate_module(module, wrt=(0, 1, 2))

    x = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float32)
    y = np.array([[2.0, -3.0, 0.5]], dtype=np.float32)
    bias = np.array(1.25, dtype=np.float32)
    dx, dy, dbias = execute_reference(differentiated, inputs=(x, y, bias))

    np.testing.assert_array_equal(dx, np.broadcast_to(y, x.shape))
    np.testing.assert_array_equal(dy, np.sum(x, axis=0, keepdims=True, dtype=np.float32))
    np.testing.assert_array_equal(dbias, np.array(6.0, dtype=np.float32))


def test_reverse_mode_accumulates_multiple_gradient_paths():
    builder = GraphBuilder()
    x = builder.input((2, 2), DType.FLOAT64)
    loss = (x * x + x).sum()
    module = builder.finish(loss)

    differentiated = differentiate_module(module, wrt=(0,))
    values = np.array([[1.0, -2.0], [3.5, 0.25]], dtype=np.float64)
    (gradient,) = (execute_reference(differentiated, inputs=(values,)),)

    np.testing.assert_allclose(gradient, 2.0 * values + 1.0, rtol=0.0, atol=0.0)


def test_reverse_mode_sum_axis_and_reshape_are_executable_natively():
    builder = GraphBuilder()
    x = builder.input((2, 3, 4), DType.FLOAT32)
    reshaped = x.reshape((6, 4))
    weights = builder.input((1, 4), DType.FLOAT32)
    loss = (reshaped * weights).sum(axis=(0, 1)).reshape(())
    module = builder.finish(loss)

    differentiated = differentiate_module(module, wrt=(0, 1))
    x_value = np.arange(24, dtype=np.float32).reshape(2, 3, 4) - 7.0
    weight_value = np.array([[1.0, -2.0, 0.5, 3.0]], dtype=np.float32)

    expected_dx = np.broadcast_to(weight_value, (6, 4)).reshape(2, 3, 4)
    expected_dw = np.sum(x_value.reshape(6, 4), axis=0, keepdims=True, dtype=np.float32)

    executable = compile_module(differentiated)
    dx, dw = executable(inputs=(x_value, weight_value))
    np.testing.assert_array_equal(dx, expected_dx)
    np.testing.assert_array_equal(dw, expected_dw)


def test_reverse_mode_returns_exact_zero_for_unused_requested_input():
    builder = GraphBuilder()
    x = builder.input((2,), DType.FLOAT32)
    builder.input((3,), DType.FLOAT32)
    loss = (x * x).sum()
    module = builder.finish(loss)

    differentiated = differentiate_module(module, wrt=(1,))
    unused_gradient = execute_reference(
        differentiated,
        inputs=(
            np.array([2.0, -1.0], dtype=np.float32),
            np.array([4.0, 5.0, 6.0], dtype=np.float32),
        ),
    )
    np.testing.assert_array_equal(unused_gradient, np.zeros((3,), dtype=np.float32))


@pytest.mark.parametrize(
    "build",
    [
        lambda x: x.relu().sum(),
        lambda x: x.prod(),
        lambda x: x.transpose((1, 0)).sum(),
    ],
)
def test_reverse_mode_rejects_unsupported_backward_ops_fail_closed(build):
    builder = GraphBuilder()
    x = builder.input((2, 3), DType.FLOAT32)
    module = builder.finish(build(x))

    with pytest.raises(AutodiffError, match="unsupported.*backward"):
        differentiate_module(module, wrt=(0,))


def test_reverse_mode_rejects_non_scalar_symbolic_and_integer_contracts():
    builder = GraphBuilder()
    x = builder.input((2,), DType.FLOAT32)
    vector_module = builder.finish(x * x)
    with pytest.raises(AutodiffError, match="scalar"):
        differentiate_module(vector_module, wrt=(0,))

    builder = GraphBuilder()
    integer = builder.input((2,), DType.INT32)
    integer_module = builder.finish((integer * integer).sum())
    with pytest.raises(AutodiffError, match="floating"):
        differentiate_module(integer_module, wrt=(0,))


def test_reverse_mode_validates_wrt_indices_and_output_index():
    module = _scalar_loss_module()
    with pytest.raises(AutodiffError, match="duplicate"):
        differentiate_module(module, wrt=(0, 0))
    with pytest.raises(AutodiffError, match="runtime input"):
        differentiate_module(module, wrt=(3,))
    with pytest.raises(AutodiffError, match="output index"):
        differentiate_module(module, output_index=1, wrt=(0,))
