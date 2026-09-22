from __future__ import annotations

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    compile_module,
    execute_cpu,
    execute_loop,
    execute_reference,
    lower_to_cpu,
    lower_to_loops,
)
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


def test_reverse_mode_composes_through_view_and_matmul_across_backends():
    builder = GraphBuilder()
    lhs = builder.input((2, 3), DType.FLOAT64)
    rhs = builder.input((3, 2), DType.FLOAT64)
    viewed = lhs.view((3, 2))
    view_loss = (viewed * viewed).sum()
    matmul_loss = (lhs @ rhs).sum()
    module = builder.finish((view_loss, matmul_loss))

    view_grad = differentiate_module(module, output_index=0, wrt=(0,))
    matmul_grad = differentiate_module(module, output_index=1, wrt=(0, 1))

    lhs_value = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -1.5]], dtype=np.float64)
    rhs_value = np.array([[2.0, -1.0], [0.25, 3.0], [-2.0, 4.0]], dtype=np.float64)

    expected_view = 2.0 * lhs_value
    np.testing.assert_allclose(
        execute_reference(view_grad, inputs=(lhs_value, rhs_value)),
        expected_view,
        rtol=0.0,
        atol=0.0,
    )

    ones = np.ones((2, 2), dtype=np.float64)
    expected_lhs = ones @ rhs_value.T
    expected_rhs = lhs_value.T @ ones

    reference = execute_reference(matmul_grad, inputs=(lhs_value, rhs_value))
    cpu_program = lower_to_cpu(matmul_grad)
    cpu = execute_cpu(cpu_program, inputs=(lhs_value, rhs_value))
    loop = execute_loop(lower_to_loops(cpu_program), inputs=(lhs_value, rhs_value))
    native = compile_module(matmul_grad)(inputs=(lhs_value, rhs_value))

    for actual in (reference, cpu, loop, native):
        assert isinstance(actual, tuple)
        np.testing.assert_allclose(actual[0], expected_lhs, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(actual[1], expected_rhs, rtol=0.0, atol=0.0)


def test_reverse_mode_alias_vjp_scatter_composes_across_backends():
    builder = GraphBuilder()
    x = builder.input((2, 3, 6), DType.FLOAT64)
    weights = builder.input((3, 2, 3), DType.FLOAT64)
    aliased = x.transpose((2, 0, 1)).reverse(0).slice(
        axis=0,
        start=1,
        stop=6,
        step=2,
    )
    module = builder.finish((aliased * weights).sum())

    differentiated = differentiate_module(module, wrt=(0,))
    x_value = np.arange(36, dtype=np.float64).reshape(2, 3, 6) - 11.0
    weights_value = np.array(
        [
            [[1.0, -2.0, 3.0], [4.0, 0.5, -1.0]],
            [[-3.0, 2.5, 1.0], [0.25, -4.0, 2.0]],
            [[5.0, -0.5, 1.5], [-2.0, 3.0, 4.0]],
        ],
        dtype=np.float64,
    )

    scattered = np.zeros((6, 2, 3), dtype=np.float64)
    scattered[1:6:2] = weights_value
    expected = scattered[::-1].transpose((1, 2, 0))

    cpu_program = lower_to_cpu(differentiated)
    loops = lower_to_loops(cpu_program)
    assert len(loops.copies) == 1
    assert "copy_into" in differentiated.dump()

    reference = execute_reference(differentiated, inputs=(x_value, weights_value))
    cpu = execute_cpu(cpu_program, inputs=(x_value, weights_value))
    loop = execute_loop(loops, inputs=(x_value, weights_value))
    native = compile_module(
        differentiated,
        borrow_inputs=True,
        parallel=True,
    )(inputs=(x_value, weights_value))

    for actual in (reference, cpu, loop, native):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "build",
    [
        lambda x: x.relu().sum(),
        lambda x: x.prod(),
    ],
)
def test_reverse_mode_rejects_unsupported_backward_ops_fail_closed(build):
    builder = GraphBuilder()
    x = builder.input((2, 3), DType.FLOAT32)
    module = builder.finish(build(x))

    with pytest.raises(AutodiffError, match="unsupported.*backward"):
        differentiate_module(module, wrt=(0,))


def test_reverse_mode_rejects_mixed_precision_backward_slice_fail_closed():
    builder = GraphBuilder()
    x = builder.input((2,), DType.FLOAT32)
    y = builder.input((2,), DType.FLOAT64)
    module = builder.finish((x * y).sum())

    with pytest.raises(AutodiffError, match="mixed-precision"):
        differentiate_module(module, wrt=(0, 1))


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
