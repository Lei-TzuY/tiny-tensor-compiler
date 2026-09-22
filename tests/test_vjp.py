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
    vector_jacobian_product_module,
)
from tiny_tensor_compiler.autodiff import AutodiffError, differentiate_module
from tiny_tensor_compiler.ir import DType


def _execute_all_backends(module, inputs):
    reference = execute_reference(module, inputs=inputs)
    cpu_program = lower_to_cpu(module)
    cpu = execute_cpu(cpu_program, inputs=inputs)
    loop = execute_loop(lower_to_loops(cpu_program), inputs=inputs)
    native = compile_module(module)(inputs=inputs)
    return reference, cpu, loop, native


def test_runtime_seeded_vjp_supports_vector_outputs_across_backends():
    builder = GraphBuilder("vector-vjp")
    x = builder.input((3,), DType.FLOAT64)
    module = builder.finish(x * x)

    vjp = vector_jacobian_product_module(module, wrt=(0,))
    x_value = np.array([1.5, -2.0, 0.25], dtype=np.float64)
    cotangent = np.array([2.0, -3.0, 4.0], dtype=np.float64)
    expected = 2.0 * x_value * cotangent

    for actual in _execute_all_backends(vjp, (x_value, cotangent)):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_runtime_seeded_vjp_composes_into_hessian_vector_product():
    builder = GraphBuilder("cubic-loss")
    x = builder.input((4,), DType.FLOAT64)
    loss = (x * x * x).sum()
    module = builder.finish(loss)

    gradient = differentiate_module(module, wrt=(0,))
    hvp = vector_jacobian_product_module(gradient, wrt=(0,))

    x_value = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float64)
    vector = np.array([2.0, 3.0, -4.0, 0.25], dtype=np.float64)
    expected = 6.0 * x_value * vector

    for actual in _execute_all_backends(hvp, (x_value, vector)):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_runtime_seeded_vjp_differentiates_direct_slice_copy_into_across_backends():
    builder = GraphBuilder("copy-into-vjp")
    base = builder.input((6,), DType.FLOAT64)
    patch = builder.input((3,), DType.FLOAT64)
    owned = base + builder.tensor(0.0, dtype=DType.FLOAT64)
    target = owned.slice(axis=0, start=1, stop=6, step=2)
    module = builder.finish(owned.copy_into(target, patch))

    vjp = vector_jacobian_product_module(module, wrt=(0, 1))
    base_value = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=np.float64)
    patch_value = np.array([7.0, 8.0, 9.0], dtype=np.float64)
    cotangent = np.array(
        [np.inf, -3.0, -0.0, 5.0, 11.0, -7.0],
        dtype=np.float64,
    )
    expected_base = np.array(
        [np.inf, 0.0, -0.0, 0.0, 11.0, 0.0],
        dtype=np.float64,
    )
    expected_patch = cotangent[1:6:2]

    for actual in _execute_all_backends(
        vjp,
        (base_value, patch_value, cotangent),
    ):
        assert isinstance(actual, tuple)
        np.testing.assert_array_equal(actual[0], expected_base)
        np.testing.assert_array_equal(actual[1], expected_patch)


def test_runtime_seeded_vjp_composes_through_slice_scatter_gradient():
    builder = GraphBuilder("effectful-gradient")
    x = builder.input((6,), DType.FLOAT64)
    sliced = x.slice(axis=0, start=1, stop=6, step=2)
    gradient = differentiate_module(
        builder.finish((sliced * sliced).sum()),
        wrt=(0,),
    )
    assert "copy_into" in gradient.dump()

    hvp = vector_jacobian_product_module(gradient, wrt=(0,))
    x_value = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=np.float64)
    vector = np.array([2.0, 3.0, -4.0, 0.25, 7.0, -5.0], dtype=np.float64)
    expected = np.zeros_like(vector)
    expected[1:6:2] = 2.0 * vector[1:6:2]

    for actual in _execute_all_backends(hvp, (x_value, vector)):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_runtime_seeded_vjp_rejects_non_slice_copy_target():
    builder = GraphBuilder("non-slice-copy-target")
    base = builder.input((4,), DType.FLOAT64)
    patch = builder.input((4,), DType.FLOAT64)
    owned = base + builder.tensor(0.0, dtype=DType.FLOAT64)
    module = builder.finish(owned.copy_into(owned.reverse(0), patch))

    with pytest.raises(
        AutodiffError,
        match="copy_into backward currently requires a direct slice target",
    ):
        vector_jacobian_product_module(module, wrt=(0, 1))



def test_runtime_seeded_vjp_tapes_prewrite_slice_primal_across_backends():
    builder = GraphBuilder("copy-source-prewrite-dependency")
    base = builder.input((6,), DType.FLOAT64)
    owned = base + builder.tensor(0.0, dtype=DType.FLOAT64)
    target = owned.slice(axis=0, start=1, stop=6, step=2)
    source = target * target
    module = builder.finish(owned.copy_into(target, source))

    vjp = vector_jacobian_product_module(module, wrt=(0,))
    base_value = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=np.float64)
    cotangent = np.array([2.0, 3.0, -4.0, 0.25, 7.0, -5.0], dtype=np.float64)
    expected = np.array(cotangent, copy=True)
    expected[1:6:2] = (
        2.0 * base_value[1:6:2] * cotangent[1:6:2]
    )

    for actual in _execute_all_backends(vjp, (base_value, cotangent)):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)



def test_runtime_seeded_vjp_tapes_each_ordered_copy_generation():
    builder = GraphBuilder("ordered-copy-primal-tape")
    base = builder.input((6,), DType.FLOAT64)
    owned = base + builder.tensor(0.0, dtype=DType.FLOAT64)

    even_target = owned.slice(axis=0, start=0, stop=6, step=2)
    even_source = even_target * even_target
    after_even = owned.copy_into(even_target, even_source)

    odd_target = after_even.slice(axis=0, start=1, stop=6, step=2)
    odd_source = odd_target * odd_target
    module = builder.finish(after_even.copy_into(odd_target, odd_source))

    vjp = vector_jacobian_product_module(module, wrt=(0,))
    base_value = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=np.float64)
    cotangent = np.array([2.0, 3.0, -4.0, 0.25, 7.0, -5.0], dtype=np.float64)
    expected = 2.0 * base_value * cotangent

    for actual in _execute_all_backends(vjp, (base_value, cotangent)):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)



def test_runtime_seeded_vjp_differentiates_binary_into_add_across_backends():
    builder = GraphBuilder("binary-into-add-vjp")
    base = builder.input((6,), DType.FLOAT64)
    source = builder.input((3,), DType.FLOAT64)
    root = base + builder.tensor(0.0, dtype=DType.FLOAT64)
    target = root.slice(axis=0, start=1, stop=6, step=2)
    module = builder.finish(root.binary_into(target, source, operator="add"))

    vjp = vector_jacobian_product_module(module, wrt=(0, 1))
    base_value = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=np.float64)
    source_value = np.array([7.0, -8.0, 9.0], dtype=np.float64)
    cotangent = np.array([2.0, 3.0, -4.0, 0.25, 7.0, -5.0], dtype=np.float64)

    expected_base = np.array(cotangent, copy=True)
    expected_source = cotangent[1:6:2]

    for actual in _execute_all_backends(
        vjp,
        (base_value, source_value, cotangent),
    ):
        assert isinstance(actual, tuple)
        np.testing.assert_allclose(actual[0], expected_base, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(actual[1], expected_source, rtol=0.0, atol=0.0)


def test_runtime_seeded_vjp_differentiates_broadcast_binary_into_mul_across_backends():
    builder = GraphBuilder("binary-into-mul-vjp")
    base = builder.input((6,), DType.FLOAT64)
    scale = builder.input((), DType.FLOAT64)
    root = base + builder.tensor(0.0, dtype=DType.FLOAT64)
    target = root.slice(axis=0, start=0, stop=6, step=2)
    module = builder.finish(root.binary_into(target, scale, operator="mul"))

    vjp = vector_jacobian_product_module(module, wrt=(0, 1))
    base_value = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=np.float64)
    scale_value = np.array(-1.5, dtype=np.float64)
    cotangent = np.array([2.0, 3.0, -4.0, 0.25, 7.0, -5.0], dtype=np.float64)

    expected_base = np.array(cotangent, copy=True)
    expected_base[0:6:2] *= scale_value
    expected_scale = np.array(
        np.sum(cotangent[0:6:2] * base_value[0:6:2]),
        dtype=np.float64,
    )

    for actual in _execute_all_backends(
        vjp,
        (base_value, scale_value, cotangent),
    ):
        assert isinstance(actual, tuple)
        np.testing.assert_allclose(actual[0], expected_base, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(actual[1], expected_scale, rtol=0.0, atol=0.0)
