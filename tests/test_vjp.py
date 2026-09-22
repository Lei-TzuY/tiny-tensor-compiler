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


def test_runtime_seeded_vjp_keeps_effectful_higher_order_fail_closed():
    builder = GraphBuilder("effectful-gradient")
    x = builder.input((6,), DType.FLOAT64)
    weights = builder.input((3,), DType.FLOAT64)
    sliced = x.reverse(0).slice(axis=0, start=1, stop=6, step=2)
    gradient = differentiate_module(
        builder.finish((sliced * weights).sum()),
        wrt=(0,),
    )
    assert "copy_into" in gradient.dump()

    with pytest.raises(AutodiffError, match="unsupported.*copy_into.*backward"):
        vector_jacobian_product_module(gradient, wrt=(0,))
