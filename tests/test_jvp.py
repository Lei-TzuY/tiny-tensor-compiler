from __future__ import annotations

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    compile_module,
    execute_cpu,
    execute_loop,
    execute_reference,
    jacobian_vector_product_module,
    lower_to_cpu,
    lower_to_loops,
)
from tiny_tensor_compiler.autodiff import AutodiffError
from tiny_tensor_compiler.ir import DType


def _execute_all_backends(module, inputs):
    reference = execute_reference(module, inputs=inputs)
    cpu_program = lower_to_cpu(module)
    cpu = execute_cpu(cpu_program, inputs=inputs)
    loop = execute_loop(lower_to_loops(cpu_program), inputs=inputs)
    native = compile_module(module)(inputs=inputs)
    return reference, cpu, loop, native


def test_jvp_broadcast_product_rule_and_sum_across_backends():
    builder = GraphBuilder("jvp-broadcast")
    x = builder.input((2, 3), DType.FLOAT64)
    y = builder.input((1, 3), DType.FLOAT64)
    output = (x * y + x).sum(axis=0)
    module = builder.finish(output)

    jvp = jacobian_vector_product_module(module, wrt=(0, 1))
    x_value = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -1.0]], dtype=np.float64)
    y_value = np.array([[2.0, -3.0, 0.25]], dtype=np.float64)
    x_tangent = np.array([[0.5, 2.0, -1.0], [3.0, -4.0, 0.75]], dtype=np.float64)
    y_tangent = np.array([[1.5, -0.25, 2.0]], dtype=np.float64)
    expected = np.sum(
        x_tangent * y_value + x_value * y_tangent + x_tangent,
        axis=0,
    )

    for actual in _execute_all_backends(
        jvp,
        (x_value, y_value, x_tangent, y_tangent),
    ):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_jvp_alias_chain_preserves_tensor_output_across_backends():
    builder = GraphBuilder("jvp-alias")
    x = builder.input((2, 3, 6), DType.FLOAT64)
    aliased = (
        x.view((6, 6))
        .transpose((1, 0))
        .reverse(0)
        .slice(axis=0, start=1, stop=6, step=2)
    )
    module = builder.finish((x, aliased))

    jvp = jacobian_vector_product_module(module, output_index=1, wrt=(0,))
    x_value = np.arange(36, dtype=np.float64).reshape(2, 3, 6) - 11.0
    tangent = np.arange(36, dtype=np.float64).reshape(2, 3, 6) * 0.125 - 2.0
    expected = tangent.reshape(6, 6).T[::-1][1:6:2]

    for actual in _execute_all_backends(jvp, (x_value, tangent)):
        np.testing.assert_array_equal(actual, expected)


def test_jvp_composes_through_matmul_primitives():
    builder = GraphBuilder("jvp-matmul")
    lhs = builder.input((2, 3), DType.FLOAT64)
    rhs = builder.input((3, 2), DType.FLOAT64)
    module = builder.finish(lhs @ rhs)

    jvp = jacobian_vector_product_module(module, wrt=(0,))
    lhs_value = np.array([[1.0, -2.0, 3.0], [4.0, 0.5, -1.5]], dtype=np.float64)
    rhs_value = np.array([[2.0, -1.0], [0.25, 3.0], [-2.0, 4.0]], dtype=np.float64)
    tangent = np.array([[0.5, 2.0, -1.0], [-3.0, 1.25, 4.0]], dtype=np.float64)
    expected = tangent @ rhs_value

    for actual in _execute_all_backends(jvp, (lhs_value, rhs_value, tangent)):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_jvp_returns_exact_zero_when_requested_input_does_not_reach_output():
    builder = GraphBuilder("jvp-unused")
    x = builder.input((3,), DType.FLOAT32)
    builder.input((3,), DType.FLOAT32)
    module = builder.finish(x * x)

    jvp = jacobian_vector_product_module(module, wrt=(1,))
    x_value = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    y_value = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    tangent = np.array([-1.0, 2.0, 0.5], dtype=np.float32)

    for actual in _execute_all_backends(jvp, (x_value, y_value, tangent)):
        np.testing.assert_array_equal(actual, np.zeros_like(x_value))


@pytest.mark.parametrize(
    "build",
    [
        lambda x: x.relu(),
        lambda x: x + 0.0,
    ],
)
def test_jvp_rejects_unsupported_ops_and_write_effects_fail_closed(build):
    builder = GraphBuilder("jvp-unsupported")
    x = builder.input((4,), DType.FLOAT64)
    value = build(x)
    if value.value.producer is not None and value.value.producer.opcode == "add":
        target = value.slice(axis=0, start=0, stop=2, step=1)
        patch = builder.input((2,), DType.FLOAT64)
        value = value.copy_into(target, patch)
    module = builder.finish(value)

    with pytest.raises(AutodiffError, match="unsupported.*forward-mode"):
        jacobian_vector_product_module(module, wrt=(0,))


def test_jvp_rejects_symbolic_and_mixed_precision_contracts():
    from tiny_tensor_compiler import SymbolicDim

    batch = SymbolicDim("B")
    builder = GraphBuilder("jvp-symbolic")
    x = builder.input((batch, 2), DType.FLOAT64)
    module = builder.finish(x * x)
    with pytest.raises(AutodiffError, match="static"):
        jacobian_vector_product_module(module, wrt=(0,))

    builder = GraphBuilder("jvp-mixed")
    x = builder.input((2,), DType.FLOAT32)
    y = builder.input((2,), DType.FLOAT64)
    module = builder.finish(x * y)
    with pytest.raises(AutodiffError, match="mixed-precision"):
        jacobian_vector_product_module(module, wrt=(0,))
