from __future__ import annotations

import numpy as np

from tiny_tensor_compiler import (
    GraphBuilder,
    compile_module,
    execute_cpu,
    execute_loop,
    execute_reference,
    lower_to_cpu,
    lower_to_loops,
    value_and_jacobian_vector_product_module,
)
from tiny_tensor_compiler.ir import DType


def _execute_all_backends(module, inputs):
    reference = execute_reference(module, inputs=inputs)
    cpu_program = lower_to_cpu(module)
    cpu = execute_cpu(cpu_program, inputs=inputs)
    loop = execute_loop(lower_to_loops(cpu_program), inputs=inputs)
    native = compile_module(module)(inputs=inputs)
    return reference, cpu, loop, native


def test_value_and_jvp_shares_primal_graph_across_backends():
    builder = GraphBuilder("value-and-jvp")
    x = builder.input((2, 3), DType.FLOAT64)
    y = builder.input((1, 3), DType.FLOAT64)
    output = (x * y + x).sum(axis=0)
    module = builder.finish(output)

    linearized = value_and_jacobian_vector_product_module(
        module,
        wrt=(0, 1),
    )
    x_value = np.array(
        [[1.0, -2.0, 3.0], [4.0, 0.5, -1.0]],
        dtype=np.float64,
    )
    y_value = np.array([[2.0, -3.0, 0.25]], dtype=np.float64)
    x_tangent = np.array(
        [[0.5, 2.0, -1.0], [3.0, -4.0, 0.75]],
        dtype=np.float64,
    )
    y_tangent = np.array([[1.5, -0.25, 2.0]], dtype=np.float64)
    expected_value = np.sum(x_value * y_value + x_value, axis=0)
    expected_tangent = np.sum(
        x_tangent * y_value + x_value * y_tangent + x_tangent,
        axis=0,
    )

    for actual in _execute_all_backends(
        linearized,
        (x_value, y_value, x_tangent, y_tangent),
    ):
        assert isinstance(actual, tuple)
        assert len(actual) == 2
        np.testing.assert_allclose(actual[0], expected_value, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(actual[1], expected_tangent, rtol=0.0, atol=0.0)


def test_value_and_jvp_preserves_primal_and_tangent_write_generations():
    builder = GraphBuilder("value-and-jvp-write")
    base = builder.input((6,), DType.FLOAT64)
    scale = builder.input((), DType.FLOAT64)
    root = base + builder.tensor(0.0, dtype=DType.FLOAT64)
    target = root.slice(axis=0, start=0, stop=6, step=2)
    module = builder.finish(root.binary_into(target, scale, operator="mul"))

    linearized = value_and_jacobian_vector_product_module(
        module,
        wrt=(0, 1),
    )
    base_value = np.array(
        [1.0, -2.0, 3.0, -4.0, 5.0, -6.0],
        dtype=np.float64,
    )
    scale_value = np.array(-1.5, dtype=np.float64)
    base_tangent = np.array(
        [0.5, -1.0, 1.5, -2.0, 2.5, -3.0],
        dtype=np.float64,
    )
    scale_tangent = np.array(0.25, dtype=np.float64)

    expected_value = np.array(base_value, copy=True)
    expected_value[0:6:2] *= scale_value
    expected_tangent = np.array(base_tangent, copy=True)
    expected_tangent[0:6:2] = (
        base_tangent[0:6:2] * scale_value
        + base_value[0:6:2] * scale_tangent
    )

    for actual in _execute_all_backends(
        linearized,
        (base_value, scale_value, base_tangent, scale_tangent),
    ):
        assert isinstance(actual, tuple)
        assert len(actual) == 2
        np.testing.assert_allclose(actual[0], expected_value, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(actual[1], expected_tangent, rtol=0.0, atol=0.0)
