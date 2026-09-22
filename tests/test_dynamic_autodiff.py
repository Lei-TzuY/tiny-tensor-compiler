from __future__ import annotations

import numpy as np
import pytest

from tiny_tensor_compiler import (
    AutodiffError,
    CompileBudget,
    GraphBuilder,
    SymbolicDim,
    compile_adaptive_dynamic_gradient_module,
    compile_adaptive_dynamic_vjp_module,
    compile_dynamic_gradient_module,
    compile_dynamic_vjp_module,
    differentiate_module,
    specialize_module,
    vector_jacobian_product_module,
)
from tiny_tensor_compiler.analysis import analyze_module
from tiny_tensor_compiler.ir import DType


def test_dynamic_gradient_specializes_before_autodiff_and_reuses_cache():
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    x = builder.input((batch, 6), DType.FLOAT64)
    weights = builder.input((batch, 3), DType.FLOAT64)
    aliased = x.reverse(1).slice(axis=1, start=1, stop=6, step=2)
    loss = (aliased * weights).sum()
    module = builder.finish(loss)

    executable = compile_dynamic_gradient_module(
        module,
        wrt=(0, 1),
        borrow_inputs=True,
        parallel=True,
    )

    assert executable.symbolic_dim == batch

    for size in (2, 5, 2):
        x_value = (
            np.arange(size * 6, dtype=np.float64).reshape(size, 6) - 7.0
        )
        weights_value = (
            np.arange(size * 3, dtype=np.float64).reshape(size, 3) * 0.25 + 1.0
        )

        dx, dweights = executable(inputs=(x_value, weights_value))

        selected = np.array([4, 2, 0])
        expected_dx = np.zeros_like(x_value)
        expected_dx[:, selected] = weights_value
        expected_dweights = x_value[:, selected]

        np.testing.assert_allclose(dx, expected_dx, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(dweights, expected_dweights, rtol=0.0, atol=0.0)

    assert executable.cached_batch_sizes == (2, 5)


def test_dynamic_gradient_fails_closed_after_specialization_for_unsupported_ops():
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    x = builder.input((batch, 3), DType.FLOAT32)
    module = builder.finish(x.relu().sum())
    executable = compile_dynamic_gradient_module(module, wrt=(0,))

    with pytest.raises(AutodiffError, match="unsupported.*backward"):
        executable(
            inputs=(np.ones((2, 3), dtype=np.float32),)
        )



def test_adaptive_dynamic_gradient_uses_concrete_gradient_budget_per_binding():
    batch = SymbolicDim("B")
    width = SymbolicDim("W")
    builder = GraphBuilder("adaptive-dynamic-gradient")
    x = builder.input((batch, width), DType.FLOAT64)
    weights = builder.input((batch, width), DType.FLOAT64)
    module = builder.finish((x * weights).sum())

    small_binding = {batch: 1, width: 2}
    large_binding = {batch: 4, width: 4}
    small_gradient = differentiate_module(
        specialize_module(module, small_binding),
        wrt=(0, 1),
    )
    large_gradient = differentiate_module(
        specialize_module(module, large_binding),
        wrt=(0, 1),
    )
    small_bytes = analyze_module(small_gradient).planned_owning_storage_bytes
    large_bytes = analyze_module(large_gradient).planned_owning_storage_bytes
    assert small_bytes < large_bytes

    executable = compile_adaptive_dynamic_gradient_module(
        module,
        budget=CompileBudget(max_planned_storage_bytes=small_bytes),
        wrt=(0, 1),
        borrow_inputs=True,
        parallel=True,
    )

    small_x = np.array([[1.0, -2.0]], dtype=np.float64)
    small_weights = np.array([[3.0, 4.0]], dtype=np.float64)
    small_dx, small_dweights = executable(inputs=(small_x, small_weights))
    np.testing.assert_allclose(small_dx, small_weights, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(small_dweights, small_x, rtol=0.0, atol=0.0)

    large_x = np.arange(16, dtype=np.float64).reshape(4, 4) - 5.0
    large_weights = (
        np.arange(16, dtype=np.float64).reshape(4, 4) * 0.5 + 1.0
    )
    large_dx, large_dweights = executable(inputs=(large_x, large_weights))
    np.testing.assert_allclose(large_dx, large_weights, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(large_dweights, large_x, rtol=0.0, atol=0.0)

    small_specialization = executable.specialize({width: 2, batch: 1})
    large_specialization = executable.specialize({width: 4, batch: 4})
    assert small_specialization.backend == "native"
    assert small_specialization.budget_exceeded is None
    assert large_specialization.backend == "loop"
    assert large_specialization.budget_exceeded is not None
    assert large_specialization.budget_exceeded.metric == "planned_owning_storage_bytes"
    assert large_specialization.budget_exceeded.limit == small_bytes
    assert large_specialization.budget_exceeded.actual == large_bytes

    assert executable.specialize(small_binding) is small_specialization
    assert executable.specialize(large_binding) is large_specialization
    assert executable.cached_bindings == (
        (("B", 1), ("W", 2)),
        (("B", 4), ("W", 4)),
    )
    assert executable.cached_binding_backends == (
        ((("B", 1), ("W", 2)), "native"),
        ((("B", 4), ("W", 4)), "loop"),
    )


def test_adaptive_dynamic_gradient_requires_explicit_budget():
    batch = SymbolicDim("B")
    builder = GraphBuilder("adaptive-dynamic-gradient-budget")
    x = builder.input((batch, 2), DType.FLOAT64)
    module = builder.finish((x * x).sum())

    with pytest.raises(TypeError, match="budget must be a CompileBudget"):
        compile_adaptive_dynamic_gradient_module(  # type: ignore[arg-type]
            module,
            budget=None,
        )



def test_dynamic_vjp_specializes_output_cotangent_shape_and_reuses_multi_symbol_cache():
    batch = SymbolicDim("B")
    width = SymbolicDim("W")
    builder = GraphBuilder("dynamic-vjp")
    x = builder.input((batch, width), DType.FLOAT64)
    module = builder.finish(x.transpose((1, 0)))

    executable = compile_dynamic_vjp_module(
        module,
        wrt=(0,),
        borrow_inputs=True,
    )

    cases = (
        (2, 3),
        (4, 1),
        (2, 3),
    )
    for batch_size, width_size in cases:
        x_value = np.arange(
            batch_size * width_size,
            dtype=np.float64,
        ).reshape(batch_size, width_size)
        cotangent = (
            np.arange(
                width_size * batch_size,
                dtype=np.float64,
            ).reshape(width_size, batch_size)
            + 0.5
        )

        actual = executable(inputs=(x_value, cotangent))
        np.testing.assert_array_equal(actual, cotangent.transpose(1, 0))

    assert executable.cached_bindings == (
        (("B", 2), ("W", 3)),
        (("B", 4), ("W", 1)),
    )


def test_dynamic_vjp_validates_appended_cotangent_after_forward_shape_binding():
    batch = SymbolicDim("B")
    width = SymbolicDim("W")
    builder = GraphBuilder("dynamic-vjp-cotangent-contract")
    x = builder.input((batch, width), DType.FLOAT32)
    module = builder.finish(x.transpose((1, 0)))
    executable = compile_dynamic_vjp_module(module, wrt=(0,))

    x_value = np.ones((2, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        executable(
            inputs=(
                x_value,
                np.ones((2, 3), dtype=np.float32),
            )
        )

    with pytest.raises(ValueError):
        executable(
            inputs=(
                x_value,
                np.ones((3, 2), dtype=np.float64),
            )
        )

    assert executable.cached_bindings == ((("B", 2), ("W", 3)),)


def test_dynamic_vjp_requires_forward_inputs_plus_one_cotangent():
    batch = SymbolicDim("B")
    builder = GraphBuilder("dynamic-vjp-input-count")
    x = builder.input((batch, 2), DType.FLOAT64)
    module = builder.finish(x * x)
    executable = compile_dynamic_vjp_module(module, wrt=(0,))

    x_value = np.ones((3, 2), dtype=np.float64)
    cotangent = np.ones((3, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="1 forward inputs plus one cotangent"):
        executable(inputs=(x_value,))

    with pytest.raises(ValueError, match="1 forward inputs plus one cotangent"):
        executable(inputs=(x_value, cotangent, cotangent))



def test_adaptive_dynamic_vjp_uses_concrete_vjp_budget_per_binding():
    batch = SymbolicDim("B")
    builder = GraphBuilder("adaptive-dynamic-vjp")
    x = builder.input((batch, 4), DType.FLOAT64)
    module = builder.finish(x * x)

    small_vjp = vector_jacobian_product_module(
        specialize_module(module, {batch: 1}),
        wrt=(0,),
    )
    large_vjp = vector_jacobian_product_module(
        specialize_module(module, {batch: 4}),
        wrt=(0,),
    )
    small_bytes = analyze_module(small_vjp).planned_owning_storage_bytes
    large_bytes = analyze_module(large_vjp).planned_owning_storage_bytes
    assert small_bytes < large_bytes

    executable = compile_adaptive_dynamic_vjp_module(
        module,
        budget=CompileBudget(max_planned_storage_bytes=small_bytes),
        wrt=(0,),
    )

    small_x = np.array([[1.0, -2.0, 3.0, -4.0]], dtype=np.float64)
    small_cotangent = np.array([[2.0, 0.5, -3.0, 4.0]], dtype=np.float64)
    np.testing.assert_allclose(
        executable(inputs=(small_x, small_cotangent)),
        2.0 * small_x * small_cotangent,
        rtol=0.0,
        atol=0.0,
    )

    large_x = np.arange(16, dtype=np.float64).reshape(4, 4) - 5.0
    large_cotangent = (
        np.arange(16, dtype=np.float64).reshape(4, 4) * 0.25 + 1.0
    )
    np.testing.assert_allclose(
        executable(inputs=(large_x, large_cotangent)),
        2.0 * large_x * large_cotangent,
        rtol=0.0,
        atol=0.0,
    )

    small_specialization = executable.specialize({batch: 1})
    large_specialization = executable.specialize({batch: 4})
    assert small_specialization.backend == "native"
    assert small_specialization.budget_exceeded is None
    assert large_specialization.backend == "loop"
    assert large_specialization.budget_exceeded is not None
    assert large_specialization.budget_exceeded.metric == "planned_owning_storage_bytes"
    assert large_specialization.budget_exceeded.limit == small_bytes
    assert large_specialization.budget_exceeded.actual == large_bytes
    assert executable.cached_binding_backends == (
        ((("B", 1),), "native"),
        ((("B", 4),), "loop"),
    )

    assert executable.specialize({batch: 1}) is small_specialization
    assert executable.specialize({batch: 4}) is large_specialization


def test_adaptive_dynamic_vjp_requires_explicit_budget():
    batch = SymbolicDim("B")
    builder = GraphBuilder("adaptive-dynamic-vjp-budget")
    x = builder.input((batch, 2), DType.FLOAT64)
    module = builder.finish(x * x)

    with pytest.raises(TypeError, match="budget must be a CompileBudget"):
        compile_adaptive_dynamic_vjp_module(  # type: ignore[arg-type]
            module,
            budget=None,
        )
