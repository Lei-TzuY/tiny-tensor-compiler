from __future__ import annotations

import numpy as np

from tiny_tensor_compiler import (
    CompileBudget,
    GraphBuilder,
    SymbolicDim,
    compile_adaptive_dynamic_linearization,
    compile_dynamic_linearization,
)
from tiny_tensor_compiler.analysis import analyze_module
from tiny_tensor_compiler.autodiff import _shared_linearization_modules
from tiny_tensor_compiler.ir import DType
from tiny_tensor_compiler.symbolic import specialize_module


def test_dynamic_linearization_specializes_multi_symbol_state_and_reuses_cache():
    batch = SymbolicDim("B")
    width = SymbolicDim("W")
    builder = GraphBuilder("dynamic-linearization")
    x = builder.input((batch, width), DType.FLOAT64)
    y = builder.input((1, width), DType.FLOAT64)
    hidden = x * y + x
    module = builder.finish((hidden * hidden).sum(axis=0))

    executable = compile_dynamic_linearization(module, wrt=(0, 1))
    assert executable.symbolic_dims == (batch, width)

    x_value = np.array(
        [[1.0, -2.0, 3.0], [4.0, 0.5, -1.0]],
        dtype=np.float64,
    )
    y_value = np.array([[2.0, -3.0, 0.25]], dtype=np.float64)
    frozen_x = np.array(x_value, copy=True)
    frozen_y = np.array(y_value, copy=True)
    hidden_value = frozen_x * frozen_y + frozen_x
    expected_primal = np.sum(hidden_value * hidden_value, axis=0)

    concrete = executable.specialize({batch: 2, width: 3})
    assert concrete is executable.specialize({"B": 2, "W": 3})
    state = executable.linearize((x_value, y_value))
    np.testing.assert_allclose(state.primal, expected_primal, rtol=0.0, atol=0.0)

    x_value[...] = 1000.0
    y_value[...] = -1000.0

    x_tangent = np.array(
        [[0.5, 2.0, -1.0], [3.0, -4.0, 0.75]],
        dtype=np.float64,
    )
    y_tangent = np.array([[1.5, -0.25, 2.0]], dtype=np.float64)
    hidden_tangent = (
        x_tangent * frozen_y
        + frozen_x * y_tangent
        + x_tangent
    )
    expected_pushforward = np.sum(
        2.0 * hidden_value * hidden_tangent,
        axis=0,
    )
    np.testing.assert_allclose(
        state.pushforward((x_tangent, y_tangent)),
        expected_pushforward,
        rtol=0.0,
        atol=0.0,
    )

    cotangent = np.array([0.5, -2.0, 3.0], dtype=np.float64)
    hidden_cotangent = 2.0 * hidden_value * cotangent
    expected_x = hidden_cotangent * (frozen_y + 1.0)
    expected_y = np.sum(hidden_cotangent * frozen_x, axis=0, keepdims=True)
    actual_pullback = state.pullback(cotangent)
    assert isinstance(actual_pullback, tuple)
    np.testing.assert_allclose(actual_pullback[0], expected_x, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual_pullback[1], expected_y, rtol=0.0, atol=0.0)

    other_x = np.arange(8, dtype=np.float64).reshape(4, 2) - 3.0
    other_y = np.array([[0.5, -1.5]], dtype=np.float64)
    other_state = executable.linearize((other_x, other_y))
    expected_other_hidden = other_x * other_y + other_x
    np.testing.assert_allclose(
        other_state.primal,
        np.sum(expected_other_hidden * expected_other_hidden, axis=0),
        rtol=0.0,
        atol=0.0,
    )

    assert executable.cached_bindings == (
        (("B", 2), ("W", 3)),
        (("B", 4), ("W", 2)),
    )


def test_dynamic_linearization_specializes_retained_inplace_generation():
    batch = SymbolicDim("B")
    builder = GraphBuilder("dynamic-linearization-inplace")
    base = builder.input((batch, 4), DType.FLOAT64)
    factor = builder.input((batch, 4), DType.FLOAT64)
    root = base + builder.tensor(0.0, dtype=DType.FLOAT64)
    source = factor * factor
    module = builder.finish(root.binary_inplace(source, operator="mul"))

    executable = compile_dynamic_linearization(module, wrt=(0, 1))

    base_value = np.array(
        [[1.0, -2.0, 3.0, -4.0], [5.0, 6.0, -7.0, 8.0]],
        dtype=np.float64,
    )
    factor_value = np.array(
        [[0.5, -1.5, 2.0, 3.0], [-2.0, 0.25, 1.5, -0.5]],
        dtype=np.float64,
    )
    frozen_base = np.array(base_value, copy=True)
    frozen_factor = np.array(factor_value, copy=True)
    frozen_source = frozen_factor * frozen_factor

    concrete = executable.specialize({batch: 2})
    assert concrete.tape_value_count == 2
    state = executable.linearize((base_value, factor_value))
    np.testing.assert_allclose(
        state.primal,
        frozen_base * frozen_source,
        rtol=0.0,
        atol=0.0,
    )

    base_value[...] = 1000.0
    factor_value[...] = -1000.0

    base_tangent = np.array(
        [[0.25, -0.5, 0.75, -1.0], [1.25, -1.5, 1.75, -2.0]],
        dtype=np.float64,
    )
    factor_tangent = np.array(
        [[1.5, -2.0, 0.5, 0.25], [-0.75, 1.0, -1.25, 2.0]],
        dtype=np.float64,
    )
    expected_pushforward = (
        base_tangent * frozen_source
        + frozen_base * (2.0 * frozen_factor * factor_tangent)
    )
    np.testing.assert_allclose(
        state.pushforward((base_tangent, factor_tangent)),
        expected_pushforward,
        rtol=0.0,
        atol=0.0,
    )

    cotangent = np.array(
        [[2.0, -3.0, 4.0, 0.25], [-1.0, 0.5, 3.0, -2.0]],
        dtype=np.float64,
    )
    expected_base = cotangent * frozen_source
    expected_factor = cotangent * frozen_base * (2.0 * frozen_factor)
    actual_pullback = state.pullback(cotangent)
    assert isinstance(actual_pullback, tuple)
    np.testing.assert_allclose(actual_pullback[0], expected_base, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual_pullback[1], expected_factor, rtol=0.0, atol=0.0)

    assert executable.cached_batch_sizes == (2,)



def _linearization_peak_storage_bytes(module, bindings, *, wrt):
    concrete = specialize_module(module, bindings)
    primal, pushforward, pullback, _ = _shared_linearization_modules(
        concrete,
        wrt=wrt,
    )
    return max(
        analyze_module(component).planned_owning_storage_bytes
        for component in (primal, pushforward, pullback)
    )


def test_adaptive_dynamic_linearization_uses_one_backend_per_retained_bundle():
    batch = SymbolicDim("B")
    builder = GraphBuilder("adaptive-dynamic-linearization")
    base = builder.input((batch, 4), DType.FLOAT64)
    factor = builder.input((batch, 4), DType.FLOAT64)
    root = base + builder.tensor(0.0, dtype=DType.FLOAT64)
    source = factor * factor
    module = builder.finish(root.binary_inplace(source, operator="mul"))

    small_peak = _linearization_peak_storage_bytes(
        module,
        {batch: 1},
        wrt=(0, 1),
    )
    large_peak = _linearization_peak_storage_bytes(
        module,
        {batch: 8},
        wrt=(0, 1),
    )
    assert small_peak < large_peak

    executable = compile_adaptive_dynamic_linearization(
        module,
        budget=CompileBudget(max_planned_storage_bytes=small_peak),
        wrt=(0, 1),
    )

    small = executable.specialize({batch: 1})
    assert small is executable.specialize({"B": 1})
    assert small.backend == "native"
    assert small.budget_exceeded is None

    large = executable.specialize({batch: 8})
    assert large is executable.specialize({"B": 8})
    assert large.backend == "loop"
    assert large.budget_exceeded is not None
    assert large.budget_exceeded_component in {
        "primal_tape",
        "pushforward",
        "pullback",
    }

    assert executable.cached_binding_backends == (
        ((("B", 1),), "native"),
        ((("B", 8),), "loop"),
    )

    for size, expected_backend in ((1, "native"), (8, "loop")):
        base_value = (
            np.arange(size * 4, dtype=np.float64).reshape(size, 4) - 3.0
        )
        factor_value = (
            np.arange(size * 4, dtype=np.float64).reshape(size, 4) * 0.25
            + 0.5
        )
        frozen_base = np.array(base_value, copy=True)
        frozen_factor = np.array(factor_value, copy=True)
        frozen_source = frozen_factor * frozen_factor

        state = executable.linearize((base_value, factor_value))
        assert executable.specialize({batch: size}).backend == expected_backend
        np.testing.assert_allclose(
            state.primal,
            frozen_base * frozen_source,
            rtol=0.0,
            atol=0.0,
        )

        base_value[...] = 1000.0
        factor_value[...] = -1000.0

        base_tangent = np.full((size, 4), 0.75, dtype=np.float64)
        factor_tangent = np.full((size, 4), -0.5, dtype=np.float64)
        expected_pushforward = (
            base_tangent * frozen_source
            + frozen_base * (2.0 * frozen_factor * factor_tangent)
        )
        np.testing.assert_allclose(
            state.pushforward((base_tangent, factor_tangent)),
            expected_pushforward,
            rtol=0.0,
            atol=0.0,
        )

        cotangent = np.full((size, 4), 1.25, dtype=np.float64)
        actual_pullback = state.pullback(cotangent)
        assert isinstance(actual_pullback, tuple)
        np.testing.assert_allclose(
            actual_pullback[0],
            cotangent * frozen_source,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            actual_pullback[1],
            cotangent * frozen_base * (2.0 * frozen_factor),
            rtol=0.0,
            atol=0.0,
        )
