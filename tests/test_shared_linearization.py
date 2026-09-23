from __future__ import annotations

import numpy as np
from tiny_tensor_compiler import (
    GraphBuilder,
    compile_linearization,
)
from tiny_tensor_compiler.ir import DType


def test_shared_linearization_reuses_one_frozen_tape_for_pushforward_and_pullback():
    builder = GraphBuilder("shared-linearization")
    x = builder.input((2, 3), DType.FLOAT64)
    y = builder.input((1, 3), DType.FLOAT64)
    hidden = x * y + x
    module = builder.finish((hidden * hidden).sum(axis=0))

    executable = compile_linearization(module, wrt=(0, 1))
    assert executable.tape_value_count == 1

    x_value = np.array(
        [[1.0, -2.0, 3.0], [4.0, 0.5, -1.0]],
        dtype=np.float64,
    )
    y_value = np.array([[2.0, -3.0, 0.25]], dtype=np.float64)
    frozen_x = np.array(x_value, copy=True)
    frozen_y = np.array(y_value, copy=True)
    hidden_value = frozen_x * frozen_y + frozen_x
    expected_primal = np.sum(hidden_value * hidden_value, axis=0)

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
    actual_pushforward = state.pushforward((x_tangent, y_tangent))
    np.testing.assert_allclose(
        actual_pushforward,
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

    second_pushforward = state.pushforward((-x_tangent, 2.0 * y_tangent))
    second_hidden_tangent = (
        (-x_tangent) * frozen_y
        + frozen_x * (2.0 * y_tangent)
        - x_tangent
    )
    np.testing.assert_allclose(
        second_pushforward,
        np.sum(2.0 * hidden_value * second_hidden_tangent, axis=0),
        rtol=0.0,
        atol=0.0,
    )

    second_pullback = state.pullback(-cotangent)
    np.testing.assert_allclose(second_pullback[0], -expected_x, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(second_pullback[1], -expected_y, rtol=0.0, atol=0.0)

    assert state.pushforward_query_count == 2
    assert state.pullback_query_count == 2


def test_shared_linearization_retains_copy_generation_for_repeated_queries():
    builder = GraphBuilder("shared-linearization-write")
    base = builder.input((6,), DType.FLOAT64)
    patch = builder.input((3,), DType.FLOAT64)
    root = base + builder.tensor(0.0, dtype=DType.FLOAT64)
    target = root.slice(axis=0, start=1, stop=6, step=2)
    module = builder.finish(root.copy_into(target, patch))

    executable = compile_linearization(module, wrt=(0, 1))

    base_value = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=np.float64)
    patch_value = np.array([7.0, 8.0, 9.0], dtype=np.float64)
    frozen_base = np.array(base_value, copy=True)
    frozen_patch = np.array(patch_value, copy=True)

    expected_primal = np.array(frozen_base, copy=True)
    expected_primal[1:6:2] = frozen_patch

    state = executable.linearize((base_value, patch_value))
    np.testing.assert_allclose(state.primal, expected_primal, rtol=0.0, atol=0.0)

    base_value[...] = 1000.0
    patch_value[...] = -1000.0

    base_tangent = np.array([2.0, 3.0, -4.0, 0.25, 7.0, -5.0], dtype=np.float64)
    patch_tangent = np.array([11.0, -13.0, 17.0], dtype=np.float64)
    expected_pushforward = np.array(base_tangent, copy=True)
    expected_pushforward[1:6:2] = patch_tangent

    actual_pushforward = state.pushforward((base_tangent, patch_tangent))
    np.testing.assert_allclose(
        actual_pushforward,
        expected_pushforward,
        rtol=0.0,
        atol=0.0,
    )

    cotangent = np.array([0.5, -2.0, 3.0, 4.0, -1.5, 6.0], dtype=np.float64)
    expected_base = np.array(cotangent, copy=True)
    expected_base[1:6:2] = 0.0
    expected_patch = cotangent[1:6:2]

    actual_pullback = state.pullback(cotangent)
    assert isinstance(actual_pullback, tuple)
    np.testing.assert_allclose(actual_pullback[0], expected_base, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual_pullback[1], expected_patch, rtol=0.0, atol=0.0)

    second_pushforward = state.pushforward((-base_tangent, 2.0 * patch_tangent))
    expected_second_pushforward = -np.array(base_tangent, copy=True)
    expected_second_pushforward[1:6:2] = 2.0 * patch_tangent
    np.testing.assert_allclose(
        second_pushforward,
        expected_second_pushforward,
        rtol=0.0,
        atol=0.0,
    )

    second_pullback = state.pullback(-cotangent)
    np.testing.assert_allclose(second_pullback[0], -expected_base, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(second_pullback[1], -expected_patch, rtol=0.0, atol=0.0)

    assert state.pushforward_query_count == 2
    assert state.pullback_query_count == 2
