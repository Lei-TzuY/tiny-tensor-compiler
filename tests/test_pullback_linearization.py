from __future__ import annotations

import numpy as np

from tiny_tensor_compiler import (
    GraphBuilder,
    compile_pullback_linearization,
)
from tiny_tensor_compiler.ir import DType


def test_reusable_pullback_freezes_primal_and_reuses_reverse_tape():
    builder = GraphBuilder("reusable-pullback")
    x = builder.input((2, 3), DType.FLOAT64)
    y = builder.input((1, 3), DType.FLOAT64)
    hidden = x * y + x
    module = builder.finish((hidden * hidden).sum(axis=0))

    executable = compile_pullback_linearization(module, wrt=(0, 1))
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

    cotangents = (
        np.array([0.5, -2.0, 3.0], dtype=np.float64),
        np.array([-1.5, 0.25, 2.0], dtype=np.float64),
    )
    for cotangent in cotangents:
        hidden_cotangent = 2.0 * hidden_value * cotangent
        expected_x = hidden_cotangent * (frozen_y + 1.0)
        expected_y = np.sum(hidden_cotangent * frozen_x, axis=0, keepdims=True)

        actual = state.pullback(cotangent)
        assert isinstance(actual, tuple)
        np.testing.assert_allclose(actual[0], expected_x, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(actual[1], expected_y, rtol=0.0, atol=0.0)

    assert state.query_count == 2


def test_reusable_pullback_replays_direct_copy_cotangent_generation():
    builder = GraphBuilder("reusable-pullback-write")
    base = builder.input((6,), DType.FLOAT64)
    patch = builder.input((3,), DType.FLOAT64)
    root = base + builder.tensor(0.0, dtype=DType.FLOAT64)
    target = root.slice(axis=0, start=1, stop=6, step=2)
    module = builder.finish(root.copy_into(target, patch))

    executable = compile_pullback_linearization(module, wrt=(0, 1))
    state = executable.linearize(
        (
            np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=np.float64),
            np.array([7.0, 8.0, 9.0], dtype=np.float64),
        )
    )

    cotangent = np.array([0.5, -2.0, 3.0, 4.0, -1.5, 6.0], dtype=np.float64)
    expected_base = np.array(cotangent, copy=True)
    expected_base[1:6:2] = 0.0
    expected_patch = cotangent[1:6:2]

    actual = state.pullback(cotangent)
    assert isinstance(actual, tuple)
    np.testing.assert_allclose(actual[0], expected_base, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual[1], expected_patch, rtol=0.0, atol=0.0)

    second = state.pullback(-cotangent)
    assert isinstance(second, tuple)
    np.testing.assert_allclose(second[0], -expected_base, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(second[1], -expected_patch, rtol=0.0, atol=0.0)
    assert state.query_count == 2


def test_reusable_pullback_replays_binary_into_add_generation():
    builder = GraphBuilder("reusable-pullback-binary-add")
    base = builder.input((6,), DType.FLOAT64)
    source = builder.input((3,), DType.FLOAT64)
    root = base + builder.tensor(0.0, dtype=DType.FLOAT64)
    target = root.slice(axis=0, start=1, stop=6, step=2)
    module = builder.finish(root.binary_into(target, source, operator="add"))

    executable = compile_pullback_linearization(module, wrt=(0, 1))
    base_value = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=np.float64)
    source_value = np.array([7.0, -8.0, 9.0], dtype=np.float64)
    state = executable.linearize((base_value, source_value))

    base_value[...] = 1000.0
    source_value[...] = -1000.0

    cotangent = np.array([0.5, -2.0, 3.0, 4.0, -1.5, 6.0], dtype=np.float64)
    expected_source = cotangent[1:6:2]
    actual = state.pullback(cotangent)
    assert isinstance(actual, tuple)
    np.testing.assert_allclose(actual[0], cotangent, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual[1], expected_source, rtol=0.0, atol=0.0)

    second = state.pullback(-cotangent)
    assert isinstance(second, tuple)
    np.testing.assert_allclose(second[0], -cotangent, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(second[1], -expected_source, rtol=0.0, atol=0.0)
    assert state.query_count == 2


def test_reusable_pullback_replays_binary_inplace_add_generation():
    builder = GraphBuilder("reusable-pullback-binary-inplace-add")
    base = builder.input((4,), DType.FLOAT64)
    source = builder.input((4,), DType.FLOAT64)
    root = base + builder.tensor(0.0, dtype=DType.FLOAT64)
    module = builder.finish(root.binary_inplace(source, operator="add"))

    executable = compile_pullback_linearization(module, wrt=(0, 1))
    base_value = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float64)
    source_value = np.array([5.0, 6.0, -7.0, 8.0], dtype=np.float64)
    state = executable.linearize((base_value, source_value))

    base_value[...] = 1000.0
    source_value[...] = -1000.0

    cotangent = np.array([0.5, -2.0, 3.0, 4.0], dtype=np.float64)
    actual = state.pullback(cotangent)
    assert isinstance(actual, tuple)
    np.testing.assert_allclose(actual[0], cotangent, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual[1], cotangent, rtol=0.0, atol=0.0)

    second = state.pullback(-cotangent)
    assert isinstance(second, tuple)
    np.testing.assert_allclose(second[0], -cotangent, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(second[1], -cotangent, rtol=0.0, atol=0.0)
    assert state.query_count == 2
