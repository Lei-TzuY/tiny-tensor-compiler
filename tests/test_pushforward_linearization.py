from __future__ import annotations

import numpy as np

from tiny_tensor_compiler import (
    GraphBuilder,
    compile_pushforward_linearization,
)
from tiny_tensor_compiler.ir import DType


def test_reusable_pushforward_freezes_primal_and_reuses_intermediate_tape():
    builder = GraphBuilder("reusable-pushforward")
    x = builder.input((2, 3), DType.FLOAT64)
    y = builder.input((1, 3), DType.FLOAT64)
    hidden = x * y + x
    module = builder.finish((hidden * hidden).sum(axis=0))

    executable = compile_pushforward_linearization(module, wrt=(0, 1))
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

    # Mutating caller-owned inputs after linearization must not change the retained primal.
    x_value[...] = 1000.0
    y_value[...] = -1000.0

    queries = (
        (
            np.array(
                [[0.5, 2.0, -1.0], [3.0, -4.0, 0.75]],
                dtype=np.float64,
            ),
            np.array([[1.5, -0.25, 2.0]], dtype=np.float64),
        ),
        (
            np.array(
                [[-2.0, 0.25, 3.0], [1.0, 1.5, -0.5]],
                dtype=np.float64,
            ),
            np.array([[0.0, 2.0, -1.0]], dtype=np.float64),
        ),
    )

    for x_tangent, y_tangent in queries:
        hidden_tangent = (
            x_tangent * frozen_y
            + frozen_x * y_tangent
            + x_tangent
        )
        expected = np.sum(
            2.0 * hidden_value * hidden_tangent,
            axis=0,
        )
        actual = state.pushforward((x_tangent, y_tangent))
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    assert state.query_count == 2


def test_reusable_pushforward_replays_direct_copy_tangent_generation():
    builder = GraphBuilder("reusable-pushforward-write")
    base = builder.input((6,), DType.FLOAT64)
    patch = builder.input((3,), DType.FLOAT64)
    root = base + builder.tensor(0.0, dtype=DType.FLOAT64)
    target = root.slice(axis=0, start=1, stop=6, step=2)
    module = builder.finish(root.copy_into(target, patch))

    executable = compile_pushforward_linearization(module, wrt=(0, 1))
    state = executable.linearize(
        (
            np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=np.float64),
            np.array([7.0, 8.0, 9.0], dtype=np.float64),
        )
    )

    base_tangent = np.array([2.0, 3.0, -4.0, 0.25, 7.0, -5.0], dtype=np.float64)
    patch_tangent = np.array([11.0, -13.0, 17.0], dtype=np.float64)
    expected = np.array(base_tangent, copy=True)
    expected[1:6:2] = patch_tangent

    np.testing.assert_allclose(
        state.pushforward((base_tangent, patch_tangent)),
        expected,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        state.pushforward((-base_tangent, 2.0 * patch_tangent)),
        np.array(
            [-2.0, 22.0, 4.0, -26.0, -7.0, 34.0],
            dtype=np.float64,
        ),
        rtol=0.0,
        atol=0.0,
    )
    assert state.query_count == 2


def test_reusable_pushforward_replays_binary_into_add_generation():
    builder = GraphBuilder("reusable-pushforward-binary-add")
    base = builder.input((6,), DType.FLOAT64)
    source = builder.input((3,), DType.FLOAT64)
    root = base + builder.tensor(0.0, dtype=DType.FLOAT64)
    target = root.slice(axis=0, start=1, stop=6, step=2)
    module = builder.finish(root.binary_into(target, source, operator="add"))

    executable = compile_pushforward_linearization(module, wrt=(0, 1))
    base_value = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=np.float64)
    source_value = np.array([7.0, -8.0, 9.0], dtype=np.float64)
    state = executable.linearize((base_value, source_value))

    base_value[...] = 1000.0
    source_value[...] = -1000.0

    base_tangent = np.array([0.5, -1.0, 1.5, -2.0, 2.5, -3.0], dtype=np.float64)
    source_tangent = np.array([4.0, -5.0, 6.0], dtype=np.float64)
    expected = np.array(base_tangent, copy=True)
    expected[1:6:2] += source_tangent
    np.testing.assert_allclose(
        state.pushforward((base_tangent, source_tangent)),
        expected,
        rtol=0.0,
        atol=0.0,
    )

    expected_second = -np.array(base_tangent, copy=True)
    expected_second[1:6:2] += 2.0 * source_tangent
    np.testing.assert_allclose(
        state.pushforward((-base_tangent, 2.0 * source_tangent)),
        expected_second,
        rtol=0.0,
        atol=0.0,
    )
    assert state.query_count == 2


def test_reusable_pushforward_replays_binary_inplace_add_generation():
    builder = GraphBuilder("reusable-pushforward-binary-inplace-add")
    base = builder.input((4,), DType.FLOAT64)
    source = builder.input((4,), DType.FLOAT64)
    root = base + builder.tensor(0.0, dtype=DType.FLOAT64)
    module = builder.finish(root.binary_inplace(source, operator="add"))

    executable = compile_pushforward_linearization(module, wrt=(0, 1))
    base_value = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float64)
    source_value = np.array([5.0, 6.0, -7.0, 8.0], dtype=np.float64)
    state = executable.linearize((base_value, source_value))

    base_value[...] = 1000.0
    source_value[...] = -1000.0

    base_tangent = np.array([0.5, -1.0, 1.5, -2.0], dtype=np.float64)
    source_tangent = np.array([4.0, -5.0, 6.0, -7.0], dtype=np.float64)
    expected = base_tangent + source_tangent
    np.testing.assert_allclose(
        state.pushforward((base_tangent, source_tangent)),
        expected,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        state.pushforward((-base_tangent, 2.0 * source_tangent)),
        -base_tangent + 2.0 * source_tangent,
        rtol=0.0,
        atol=0.0,
    )
    assert state.query_count == 2
