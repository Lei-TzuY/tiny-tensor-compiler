from __future__ import annotations

import numpy as np
import pytest

from tiny_tensor_compiler import (
    AutodiffError,
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


def test_reusable_pushforward_rejects_writable_effects_until_generation_tape_is_first_class():
    builder = GraphBuilder("reusable-pushforward-write")
    base = builder.input((6,), DType.FLOAT64)
    patch = builder.input((3,), DType.FLOAT64)
    root = base + builder.tensor(0.0, dtype=DType.FLOAT64)
    target = root.slice(axis=0, start=1, stop=6, step=2)
    module = builder.finish(root.copy_into(target, patch))

    with pytest.raises(
        AutodiffError,
        match="reusable pushforward linearization.*write effects",
    ):
        compile_pushforward_linearization(module, wrt=(0, 1))
