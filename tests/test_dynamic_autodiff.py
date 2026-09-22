from __future__ import annotations

import numpy as np
import pytest

from tiny_tensor_compiler import (
    AutodiffError,
    GraphBuilder,
    SymbolicDim,
    compile_dynamic_gradient_module,
)
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
