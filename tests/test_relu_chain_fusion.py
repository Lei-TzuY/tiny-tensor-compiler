import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    execute_loop,
    execute_native,
    execute_reference,
    fuse_elementwise,
    lower_to_cpu,
    lower_to_loops,
)


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


@pytest.mark.parametrize("opcode", ["add", "mul"])
def test_elementwise_fusion_absorbs_repeated_relu_tail(opcode):
    builder = GraphBuilder()
    lhs = builder.input((2, 1), dtype="float32")
    rhs = builder.input((1, 3), dtype="float32")
    binary = lhs + rhs if opcode == "add" else lhs * rhs
    module = builder.finish(binary.relu().relu().relu())

    original = lower_to_loops(lower_to_cpu(module))
    fused = fuse_elementwise(original)

    assert [kernel.opcode for kernel in original.kernels] == [opcode, "relu", "relu", "relu"]
    assert [kernel.opcode for kernel in fused.kernels] == [f"relu_{opcode}"]
    assert fused.kernels[0].input_maps == original.kernels[0].input_maps

    inputs = [
        np.array([[-3.0], [2.0]], dtype=np.float32),
        np.array([[1.0, 2.0, 4.0]], dtype=np.float32),
    ]
    np.testing.assert_array_equal(
        execute_loop(fused, inputs=inputs),
        execute_reference(module, inputs=inputs),
    )


def test_elementwise_fusion_collapses_pure_relu_chain_when_alias_safe():
    builder = GraphBuilder()
    value = builder.input((3,), dtype="float32")
    first = value.relu()
    second = first.relu()
    module = builder.finish(second + value)

    original = lower_to_loops(lower_to_cpu(module))
    fused = fuse_elementwise(original)

    assert [kernel.opcode for kernel in original.kernels] == ["relu", "relu", "add"]
    assert [kernel.opcode for kernel in fused.kernels] == ["relu", "add"]

    inputs = [np.array([-3.0, -0.0, 2.0], dtype=np.float32)]
    np.testing.assert_array_equal(
        execute_loop(fused, inputs=inputs),
        execute_reference(module, inputs=inputs),
    )


def test_elementwise_fusion_stops_when_relu_result_has_another_live_use():
    builder = GraphBuilder()
    lhs = builder.input((2, 1), dtype="float32")
    rhs = builder.input((1, 3), dtype="float32")
    shared = (lhs + rhs).relu()
    module = builder.finish(shared.relu() + shared)

    original = lower_to_loops(lower_to_cpu(module))
    fused = fuse_elementwise(original)

    assert [kernel.opcode for kernel in fused.kernels] == ["relu_add", "relu", "add"]

    inputs = [
        np.array([[-3.0], [2.0]], dtype=np.float32),
        np.array([[1.0, 2.0, 4.0]], dtype=np.float32),
    ]
    np.testing.assert_array_equal(
        execute_loop(fused, inputs=inputs),
        execute_reference(module, inputs=inputs),
    )


def test_elementwise_fusion_is_idempotent_after_relu_tail_absorption():
    builder = GraphBuilder()
    lhs = builder.input((2, 1), dtype="float32")
    rhs = builder.input((1, 3), dtype="float32")
    module = builder.finish((lhs + rhs).relu().relu().relu())

    once = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))
    twice = fuse_elementwise(once)

    assert twice.dump() == once.dump()


def test_repeated_relu_tail_preserves_float_edge_semantics_in_native_execution():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    lhs = builder.input((2, 1), dtype="float64")
    rhs = builder.input((1, 3), dtype="float64")
    module = builder.finish((lhs + rhs).relu().relu().relu())
    fused = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))

    lhs_value = np.array([[np.nan], [-0.0]], dtype=np.float64)
    rhs_value = np.array([[0.0, -np.inf, 1.0]], dtype=np.float64)
    inputs = [lhs_value, rhs_value]
    expected = execute_reference(module, inputs=inputs)
    actual = execute_native(fused, inputs=inputs)

    np.testing.assert_array_equal(actual, expected)
    assert np.signbit(actual[1, 0]) == np.signbit(expected[1, 0])
