import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    NativeCompilationError,
    execute_loop,
    execute_native,
    execute_reference,
    lower_to_cpu,
    lower_to_loops,
)


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _native_for(module):
    _default_compiler_or_skip()
    loops = lower_to_loops(lower_to_cpu(module))
    return execute_native(loops), execute_loop(loops)


@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_native_execution_matches_reference_and_loop_for_broadcast_relu(dtype):
    builder = GraphBuilder()
    lhs = builder.tensor([[-3], [2]], dtype=dtype)
    rhs = builder.tensor([[1, 2, 4]], dtype=dtype)
    module = builder.finish((lhs + rhs).relu())

    native, loop = _native_for(module)
    reference = execute_reference(module)

    np.testing.assert_array_equal(native, loop)
    np.testing.assert_array_equal(native, reference)
    assert native.dtype == reference.dtype
    assert native.shape == reference.shape


def test_native_integer_overflow_matches_numpy_wrap_semantics():
    builder = GraphBuilder()
    maximum = builder.tensor([2_147_483_647], dtype="int32")
    one = builder.tensor([1], dtype="int32")
    module = builder.finish(maximum + one)

    native, loop = _native_for(module)

    np.testing.assert_array_equal(native, loop)
    np.testing.assert_array_equal(native, execute_reference(module))
    np.testing.assert_array_equal(native, np.array([-2_147_483_648], dtype=np.int32))


def test_native_float_relu_matches_numpy_signed_zero_semantics():
    builder = GraphBuilder()
    value = builder.tensor(-0.0, dtype="float32")
    module = builder.finish(value.relu())

    native, loop = _native_for(module)

    np.testing.assert_array_equal(native, loop)
    assert native.shape == ()
    assert not np.signbit(native).item()


def test_native_execution_preserves_zero_extent_result():
    builder = GraphBuilder()
    value = builder.tensor([], dtype="int32")
    module = builder.finish(value.relu())

    native, loop = _native_for(module)

    np.testing.assert_array_equal(native, loop)
    assert native.shape == (0,)
    assert native.dtype == np.dtype(np.int32)


def test_native_execution_reports_missing_compiler():
    builder = GraphBuilder()
    module = builder.finish(builder.tensor([1, 2, 3], dtype="int32"))
    loops = lower_to_loops(lower_to_cpu(module))

    with pytest.raises(NativeCompilationError, match="compiler executable not found"):
        execute_native(loops, compiler="tiny-tensor-compiler-missing-cc")
