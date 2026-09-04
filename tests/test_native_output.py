import os
import shutil

import numpy as np
import pytest

import tiny_tensor_compiler.native as native_module
from tiny_tensor_compiler import (
    GraphBuilder,
    compile_native,
    execute_native,
    lower_to_cpu,
    lower_to_loops,
)


@pytest.fixture(autouse=True)
def _clear_native_artifact_cache():
    native_module.clear_native_cache()
    yield
    native_module.clear_native_cache()


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _input_program(shape=(2, 3), dtype="int32"):
    builder = GraphBuilder()
    value = builder.input(shape, dtype=dtype)
    module = builder.finish((value + 1).relu())
    return lower_to_loops(lower_to_cpu(module))


def test_execute_native_writes_preallocated_output_and_returns_same_array():
    _default_compiler_or_skip()
    program = _input_program()
    runtime_input = np.array([[-2, 0, 4], [7, -9, 3]], dtype=np.int32)
    output = np.empty((2, 3), dtype=np.int32)

    result = execute_native(program, inputs=[runtime_input], out=output)

    assert result is output
    np.testing.assert_array_equal(output, np.maximum(runtime_input + np.int32(1), 0))


def test_reusable_executable_reuses_compile_with_preallocated_outputs(monkeypatch):
    _default_compiler_or_skip()
    program = _input_program()
    compile_calls = 0
    original_run = native_module.subprocess.run

    def counting_run(*args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return original_run(*args, **kwargs)

    monkeypatch.setattr(native_module.subprocess, "run", counting_run)
    executable = compile_native(program)

    first_input = np.arange(6, dtype=np.int32).reshape(2, 3)
    second_input = -first_input
    first_output = np.empty((2, 3), dtype=np.int32)
    second_output = np.empty((2, 3), dtype=np.int32)

    first = executable(inputs=[first_input], out=first_output)
    second = executable.execute(inputs=[second_input], out=second_output)

    assert compile_calls == 1
    assert first is first_output
    assert second is second_output
    np.testing.assert_array_equal(first, np.maximum(first_input + np.int32(1), 0))
    np.testing.assert_array_equal(second, np.maximum(second_input + np.int32(1), 0))


def test_preallocated_output_rejects_wrong_shape_and_dtype():
    _default_compiler_or_skip()
    program = _input_program()
    runtime_input = np.zeros((2, 3), dtype=np.int32)

    with pytest.raises(ValueError, match="output shape"):
        execute_native(
            program,
            inputs=[runtime_input],
            out=np.empty((3, 2), dtype=np.int32),
        )

    with pytest.raises(ValueError, match="output dtype"):
        execute_native(
            program,
            inputs=[runtime_input],
            out=np.empty((2, 3), dtype=np.int64),
        )


def test_preallocated_output_requires_ndarray_contiguous_writable_and_aligned():
    _default_compiler_or_skip()
    program = _input_program()
    runtime_input = np.zeros((2, 3), dtype=np.int32)

    with pytest.raises(TypeError, match="output must be a numpy.ndarray"):
        execute_native(program, inputs=[runtime_input], out=[[0, 0, 0], [0, 0, 0]])

    noncontiguous = np.empty((3, 2), dtype=np.int32).T
    assert not noncontiguous.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        execute_native(program, inputs=[runtime_input], out=noncontiguous)

    readonly = np.empty((2, 3), dtype=np.int32)
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        execute_native(program, inputs=[runtime_input], out=readonly)

    storage = bytearray(1 + 6 * np.dtype(np.int32).itemsize)
    unaligned = np.ndarray((2, 3), dtype=np.int32, buffer=storage, offset=1)
    assert not unaligned.flags.aligned
    with pytest.raises(ValueError, match="aligned"):
        execute_native(program, inputs=[runtime_input], out=unaligned)


def test_preallocated_output_rejects_overlap_with_runtime_input():
    _default_compiler_or_skip()
    program = _input_program()
    shared = np.arange(6, dtype=np.int32).reshape(2, 3)

    with pytest.raises(ValueError, match="overlap runtime input 0"):
        execute_native(program, inputs=[shared], out=shared)


def test_preallocated_output_supports_scalar_and_zero_extent_results():
    _default_compiler_or_skip()

    scalar_builder = GraphBuilder()
    scalar_input = scalar_builder.input((), dtype="float32")
    scalar_program = lower_to_loops(lower_to_cpu(scalar_builder.finish(scalar_input.relu())))
    scalar_out = np.empty((), dtype=np.float32)
    scalar_result = execute_native(
        scalar_program,
        inputs=[np.array(-0.0, dtype=np.float32)],
        out=scalar_out,
    )
    assert scalar_result is scalar_out
    assert scalar_result.shape == ()
    assert not np.signbit(scalar_result).item()

    empty_builder = GraphBuilder()
    empty_input = empty_builder.input((0,), dtype="int32")
    empty_program = lower_to_loops(lower_to_cpu(empty_builder.finish(empty_input.relu())))
    empty_out = np.empty((0,), dtype=np.int32)
    empty_result = execute_native(
        empty_program,
        inputs=[np.empty((0,), dtype=np.int32)],
        out=empty_out,
    )
    assert empty_result is empty_out
    assert empty_result.shape == (0,)
    assert empty_result.dtype == np.dtype(np.int32)
