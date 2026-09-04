import os
import shutil

import numpy as np
import pytest

import tiny_tensor_compiler.native as native_module
from tiny_tensor_compiler import (
    GraphBuilder,
    compile_module,
    compile_native,
    execute_loop,
    execute_native,
    execute_reference,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
)
from tiny_tensor_compiler.c_codegen import generate_c as generate_single_output_c


@pytest.fixture(autouse=True)
def _clear_native_artifact_cache():
    native_module.clear_native_cache()
    yield
    native_module.clear_native_cache()


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _mixed_output_module():
    builder = GraphBuilder()
    integers = builder.input((3,), dtype="int32")
    floats = builder.input((2,), dtype="float32")
    integer_output = (integers + 1).relu()
    float_output = (floats * 2).relu()
    return builder.finish((integer_output, float_output))


def _same_type_output_program():
    builder = GraphBuilder()
    value = builder.input((3,), dtype="int32")
    summed = value + 1
    return lower_to_loops(lower_to_cpu(builder.finish((summed, summed.relu()))))


def test_single_output_c_source_remains_byte_for_byte_compatible():
    builder = GraphBuilder()
    value = builder.input((3,), dtype="int32")
    loops = lower_to_loops(lower_to_cpu(builder.finish((value + 1).relu())))

    assert generate_c(loops) == generate_single_output_c(loops)


def test_native_multi_output_matches_reference_and_loop_with_mixed_types():
    _default_compiler_or_skip()
    module = _mixed_output_module()
    loops = lower_to_loops(lower_to_cpu(module))
    inputs = [
        np.array([-2, 0, 4], dtype=np.int32),
        np.array([-1.5, 3.0], dtype=np.float32),
    ]

    native = execute_native(loops, inputs=inputs)
    interpreted = execute_loop(loops, inputs=inputs)
    reference = execute_reference(module, inputs=inputs)

    assert isinstance(native, tuple)
    assert isinstance(interpreted, tuple)
    assert isinstance(reference, tuple)
    assert len(native) == 2
    for actual, expected_loop, expected_reference in zip(
        native, interpreted, reference, strict=True
    ):
        np.testing.assert_array_equal(actual, expected_loop)
        np.testing.assert_array_equal(actual, expected_reference)


def test_compile_module_executes_multi_output_native_pipeline():
    _default_compiler_or_skip()
    module = _mixed_output_module()
    inputs = [
        np.array([1, -5, 7], dtype=np.int32),
        np.array([2.5, -4.0], dtype=np.float32),
    ]

    executable = compile_module(module)
    actual = executable(inputs=inputs)
    expected = execute_reference(module, inputs=inputs)

    assert isinstance(actual, tuple)
    assert isinstance(expected, tuple)
    for actual_output, expected_output in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_output, expected_output)


def test_multi_output_native_writes_preallocated_arrays_and_reuses_executable():
    _default_compiler_or_skip()
    module = _mixed_output_module()
    loops = lower_to_loops(lower_to_cpu(module))
    executable = compile_native(loops)
    inputs = [
        np.array([-3, 2, 8], dtype=np.int32),
        np.array([-2.0, 5.0], dtype=np.float32),
    ]
    integer_out = np.empty((3,), dtype=np.int32)
    float_out = np.empty((2,), dtype=np.float32)

    result = executable(inputs=inputs, out=(integer_out, float_out))
    expected = execute_reference(module, inputs=inputs)

    assert isinstance(result, tuple)
    assert result[0] is integer_out
    assert result[1] is float_out
    assert isinstance(expected, tuple)
    np.testing.assert_array_equal(result[0], expected[0])
    np.testing.assert_array_equal(result[1], expected[1])


def test_multi_output_preallocation_validates_count_shape_dtype_and_container():
    program = _same_type_output_program()
    runtime_input = np.array([1, 2, 3], dtype=np.int32)

    with pytest.raises(TypeError, match="requires a sequence"):
        execute_native(program, inputs=[runtime_input], out=np.empty((3,), dtype=np.int32))

    with pytest.raises(ValueError, match="requires 2 output arrays, got 1"):
        execute_native(
            program,
            inputs=[runtime_input],
            out=(np.empty((3,), dtype=np.int32),),
        )

    with pytest.raises(ValueError, match="output 1 shape"):
        execute_native(
            program,
            inputs=[runtime_input],
            out=(
                np.empty((3,), dtype=np.int32),
                np.empty((2,), dtype=np.int32),
            ),
        )

    with pytest.raises(ValueError, match="output 1 dtype"):
        execute_native(
            program,
            inputs=[runtime_input],
            out=(
                np.empty((3,), dtype=np.int32),
                np.empty((3,), dtype=np.int64),
            ),
        )


def test_multi_output_preallocation_rejects_input_and_output_aliasing():
    program = _same_type_output_program()
    runtime_input = np.array([1, 2, 3], dtype=np.int32)

    with pytest.raises(ValueError, match="output 0 must not overlap runtime input 0"):
        execute_native(
            program,
            inputs=[runtime_input],
            out=(runtime_input, np.empty((3,), dtype=np.int32)),
        )

    shared = np.empty((3,), dtype=np.int32)
    with pytest.raises(ValueError, match="outputs 0 and 1 must not overlap"):
        execute_native(
            program,
            inputs=[runtime_input],
            out=(shared, shared),
        )


def test_multi_output_native_handles_scalar_and_zero_extent_returns():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    scalar = builder.input((), dtype="float32")
    empty = builder.input((0,), dtype="int32")
    module = builder.finish((scalar.relu(), empty.relu()))
    loops = lower_to_loops(lower_to_cpu(module))

    result = execute_native(
        loops,
        inputs=[
            np.array(-0.0, dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        ],
    )

    assert isinstance(result, tuple)
    assert result[0].shape == ()
    assert not np.signbit(result[0]).item()
    assert result[1].shape == (0,)
    assert result[1].dtype == np.dtype(np.int32)
