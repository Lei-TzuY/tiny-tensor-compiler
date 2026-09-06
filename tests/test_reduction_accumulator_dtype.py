import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    compile_module,
    execute_loop,
    execute_reference,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
)
from tiny_tensor_compiler.inference import TypeInferenceError
from tiny_tensor_compiler.ir import DType
from tiny_tensor_compiler.serialization import deserialize_module, serialize_module


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _reduction_op(module, opcode: str):
    return next(op for op in module.function.ops if op.opcode == opcode)


def test_sum_i32_to_i64_changes_accumulator_and_result_dtype() -> None:
    builder = GraphBuilder()
    value = builder.input((2,), dtype="int32")
    result = value.sum(dtype="int64")
    module = builder.finish(result)

    assert result.type.dtype is DType.INT64
    assert _reduction_op(module, "sum").attrs == {"dtype": "i64"}

    runtime = np.array([np.iinfo(np.int32).max, 1], dtype=np.int32)
    expected = np.array(2**31, dtype=np.int64)
    np.testing.assert_array_equal(execute_reference(module, inputs=[runtime]), expected)
    np.testing.assert_array_equal(
        execute_loop(lower_to_loops(lower_to_cpu(module)), inputs=[runtime]),
        expected,
    )


def test_prod_i32_to_i64_avoids_i32_intermediate_wrap() -> None:
    builder = GraphBuilder()
    value = builder.input((2,), dtype="int32")
    module = builder.finish(value.prod(dtype=DType.INT64))
    runtime = np.array([50_000, 50_000], dtype=np.int32)

    expected = np.array(2_500_000_000, dtype=np.int64)
    np.testing.assert_array_equal(execute_reference(module, inputs=[runtime]), expected)
    np.testing.assert_array_equal(
        execute_loop(lower_to_loops(lower_to_cpu(module)), inputs=[runtime]), expected
    )


def test_sum_f32_to_f64_uses_f64_left_fold() -> None:
    builder = GraphBuilder()
    value = builder.input((3,), dtype="float32")
    module = builder.finish(value.sum(dtype=np.dtype("float64")))
    runtime = np.array([1e8, 1.0, -1e8], dtype=np.float32)

    actual = execute_reference(module, inputs=[runtime])
    assert actual.dtype == np.dtype("float64")
    assert actual.item() == 1.0


def test_widened_reduction_supports_multi_axis_keepdims_and_views() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="int32")
    viewed = value.transpose((2, 0, 1)).reverse(2)
    result = viewed.sum(axis=(0, 2), keepdims=True, dtype="int64")
    module = builder.finish(result)

    assert result.type.shape == (1, 2, 1)
    assert result.type.dtype is DType.INT64
    runtime = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    logical = runtime.transpose(2, 0, 1)[:, :, ::-1]
    expected = logical.sum(axis=(0, 2), dtype=np.int64, keepdims=True)
    np.testing.assert_array_equal(execute_reference(module, inputs=[runtime]), expected)
    np.testing.assert_array_equal(
        execute_loop(lower_to_loops(lower_to_cpu(module)), inputs=[runtime]), expected
    )


def test_reduction_dtype_rejects_narrowing_and_cross_kind_conversion() -> None:
    cases = [
        ("int64", "int32"),
        ("float64", "float32"),
        ("int32", "float64"),
        ("float32", "int64"),
    ]
    for source_dtype, result_dtype in cases:
        builder = GraphBuilder()
        value = builder.input((2,), dtype=source_dtype)
        with pytest.raises(TypeInferenceError, match="unsupported reduction dtype conversion"):
            value.sum(dtype=result_dtype)


def test_explicit_same_dtype_is_canonicalized_to_historical_ir() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="int32")
    module = builder.finish(value.sum(axis=1, dtype="int32"))

    assert _reduction_op(module, "sum").attrs == {"axis": 1}


def test_generated_c_explicitly_casts_source_to_widened_accumulator() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="int32")
    module = builder.finish(value.sum(axis=1, dtype="int64"))
    source = generate_c(lower_to_loops(lower_to_cpu(module)))

    assert "int64_t sum_value = 0LL;" in source
    assert "sum_value = ((int64_t)sum_value + (int64_t)" in source


def test_native_widened_reductions_compose_with_borrowing_and_parallel() -> None:
    _default_compiler_or_skip()
    builder = GraphBuilder()
    value = builder.input((4, 6), dtype="int32")
    module = builder.finish(
        (
            value.reverse(1).sum(axis=1, dtype="int64"),
            value.prod(axis=0, dtype="int64"),
        )
    )
    runtime = (np.arange(24, dtype=np.int32).reshape(4, 6) % 17) + 40_000

    actual = compile_module(module, borrow_inputs=True, parallel=True)(inputs=[runtime])
    expected = execute_reference(module, inputs=[runtime])
    assert isinstance(actual, tuple)
    assert isinstance(expected, tuple)
    for result, wanted in zip(actual, expected, strict=True):
        assert result.dtype == np.dtype("int64")
        np.testing.assert_array_equal(result, wanted)


def test_widened_empty_reduction_uses_identity_in_result_dtype() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 0), dtype="float32")
    module = builder.finish(
        (
            value.sum(axis=1, dtype="float64"),
            value.prod(axis=1, dtype="float64"),
        )
    )
    runtime = np.empty((2, 0), dtype=np.float32)

    actual = execute_reference(module, inputs=[runtime])
    assert isinstance(actual, tuple)
    np.testing.assert_array_equal(actual[0], np.zeros((2,), dtype=np.float64))
    np.testing.assert_array_equal(actual[1], np.ones((2,), dtype=np.float64))


def test_reduction_dtype_round_trips_through_canonical_serialization() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="float32")
    module = builder.finish(value.sum(axis=0, dtype="float64"))

    document = serialize_module(module)
    restored = deserialize_module(document)
    assert _reduction_op(restored, "sum").attrs == {"axis": 0, "dtype": "f64"}
    assert _reduction_op(restored, "sum").results[0].type.dtype is DType.FLOAT64
    assert serialize_module(restored) == document
