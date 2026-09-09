import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    TypeInferenceError,
    algebraic_simplify,
    compile_module,
    constant_fold,
    execute_reference,
    verify,
)
from tiny_tensor_compiler.casting import lossless_cast
from tiny_tensor_compiler.ir import DType


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("int32", "int64"),
        ("int32", "float64"),
        ("float32", "float64"),
    ],
)
def test_lossless_cast_lowers_to_existing_verified_arithmetic(source, target):
    builder = GraphBuilder()
    value = builder.input((2, 3), dtype=source)
    converted = lossless_cast(value, target)
    module = builder.finish(converted)

    verify(module)
    assert converted.type.shape == (2, 3)
    assert converted.type.dtype == DType.from_numpy(np.dtype(target))
    mul = next(op for op in module.function.ops if op.opcode == "mul")
    assert mul.results[0] is converted.value
    assert mul.operands[0] is value.value
    assert mul.operands[1].producer is not None
    assert mul.operands[1].producer.opcode == "const"
    assert mul.operands[1].type.shape == ()
    assert mul.operands[1].type.dtype == converted.type.dtype


def test_same_dtype_lossless_cast_is_identity_and_adds_no_operation():
    builder = GraphBuilder()
    value = builder.input((4,), dtype="int32")
    converted = lossless_cast(value, "int32")
    module = builder.finish(converted)

    assert converted is value
    assert [op.opcode for op in module.function.ops] == ["input", "return"]


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("int64", "float64"),
        ("int32", "float32"),
        ("float32", "int32"),
        ("float64", "float32"),
        ("int64", "int32"),
    ],
)
def test_lossless_cast_rejects_conversion_without_full_value_domain_proof(source, target):
    builder = GraphBuilder()
    value = builder.input((2,), dtype=source)

    with pytest.raises(TypeInferenceError, match="lossless cast is not defined"):
        lossless_cast(value, target)


def test_lossless_cast_reference_preserves_integer_extremes_exactly():
    builder = GraphBuilder()
    value = builder.input((4,), dtype="int32")
    as_i64 = lossless_cast(value, "int64")
    as_f64 = lossless_cast(value, "float64")
    module = builder.finish((as_i64, as_f64))
    inputs = np.array([np.iinfo(np.int32).min, -1, 0, np.iinfo(np.int32).max], dtype=np.int32)

    actual_i64, actual_f64 = execute_reference(module, inputs=[inputs])
    np.testing.assert_array_equal(actual_i64, inputs.astype(np.int64))
    np.testing.assert_array_equal(actual_f64, inputs.astype(np.float64))


def test_lossless_cast_reference_preserves_float32_values_and_signed_zero():
    builder = GraphBuilder()
    value = builder.input((7,), dtype="float32")
    converted = lossless_cast(value, "float64")
    module = builder.finish(converted)
    inputs = np.array([-0.0, 0.0, -1.5, 2.25, np.inf, -np.inf, np.nan], dtype=np.float32)

    actual = execute_reference(module, inputs=[inputs])
    expected = inputs.astype(np.float64)
    np.testing.assert_array_equal(actual[:-1], expected[:-1])
    assert np.isnan(actual[-1])
    assert np.signbit(actual[0])
    assert not np.signbit(actual[1])


def test_lossless_cast_is_not_removed_as_integer_mul_identity_when_dtype_changes():
    builder = GraphBuilder()
    value = builder.input((3,), dtype="int32")
    module = builder.finish(lossless_cast(value, "int64"))

    assert algebraic_simplify(module) == 0
    assert any(op.opcode == "mul" for op in module.function.ops)
    verify(module)


def test_constant_lossless_cast_uses_existing_constant_folding_semantics():
    builder = GraphBuilder()
    value = builder.tensor(np.array([np.iinfo(np.int32).min, 7], dtype=np.int32))
    converted = lossless_cast(value, "int64")
    module = builder.finish(converted)

    assert constant_fold(module) == 1
    actual = execute_reference(module)
    np.testing.assert_array_equal(actual, np.array([np.iinfo(np.int32).min, 7], dtype=np.int64))


def test_lossless_cast_composes_with_exact_dtype_broadcast_copy_and_parallel_native():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    base = builder.input((2, 4), dtype="int64")
    source = builder.input((), dtype="int32")
    root = base.relu()
    converted = lossless_cast(source, "int64")
    module = builder.finish(root.copy_into(root, converted))

    base_value = np.arange(8, dtype=np.int64).reshape(2, 4) - 3
    source_value = np.array(11, dtype=np.int32)
    reference = execute_reference(module, inputs=[base_value, source_value])
    native = compile_module(module, borrow_inputs=True, parallel=True)(
        inputs=[base_value, source_value]
    )
    expected = np.full((2, 4), 11, dtype=np.int64)
    np.testing.assert_array_equal(reference, expected)
    np.testing.assert_array_equal(native, expected)
