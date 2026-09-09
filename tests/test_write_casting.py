import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    compile_module,
    execute_reference,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
)
from tiny_tensor_compiler.serialization import deserialize_module, serialize_module


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _copy_module(*, target_dtype: str, source_dtype: str, casting: str = "widen"):
    builder = GraphBuilder()
    base = builder.input((2, 3), dtype=target_dtype)
    source = builder.input((2, 1), dtype=source_dtype)
    root = base.relu()
    target = root.reverse(0)
    return builder.finish(root.copy_into(target, source, casting=casting))


def _binary_module(*, casting: str = "widen"):
    builder = GraphBuilder()
    base = builder.input((2, 6), dtype="float64")
    source = builder.input((3,), dtype="float32")
    root = base.relu()
    target = root.slice(axis=1, start=0, stop=6, step=2)
    return builder.finish(root.add_into(target, source, casting=casting))


@pytest.mark.parametrize(
    ("source_dtype", "target_dtype"),
    (("int32", "int64"), ("int32", "float64"), ("float32", "float64")),
)
def test_widening_write_materializes_exact_typed_source(source_dtype, target_dtype):
    module = _copy_module(target_dtype=target_dtype, source_dtype=source_dtype)
    effect = next(op for op in module.function.ops if op.opcode == "copy_into")
    materialized = effect.operands[2]

    assert materialized.type.dtype == effect.operands[1].type.dtype
    assert materialized.producer is not None
    assert materialized.producer.opcode == "mul"
    assert materialized.producer.operands[0].type.dtype.value == {
        "int32": "i32",
        "float32": "f32",
    }[source_dtype]
    identity = materialized.producer.operands[1]
    assert identity.producer is not None
    assert identity.producer.opcode == "const"
    assert identity.type.dtype == materialized.type.dtype
    assert np.asarray(identity.producer.attrs["value"]).item() == 1


@pytest.mark.parametrize(
    ("source_dtype", "target_dtype"),
    (("int64", "float64"), ("float64", "float32"), ("float32", "int64")),
)
def test_widening_write_rejects_non_exact_or_narrowing_conversions(source_dtype, target_dtype):
    with pytest.raises(ValueError, match="cannot widen"):
        _copy_module(target_dtype=target_dtype, source_dtype=source_dtype)


def test_write_casting_default_remains_exact_and_policy_is_explicit():
    builder = GraphBuilder()
    base = builder.input((2, 3), dtype="float64")
    source = builder.input((2, 3), dtype="float32")
    root = base.relu()
    with pytest.raises(ValueError, match="dtypes must exactly match"):
        root.copy_into(root, source)
    with pytest.raises(ValueError, match="casting must be 'exact' or 'widen'"):
        root.copy_into(root, source, casting="unsafe")


def test_same_dtype_widen_policy_does_not_insert_materialization():
    builder = GraphBuilder()
    base = builder.input((2, 3), dtype="int32")
    source = builder.input((2, 3), dtype="int32")
    root = base.relu()
    module = builder.finish(root.copy_into(root, source, casting="widen"))
    effect = next(op for op in module.function.ops if op.opcode == "copy_into")

    assert effect.operands[2] is source.value
    assert sum(op.opcode == "mul" for op in module.function.ops) == 0


def test_widening_write_serialization_records_lowered_exact_ir():
    module = _copy_module(target_dtype="float64", source_dtype="int32")
    restored = deserialize_module(serialize_module(module))
    effect = next(op for op in restored.function.ops if op.opcode == "copy_into")

    assert effect.attrs == {}
    assert effect.operands[2].type.dtype == effect.operands[1].type.dtype
    assert effect.operands[2].producer is not None
    assert effect.operands[2].producer.opcode == "mul"


def test_float_widening_preserves_signed_zero_reference_and_native():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    base = builder.input((2,), dtype="float64")
    source_tensor = builder.input((2,), dtype="float32")
    root = base.relu()
    module = builder.finish(root.copy_into(root, source_tensor, casting="widen"))

    base_value = np.array([3.0, 4.0], dtype=np.float64)
    source = np.array([-0.0, 0.0], dtype=np.float32)
    expected = source.astype(np.float64)
    reference = execute_reference(module, inputs=[base_value, source])
    actual = compile_module(module, borrow_inputs=True)(inputs=[base_value, source])

    np.testing.assert_array_equal(reference, expected)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(reference), np.signbit(expected))
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


def test_widening_copy_native_handles_broadcast_negative_stride_borrow_and_openmp():
    _default_compiler_or_skip()
    module = _copy_module(target_dtype="float64", source_dtype="int32")
    base = np.array([[-4.0, 2.0, 3.0], [5.0, -6.0, 7.0]], dtype=np.float64)
    source = np.array([[11], [23]], dtype=np.int32)
    base_before = base.copy()
    source_before = source.copy()

    reference = execute_reference(module, inputs=[base, source])
    actual = compile_module(module, borrow_inputs=True, parallel=True)(inputs=[base, source])
    expected = np.maximum(base, 0)
    expected[::-1, :] = source.astype(np.float64)

    np.testing.assert_array_equal(reference, expected)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(base, base_before)
    np.testing.assert_array_equal(source, source_before)


def test_widening_binary_into_native_handles_broadcast_partial_target_and_openmp():
    _default_compiler_or_skip()
    module = _binary_module()
    base = np.arange(12, dtype=np.float64).reshape(2, 6) - 4
    source = np.array([0.5, 1.25, 2.0], dtype=np.float32)

    reference = execute_reference(module, inputs=[base, source])
    actual = compile_module(module, borrow_inputs=True, parallel=True)(inputs=[base, source])
    expected = np.maximum(base, 0)
    expected[:, 0:6:2] += source.astype(np.float64)

    np.testing.assert_array_equal(reference, expected)
    np.testing.assert_array_equal(actual, expected)


def test_widening_policy_lowers_before_effect_codegen():
    module = _binary_module()
    loops = lower_to_loops(lower_to_cpu(module))
    effect = loops.binary_intos[0]
    generated = generate_c(loops, parallel=True)

    assert loops.value_types[effect.source].dtype == loops.value_types[effect.target].dtype
    assert "#pragma omp parallel for schedule(static)" in generated
    assert f"p{effect.source}[i1]" in generated
