from __future__ import annotations

import json

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    SymbolicDim,
    compile_dynamic_module,
    compile_module,
    execute_reference,
)
from tiny_tensor_compiler.serialization import (
    IRSerializationError,
    deserialize_module,
    module_sha256,
    serialize_module,
)


def _build_alias_effect_module():
    builder = GraphBuilder()
    x = builder.input((2, 4), dtype="int32")
    root = x + 1
    target = root.view((4, 2)).reverse(1).transpose((1, 0))
    fresh = root.copy_into(target, x)
    gathered = fresh.transpose((1, 0)).slice(axis=0, start=0, stop=4, step=2).reverse(1)
    return builder.finish((fresh, gathered))


def _build_dynamic_module():
    builder = GraphBuilder()
    batch = SymbolicDim("B")
    width = SymbolicDim("W")
    x = builder.input((batch + width, 2 * batch + width + 1), dtype="float32")
    y = builder.input((2 * batch + 1, width), dtype="float32")
    return builder.finish((x.relu(), y.relu()))


def test_canonical_round_trip_preserves_alias_effect_semantics_and_native_execution():
    module = _build_alias_effect_module()
    document = serialize_module(module)
    restored = deserialize_module(document)

    assert serialize_module(restored) == document
    assert module_sha256(restored) == module_sha256(module)
    assert restored.dump() == module.dump()

    x = np.arange(8, dtype=np.int32).reshape(2, 4)
    expected = execute_reference(module, inputs=[x])
    actual_reference = execute_reference(restored, inputs=[x])
    actual_native = compile_module(restored)(inputs=[x])

    assert isinstance(expected, tuple)
    assert isinstance(actual_reference, tuple)
    assert isinstance(actual_native, tuple)
    for expected_output, reference_output, native_output in zip(
        expected,
        actual_reference,
        actual_native,
        strict=True,
    ):
        np.testing.assert_array_equal(reference_output, expected_output)
        np.testing.assert_array_equal(native_output, expected_output)


def test_dynamic_symbolic_shape_terms_round_trip_and_specialize_natively():
    module = _build_dynamic_module()
    document = serialize_module(module)
    restored = deserialize_module(document)

    assert serialize_module(restored) == document
    assert module_sha256(restored) == module_sha256(module)

    x = np.arange(40, dtype=np.float32).reshape(5, 8) - 20
    y = np.arange(15, dtype=np.float32).reshape(5, 3) - 7
    expected = execute_reference(module, inputs=[x, y])
    actual_reference = execute_reference(restored, inputs=[x, y])
    executable = compile_dynamic_module(restored)
    actual_native = executable(inputs=[x, y])

    assert executable.cached_bindings == ((('B', 2), ('W', 3)),)
    assert isinstance(expected, tuple)
    assert isinstance(actual_reference, tuple)
    assert isinstance(actual_native, tuple)
    for expected_output, reference_output, native_output in zip(
        expected,
        actual_reference,
        actual_native,
        strict=True,
    ):
        np.testing.assert_array_equal(reference_output, expected_output)
        np.testing.assert_array_equal(native_output, expected_output)


def test_constant_payload_round_trip_preserves_exact_float32_bits():
    bits = np.array([0x00000000, 0x80000000, 0x7FC12345, 0x7F800000], dtype=np.uint32)
    values = bits.view(np.float32)
    builder = GraphBuilder()
    constant = builder.tensor(values)
    module = builder.finish(constant)

    document = serialize_module(module)
    restored = deserialize_module(document)
    restored_value = restored.function.ops[0].attrs["value"]

    np.testing.assert_array_equal(restored_value.view(np.uint32), bits)
    assert "NaN" not in document
    assert "Infinity" not in document
    assert serialize_module(restored) == document


def test_scalar_constant_round_trip_preserves_rank_zero_shape():
    builder = GraphBuilder()
    module = builder.finish(builder.tensor(7, dtype="int32"))

    document = serialize_module(module)
    payload = json.loads(document)
    encoded_value = payload["function"]["ops"][0]["attrs"]["value"]
    restored = deserialize_module(document)
    restored_value = restored.function.ops[0].attrs["value"]

    assert encoded_value["shape"] == []
    assert restored.function.ops[0].results[0].type.shape == ()
    assert restored_value.shape == ()
    assert restored_value.item() == 7
    assert serialize_module(restored) == document


def test_equivalent_modules_have_identical_canonical_bytes_and_fingerprint():
    lhs = _build_alias_effect_module()
    rhs = _build_alias_effect_module()

    assert serialize_module(lhs) == serialize_module(rhs)
    assert module_sha256(lhs) == module_sha256(rhs)
    assert len(module_sha256(lhs)) == 64


def test_deserializer_rejects_unknown_version_and_unknown_top_level_fields():
    document = serialize_module(_build_alias_effect_module())
    payload = json.loads(document)
    payload["version"] = 2
    with pytest.raises(IRSerializationError, match="unsupported tensor IR version"):
        deserialize_module(json.dumps(payload))

    payload = json.loads(document)
    payload["unexpected"] = True
    with pytest.raises(IRSerializationError, match="unexpected keys"):
        deserialize_module(json.dumps(payload))


def test_deserializer_rejects_forward_value_references():
    document = serialize_module(_build_alias_effect_module())
    payload = json.loads(document)
    first_consumer = next(op for op in payload["function"]["ops"] if op["operands"])
    first_consumer["operands"][0] = 999

    with pytest.raises(IRSerializationError, match="undefined value %999"):
        deserialize_module(json.dumps(payload))


def test_deserializer_rejects_corrupt_constant_payload():
    builder = GraphBuilder()
    module = builder.finish(builder.tensor([1, 2, 3], dtype="int32"))
    payload = json.loads(serialize_module(module))
    constant = payload["function"]["ops"][0]
    constant["attrs"]["value"]["data"] = "not-base64!"

    with pytest.raises(IRSerializationError, match="base64"):
        deserialize_module(json.dumps(payload))


def test_deserializer_rejects_duplicate_json_keys():
    document = serialize_module(_build_alias_effect_module())
    duplicate = document.replace('{"format":', '{"format":"tiny-tensor-ir","format":', 1)

    with pytest.raises(IRSerializationError, match="duplicate JSON object key"):
        deserialize_module(duplicate)
