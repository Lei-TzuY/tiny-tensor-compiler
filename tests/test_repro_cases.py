from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from tiny_tensor_compiler import GraphBuilder, SymbolicDim, execute_reference
from tiny_tensor_compiler.repro import (
    ReproCaseError,
    ReproMismatchError,
    capture_repro_case,
    load_repro_case,
    replay_repro_case,
    repro_case_sha256,
)


def _build_view_module():
    builder = GraphBuilder()
    x = builder.input((2, 4), dtype="int32")
    root = (x + 1).relu()
    viewed = root.reverse(axis=1).transpose((1, 0))
    return builder.finish((root, viewed))


def _build_dynamic_module():
    builder = GraphBuilder()
    batch = SymbolicDim("B")
    x = builder.input((batch, 4), dtype="int32")
    reshaped = (x + 2).reshape((4, batch)).reverse(axis=1)
    return builder.finish(reshaped)


def _assert_same_result(actual, expected) -> None:
    actual_outputs = actual if isinstance(actual, tuple) else (actual,)
    expected_outputs = expected if isinstance(expected, tuple) else (expected,)
    assert len(actual_outputs) == len(expected_outputs)
    for actual_output, expected_output in zip(actual_outputs, expected_outputs, strict=True):
        np.testing.assert_array_equal(actual_output, expected_output)


def test_capture_is_canonical_deterministic_and_has_stable_content_fingerprint():
    module = _build_view_module()
    runtime_input = np.arange(8, dtype=np.int32).reshape(2, 4) - 3

    first = capture_repro_case(module, inputs=[runtime_input])
    second = capture_repro_case(_build_view_module(), inputs=[runtime_input.copy()])
    payload = json.loads(first)

    assert first == second
    assert set(payload) == {
        "expected_outputs",
        "format",
        "inputs",
        "module",
        "module_sha256",
        "version",
    }
    assert payload["format"] == "tiny-tensor-repro"
    assert payload["version"] == 1
    assert len(payload["module_sha256"]) == 64
    assert "timestamp" not in first
    assert "hostname" not in first

    pretty = json.dumps(payload, indent=2)
    assert repro_case_sha256(pretty) == repro_case_sha256(first)
    assert len(repro_case_sha256(first)) == 64


def test_reference_and_native_replay_match_captured_multi_output_view_case():
    module = _build_view_module()
    runtime_input = np.arange(8, dtype=np.int32).reshape(2, 4) - 4
    expected = execute_reference(module, inputs=[runtime_input])
    document = capture_repro_case(module, inputs=[runtime_input])

    _assert_same_result(replay_repro_case(document, backend="reference"), expected)
    _assert_same_result(replay_repro_case(document, backend="native"), expected)


def test_dynamic_symbolic_case_replays_through_native_specialization():
    module = _build_dynamic_module()
    runtime_input = np.arange(12, dtype=np.int32).reshape(3, 4) - 5
    expected = execute_reference(module, inputs=[runtime_input])
    document = capture_repro_case(module, inputs=[runtime_input])
    loaded = load_repro_case(document)

    assert loaded.inputs[0].shape == (3, 4)
    assert loaded.inputs[0].dtype == np.dtype(np.int32)
    _assert_same_result(replay_repro_case(document, backend="native"), expected)


def test_repro_payload_preserves_rank_zero_and_exact_float32_output_bits():
    bits = np.array([0x00000000, 0x80000000, 0x7FC12345, 0x7F800000], dtype=np.uint32)
    builder = GraphBuilder()
    module = builder.finish(
        (
            builder.tensor(7, dtype="int32"),
            builder.tensor(bits.view(np.float32)),
        )
    )

    loaded = load_repro_case(capture_repro_case(module))

    assert loaded.expected_outputs[0].shape == ()
    assert loaded.expected_outputs[0].item() == 7
    np.testing.assert_array_equal(loaded.expected_outputs[1].view(np.uint32), bits)


def test_loader_rejects_module_fingerprint_tampering_and_corrupt_array_bytes():
    module = _build_view_module()
    runtime_input = np.arange(8, dtype=np.int32).reshape(2, 4)
    payload = json.loads(capture_repro_case(module, inputs=[runtime_input]))

    payload["module_sha256"] = "0" * 64
    with pytest.raises(ReproCaseError, match="module SHA-256"):
        load_repro_case(json.dumps(payload))

    payload = json.loads(capture_repro_case(module, inputs=[runtime_input]))
    payload["inputs"][0]["data"] = "not-base64!"
    with pytest.raises(ReproCaseError, match="base64"):
        load_repro_case(json.dumps(payload))


def test_replay_reports_exact_expected_output_bit_mismatch():
    builder = GraphBuilder()
    x = builder.input((4,), dtype="int32")
    module = builder.finish(x + 1)
    runtime_input = np.array([1, 2, 3, 4], dtype=np.int32)
    document = capture_repro_case(module, inputs=[runtime_input])
    original_digest = repro_case_sha256(document)
    payload = json.loads(document)
    encoded = payload["expected_outputs"][0]["data"]
    raw = bytearray(base64.b64decode(encoded))
    raw[0] ^= 1
    payload["expected_outputs"][0]["data"] = base64.b64encode(raw).decode("ascii")
    tampered = json.dumps(payload)

    assert repro_case_sha256(tampered) != original_digest
    with pytest.raises(ReproMismatchError, match="output #0 raw bytes mismatch"):
        replay_repro_case(tampered, backend="reference")


def test_loader_rejects_unknown_schema_fields_versions_and_duplicate_keys():
    document = capture_repro_case(_build_view_module(), inputs=[np.zeros((2, 4), dtype=np.int32)])
    payload = json.loads(document)
    payload["version"] = 2
    with pytest.raises(ReproCaseError, match="version"):
        load_repro_case(json.dumps(payload))

    payload = json.loads(document)
    payload["unexpected"] = True
    with pytest.raises(ReproCaseError, match="unexpected keys"):
        load_repro_case(json.dumps(payload))

    duplicate = document.replace(
        '{"expected_outputs":',
        '{"expected_outputs":[],"expected_outputs":',
        1,
    )
    with pytest.raises(ReproCaseError, match="duplicate JSON object key"):
        load_repro_case(duplicate)


def test_replay_rejects_unknown_backend():
    document = capture_repro_case(_build_view_module(), inputs=[np.zeros((2, 4), dtype=np.int32)])

    with pytest.raises(ValueError, match="backend"):
        replay_repro_case(document, backend="gpu")
