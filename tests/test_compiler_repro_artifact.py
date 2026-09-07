import dataclasses
import hashlib
import json

import pytest

from tiny_tensor_compiler.compiler_trace import CompilerTracePhase, trace_module
from tiny_tensor_compiler.frontend import GraphBuilder
from tiny_tensor_compiler.repro_artifact import (
    CompilerReproArtifact,
    ReproArtifactError,
    capture_repro_artifact,
    deserialize_repro_artifact,
    main,
    replay_repro_artifact,
)
from tiny_tensor_compiler.serialization import deserialize_module, serialize_module


def _module(*, scale: float = 1.0):
    builder = GraphBuilder()
    value = builder.input((8,), dtype="float32")
    bias = builder.constant([scale] * 8, dtype="float32")
    return builder.finish((value + bias).relu())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_capture_is_canonical_deterministic_and_self_verifying():
    first = capture_repro_artifact(_module())
    second = capture_repro_artifact(_module())

    assert first == second
    assert first.to_json() == second.to_json()
    assert first.module_sha256 == _sha256(first.module_json)
    assert first.trace_sha256 == _sha256(first.trace_json)
    assert serialize_module(deserialize_module(first.module_json)) == first.module_json
    assert json.loads(first.trace_json) == json.loads(trace_module(_module()).to_json())

    payload = json.loads(first.to_json())
    assert payload["format"] == "tiny-tensor-compiler-repro"
    assert payload["version"] == 1
    assert payload["config"] == {"borrow_inputs": False, "parallel": False}
    core = dict(payload)
    digest = core.pop("payload_sha256")
    canonical_core = json.dumps(core, sort_keys=True, separators=(",", ":"))
    assert digest == _sha256(canonical_core)
    assert deserialize_repro_artifact(first.to_json()) == first


def test_replay_reruns_real_pipeline_with_captured_configuration():
    artifact = capture_repro_artifact(
        _module(),
        borrow_inputs=True,
        parallel=True,
    )

    result = replay_repro_artifact(artifact.to_json())

    assert result.equal is True
    assert result.comparison.equal is True
    assert result.comparison.configuration_changes == ()
    assert result.comparison.first_divergent_phase is None
    assert result.render() == "reproduced"

    trace_payload = json.loads(artifact.trace_json)
    assert trace_payload["borrow_inputs"] is True
    assert trace_payload["parallel"] is True
    assert trace_payload["phases"][-1]["name"] == "generated_c"
    assert "#pragma omp parallel for schedule(static)" in trace_payload["phases"][-1]["text"]


def test_replay_reports_first_pipeline_divergence_without_native_compilation(monkeypatch):
    import tiny_tensor_compiler.repro_artifact as repro_module

    artifact = capture_repro_artifact(_module())
    baseline = trace_module(_module())
    changed_phase = CompilerTracePhase.capture(
        "generated_c",
        baseline.phase("generated_c").text + "\n/* simulated compiler drift */\n",
    )
    changed_trace = dataclasses.replace(
        baseline,
        phases=(*baseline.phases[:-1], changed_phase),
    )
    monkeypatch.setattr(repro_module, "trace_module", lambda module, **kwargs: changed_trace)

    result = replay_repro_artifact(artifact)

    assert result.equal is False
    assert result.comparison.first_divergent_phase == "generated_c"
    assert result.comparison.changed_phases == ("generated_c",)
    assert result.render().startswith("not reproduced\n")
    assert "first divergent phase: generated_c" in result.render()


def test_artifact_rejects_payload_and_embedded_digest_tampering():
    artifact = capture_repro_artifact(_module())
    payload = json.loads(artifact.to_json())

    payload["config"]["parallel"] = True
    with pytest.raises(ReproArtifactError, match="payload SHA-256"):
        deserialize_repro_artifact(json.dumps(payload))

    payload = json.loads(artifact.to_json())
    payload["module"] += " "
    core = dict(payload)
    core.pop("payload_sha256")
    payload["payload_sha256"] = _sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"))
    )
    with pytest.raises(ReproArtifactError, match="module SHA-256"):
        deserialize_repro_artifact(json.dumps(payload))

    payload = json.loads(artifact.to_json())
    payload["trace"] += " "
    payload["trace_sha256"] = _sha256(payload["trace"])
    core = dict(payload)
    core.pop("payload_sha256")
    payload["payload_sha256"] = _sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"))
    )
    with pytest.raises(ReproArtifactError, match="trace snapshot"):
        deserialize_repro_artifact(json.dumps(payload))


def test_artifact_rejects_noncanonical_or_internally_inconsistent_content():
    artifact = capture_repro_artifact(_module())
    payload = json.loads(artifact.to_json())

    payload["version"] = 2
    core = dict(payload)
    core.pop("payload_sha256")
    payload["payload_sha256"] = _sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"))
    )
    with pytest.raises(ReproArtifactError, match="unsupported compiler repro version"):
        deserialize_repro_artifact(json.dumps(payload))

    duplicate = artifact.to_json().replace(
        '"format":"tiny-tensor-compiler-repro"',
        '"format":"tiny-tensor-compiler-repro","format":"tiny-tensor-compiler-repro"',
        1,
    )
    with pytest.raises(ReproArtifactError, match="duplicate JSON field"):
        deserialize_repro_artifact(duplicate)

    other = capture_repro_artifact(_module(scale=2.0))
    payload = json.loads(artifact.to_json())
    payload["trace"] = other.trace_json
    payload["trace_sha256"] = other.trace_sha256
    core = dict(payload)
    core.pop("payload_sha256")
    payload["payload_sha256"] = _sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"))
    )
    with pytest.raises(ReproArtifactError, match="tensor_ir does not match embedded module"):
        deserialize_repro_artifact(json.dumps(payload))


def test_cli_capture_and_replay_have_stable_exit_contract(tmp_path, capsys):
    module_path = tmp_path / "module.json"
    artifact_path = tmp_path / "repro.json"
    module_path.write_text(serialize_module(_module()), encoding="utf-8")

    assert main(
        [
            "capture",
            str(module_path),
            str(artifact_path),
            "--borrow-inputs",
            "--parallel",
        ]
    ) == 0
    captured = deserialize_repro_artifact(artifact_path.read_text(encoding="utf-8"))
    assert isinstance(captured, CompilerReproArtifact)
    assert captured.borrow_inputs is True
    assert captured.parallel is True

    assert main(["replay", str(artifact_path)]) == 0
    replay_output = capsys.readouterr().out
    assert "reproduced" in replay_output

    artifact_path.write_text("not-json", encoding="utf-8")
    assert main(["replay", str(artifact_path)]) == 2
    assert "invalid repro artifact:" in capsys.readouterr().err


def test_capture_rejects_unspecialized_symbolic_module_through_trace_contract():
    from tiny_tensor_compiler.ir import SymbolicDim

    batch = SymbolicDim("B")
    builder = GraphBuilder()
    value = builder.input((batch, 4), dtype="float32")
    module = builder.finish(value.relu())

    with pytest.raises(ValueError, match="requires concrete tensor shapes"):
        capture_repro_artifact(module)
