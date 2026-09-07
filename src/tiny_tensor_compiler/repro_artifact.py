from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compiler_trace import trace_module
from .ir import Module
from .serialization import IRSerializationError, deserialize_module, serialize_module
from .trace_diff import CompilerTraceComparison, TraceSnapshotError, compare_trace_json

_REPRO_FORMAT = "tiny-tensor-compiler-repro"
_REPRO_VERSION = 1
_REPRO_KEYS = {
    "config",
    "format",
    "module",
    "module_sha256",
    "payload_sha256",
    "trace",
    "trace_sha256",
    "version",
}
_CONFIG_KEYS = {"borrow_inputs", "parallel"}


class ReproArtifactError(ValueError):
    """Raised when a compiler repro artifact fails its format or integrity contract."""


@dataclass(frozen=True)
class CompilerReproArtifact:
    """Canonical compiler input, configuration, and expected deterministic trace."""

    module_json: str
    module_sha256: str
    trace_json: str
    trace_sha256: str
    borrow_inputs: bool
    parallel: bool
    payload_sha256: str
    format: str = _REPRO_FORMAT
    version: int = _REPRO_VERSION

    def _core_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "borrow_inputs": self.borrow_inputs,
                "parallel": self.parallel,
            },
            "format": self.format,
            "module": self.module_json,
            "module_sha256": self.module_sha256,
            "trace": self.trace_json,
            "trace_sha256": self.trace_sha256,
            "version": self.version,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._core_dict()
        payload["payload_sha256"] = self.payload_sha256
        return payload

    def to_json(self) -> str:
        """Return deterministic JSON for storage or cross-process replay."""
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class ReproReplayResult:
    """Result of replaying one validated repro artifact through the current compiler."""

    comparison: CompilerTraceComparison

    @property
    def equal(self) -> bool:
        return self.comparison.equal

    def render(self) -> str:
        if self.equal:
            return "reproduced"
        return "not reproduced\n" + self.comparison.render()


def capture_repro_artifact(
    module: Module,
    *,
    borrow_inputs: bool = False,
    parallel: bool = False,
) -> CompilerReproArtifact:
    """Capture one concrete module and its exact deterministic compiler trace."""
    if not isinstance(module, Module):
        raise TypeError("capture_repro_artifact requires a Module")
    if not isinstance(borrow_inputs, bool):
        raise TypeError("borrow_inputs must be a bool")
    if not isinstance(parallel, bool):
        raise TypeError("parallel must be a bool")

    module_json = serialize_module(module)
    trace_json = trace_module(
        module,
        borrow_inputs=borrow_inputs,
        parallel=parallel,
    ).to_json()
    core = {
        "config": {
            "borrow_inputs": borrow_inputs,
            "parallel": parallel,
        },
        "format": _REPRO_FORMAT,
        "module": module_json,
        "module_sha256": _sha256_text(module_json),
        "trace": trace_json,
        "trace_sha256": _sha256_text(trace_json),
        "version": _REPRO_VERSION,
    }
    return CompilerReproArtifact(
        module_json=module_json,
        module_sha256=core["module_sha256"],
        trace_json=trace_json,
        trace_sha256=core["trace_sha256"],
        borrow_inputs=borrow_inputs,
        parallel=parallel,
        payload_sha256=_sha256_text(_canonical_json(core)),
    )


def deserialize_repro_artifact(document: str) -> CompilerReproArtifact:
    """Validate every stored digest and rebuild one fail-closed repro artifact."""
    if not isinstance(document, str):
        raise TypeError("deserialize_repro_artifact requires a JSON string")

    try:
        payload = json.loads(document, object_pairs_hook=_object_without_duplicates)
    except json.JSONDecodeError as exc:
        raise ReproArtifactError(f"invalid compiler repro JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ReproArtifactError("compiler repro root must be a JSON object")
    _require_exact_keys(payload, _REPRO_KEYS, "compiler repro")
    if payload["format"] != _REPRO_FORMAT:
        raise ReproArtifactError("unsupported compiler repro format")
    version = payload["version"]
    if type(version) is not int or version != _REPRO_VERSION:
        raise ReproArtifactError(f"unsupported compiler repro version {version!r}")

    config = payload["config"]
    if not isinstance(config, dict):
        raise ReproArtifactError("compiler repro config must be a JSON object")
    _require_exact_keys(config, _CONFIG_KEYS, "compiler repro config")
    borrow_inputs = config["borrow_inputs"]
    parallel = config["parallel"]
    if type(borrow_inputs) is not bool:
        raise ReproArtifactError("compiler repro borrow_inputs must be a bool")
    if type(parallel) is not bool:
        raise ReproArtifactError("compiler repro parallel must be a bool")

    module_json = _require_string(payload["module"], "compiler repro module")
    trace_json = _require_string(payload["trace"], "compiler repro trace")
    module_sha256 = _require_sha256(payload["module_sha256"], "module SHA-256")
    trace_sha256 = _require_sha256(payload["trace_sha256"], "trace SHA-256")
    payload_sha256 = _require_sha256(payload["payload_sha256"], "payload SHA-256")

    core = dict(payload)
    core.pop("payload_sha256")
    if payload_sha256 != _sha256_text(_canonical_json(core)):
        raise ReproArtifactError("compiler repro payload SHA-256 does not match its content")
    if module_sha256 != _sha256_text(module_json):
        raise ReproArtifactError("compiler repro module SHA-256 does not match its content")
    if trace_sha256 != _sha256_text(trace_json):
        raise ReproArtifactError("compiler repro trace SHA-256 does not match its content")

    try:
        module = deserialize_module(module_json)
    except IRSerializationError as exc:
        raise ReproArtifactError(f"compiler repro embedded module is invalid: {exc}") from exc
    if serialize_module(module) != module_json:
        raise ReproArtifactError("compiler repro embedded module is not canonical")

    try:
        comparison = compare_trace_json(trace_json, trace_json)
    except TraceSnapshotError as exc:
        raise ReproArtifactError(f"compiler repro trace snapshot is invalid: {exc}") from exc
    if not comparison.equal:
        raise RuntimeError("validated compiler trace unexpectedly differs from itself")

    try:
        trace_payload = json.loads(trace_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError("validated compiler trace unexpectedly failed JSON decode") from exc
    if _canonical_json(trace_payload) != trace_json:
        raise ReproArtifactError("compiler repro trace snapshot is not canonical JSON")
    if trace_payload["borrow_inputs"] != borrow_inputs or trace_payload["parallel"] != parallel:
        raise ReproArtifactError("compiler repro config does not match embedded trace config")
    phases = trace_payload["phases"]
    if phases[0]["name"] != "tensor_ir" or phases[0]["text"] != module_json:
        raise ReproArtifactError("compiler repro trace tensor_ir does not match embedded module")

    return CompilerReproArtifact(
        module_json=module_json,
        module_sha256=module_sha256,
        trace_json=trace_json,
        trace_sha256=trace_sha256,
        borrow_inputs=borrow_inputs,
        parallel=parallel,
        payload_sha256=payload_sha256,
    )


def replay_repro_artifact(
    artifact: CompilerReproArtifact | str,
) -> ReproReplayResult:
    """Replay one artifact through the current compiler and localize any trace drift."""
    if isinstance(artifact, CompilerReproArtifact):
        validated = deserialize_repro_artifact(artifact.to_json())
    elif isinstance(artifact, str):
        validated = deserialize_repro_artifact(artifact)
    else:
        raise TypeError("replay_repro_artifact requires a CompilerReproArtifact or JSON string")

    try:
        module = deserialize_module(validated.module_json)
    except IRSerializationError as exc:
        raise RuntimeError("validated compiler repro module unexpectedly failed decode") from exc
    current_trace = trace_module(
        module,
        borrow_inputs=validated.borrow_inputs,
        parallel=validated.parallel,
    ).to_json()
    comparison = compare_trace_json(validated.trace_json, current_trace)
    return ReproReplayResult(comparison=comparison)


def main(argv: list[str] | None = None) -> int:
    """Capture or replay repro artifacts with stable automation-oriented exit codes."""
    parser = argparse.ArgumentParser(
        description="Capture and replay deterministic tiny-tensor-compiler repro artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser(
        "capture",
        help="capture a repro artifact from serialized tensor IR",
    )
    capture_parser.add_argument("module", help="path to a serialized tensor-IR JSON document")
    capture_parser.add_argument("output", help="path to write the repro artifact")
    capture_parser.add_argument("--borrow-inputs", action="store_true")
    capture_parser.add_argument("--parallel", action="store_true")

    replay_parser = subparsers.add_parser(
        "replay",
        help="replay a stored repro artifact through the current compiler",
    )
    replay_parser.add_argument("artifact", help="path to a compiler repro artifact")

    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            module_document = _read_text(args.module, label="module")
            module = deserialize_module(module_document)
            artifact = capture_repro_artifact(
                module,
                borrow_inputs=args.borrow_inputs,
                parallel=args.parallel,
            )
            _write_text(args.output, artifact.to_json() + "\n", label="artifact")
            print(artifact.payload_sha256)
            return 0

        document = _read_text(args.artifact, label="artifact")
        result = replay_repro_artifact(document)
        print(result.render())
        return 0 if result.equal else 1
    except (IRSerializationError, ReproArtifactError, OSError, UnicodeError, ValueError) as exc:
        print(f"invalid repro artifact: {exc}", file=sys.stderr)
        return 2


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ReproArtifactError("compiler repro contains non-canonical JSON data") from exc


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReproArtifactError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    keys = set(value)
    if keys == expected:
        return
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    details: list[str] = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if extra:
        details.append("unexpected " + ", ".join(extra))
    raise ReproArtifactError(f"{label} has invalid fields ({'; '.join(details)})")


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ReproArtifactError(f"{label} must be a string")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ReproArtifactError(f"compiler repro {label} is malformed")
    return value


def _read_text(path: str | Path, *, label: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReproArtifactError(f"cannot read {label} {path}: {exc}") from exc


def _write_text(path: str | Path, text: str, *, label: str) -> None:
    try:
        Path(path).write_text(text, encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReproArtifactError(f"cannot write {label} {path}: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
