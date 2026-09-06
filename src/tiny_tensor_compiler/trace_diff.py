from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TRACE_FORMAT = "tiny-tensor-compiler-trace"
_TRACE_VERSION = 1
_EXPECTED_PHASES = (
    "tensor_ir",
    "buffer_ir",
    "memory_plan",
    "pre_fusion_loop_ir",
    "post_fusion_loop_ir",
    "execution_loop_ir",
    "generated_c",
)
_TOP_LEVEL_KEYS = {
    "borrow_inputs",
    "format",
    "parallel",
    "phases",
    "report",
    "version",
}
_PHASE_KEYS = {"name", "text", "sha256"}


class TraceSnapshotError(ValueError):
    """Raised when a stored compiler trace fails its format or digest contract."""


@dataclass(frozen=True)
class _TraceSnapshotPhase:
    name: str
    text: str
    sha256: str


@dataclass(frozen=True)
class _TraceSnapshot:
    borrow_inputs: bool
    parallel: bool
    phases: tuple[_TraceSnapshotPhase, ...]
    report_json: str


@dataclass(frozen=True)
class CompilerTracePhaseDiff:
    """One deterministic unified diff between matching compiler trace phases."""

    name: str
    before_sha256: str
    after_sha256: str
    unified_diff: str


@dataclass(frozen=True)
class CompilerTraceComparison:
    """Structured result of comparing two validated compiler trace snapshots."""

    equal: bool
    configuration_changes: tuple[str, ...]
    report_changed: bool
    first_divergent_phase: str | None
    changed_phases: tuple[str, ...]
    phase_diffs: tuple[CompilerTracePhaseDiff, ...]

    def render(self) -> str:
        """Render a deterministic human-readable diagnostic report."""
        if self.equal:
            return "equal"

        lines = ["different"]
        if self.configuration_changes:
            lines.append(
                "configuration changes: " + ", ".join(self.configuration_changes)
            )
        if self.report_changed:
            lines.append("report: changed")
        if self.first_divergent_phase is not None:
            lines.append(f"first divergent phase: {self.first_divergent_phase}")
        if self.changed_phases:
            lines.append("changed phases: " + ", ".join(self.changed_phases))
        for phase_diff in self.phase_diffs:
            lines.extend(("", phase_diff.unified_diff))
        return "\n".join(lines)


def compare_trace_json(before_json: str, after_json: str) -> CompilerTraceComparison:
    """Compare two stored v1 compiler traces after validating their self-digests."""
    if not isinstance(before_json, str) or not isinstance(after_json, str):
        raise TypeError("compare_trace_json requires two JSON strings")

    before = _parse_trace_json(before_json, label="before")
    after = _parse_trace_json(after_json, label="after")

    configuration_changes = tuple(
        name
        for name, before_value, after_value in (
            ("borrow_inputs", before.borrow_inputs, after.borrow_inputs),
            ("parallel", before.parallel, after.parallel),
        )
        if before_value != after_value
    )
    report_changed = before.report_json != after.report_json

    phase_diffs = tuple(
        _phase_difference(before_phase, after_phase)
        for before_phase, after_phase in zip(before.phases, after.phases, strict=True)
        if before_phase.sha256 != after_phase.sha256
    )
    changed_phases = tuple(phase_diff.name for phase_diff in phase_diffs)
    first_divergent_phase = changed_phases[0] if changed_phases else None
    equal = not configuration_changes and not report_changed and not phase_diffs

    return CompilerTraceComparison(
        equal=equal,
        configuration_changes=configuration_changes,
        report_changed=report_changed,
        first_divergent_phase=first_divergent_phase,
        changed_phases=changed_phases,
        phase_diffs=phase_diffs,
    )


def compare_trace_files(
    before_path: str | Path,
    after_path: str | Path,
) -> CompilerTraceComparison:
    """Load and compare two UTF-8 compiler-trace JSON files."""
    before = _read_trace_file(before_path, label="before")
    after = _read_trace_file(after_path, label="after")
    return compare_trace_json(before, after)


def main(argv: list[str] | None = None) -> int:
    """Run the trace comparison CLI: 0 equal, 1 different, 2 invalid/unreadable."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare two tiny-tensor-compiler trace JSON snapshots and localize "
            "the first divergent compiler phase."
        )
    )
    parser.add_argument("before", help="path to the baseline compiler trace JSON")
    parser.add_argument("after", help="path to the candidate compiler trace JSON")
    args = parser.parse_args(argv)

    try:
        comparison = compare_trace_files(args.before, args.after)
    except TraceSnapshotError as exc:
        print(f"invalid trace: {exc}", file=sys.stderr)
        return 2

    print(comparison.render())
    return 0 if comparison.equal else 1


def _parse_trace_json(text: str, *, label: str) -> _TraceSnapshot:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TraceSnapshotError(f"{label}: invalid JSON: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise TraceSnapshotError(f"{label}: trace root must be a JSON object")
    keys = set(payload)
    if keys != _TOP_LEVEL_KEYS:
        missing = sorted(_TOP_LEVEL_KEYS - keys)
        extra = sorted(keys - _TOP_LEVEL_KEYS)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise TraceSnapshotError(f"{label}: invalid trace fields ({'; '.join(details)})")

    if payload["format"] != _TRACE_FORMAT:
        raise TraceSnapshotError(f"{label}: unsupported compiler trace format")
    version = payload["version"]
    if type(version) is not int or version != _TRACE_VERSION:
        raise TraceSnapshotError(f"{label}: unsupported compiler trace version {version!r}")
    if type(payload["borrow_inputs"]) is not bool:
        raise TraceSnapshotError(f"{label}: borrow_inputs must be a bool")
    if type(payload["parallel"]) is not bool:
        raise TraceSnapshotError(f"{label}: parallel must be a bool")
    if not isinstance(payload["report"], dict):
        raise TraceSnapshotError(f"{label}: report must be a JSON object")

    raw_phases = payload["phases"]
    if not isinstance(raw_phases, list):
        raise TraceSnapshotError(f"{label}: phases must be a JSON array")

    phases: list[_TraceSnapshotPhase] = []
    for index, raw_phase in enumerate(raw_phases):
        if not isinstance(raw_phase, dict) or set(raw_phase) != _PHASE_KEYS:
            raise TraceSnapshotError(f"{label}: invalid phase object at index {index}")
        name = raw_phase["name"]
        phase_text = raw_phase["text"]
        digest = raw_phase["sha256"]
        if not isinstance(name, str) or not name:
            raise TraceSnapshotError(f"{label}: phase {index} has an invalid name")
        if not isinstance(phase_text, str):
            raise TraceSnapshotError(f"{label}: phase {name!r} text must be a string")
        if not isinstance(digest, str):
            raise TraceSnapshotError(f"{label}: phase {name!r} SHA-256 must be a string")
        expected_digest = hashlib.sha256(phase_text.encode("utf-8")).hexdigest()
        if digest != expected_digest:
            raise TraceSnapshotError(f"{label}: phase {name!r} SHA-256 does not match its text")
        phases.append(_TraceSnapshotPhase(name=name, text=phase_text, sha256=digest))

    phase_names = tuple(phase.name for phase in phases)
    if phase_names != _EXPECTED_PHASES:
        raise TraceSnapshotError(
            f"{label}: invalid compiler trace phase order {phase_names!r}"
        )

    try:
        report_json = json.dumps(
            payload["report"],
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TraceSnapshotError(f"{label}: report is not canonical JSON data") from exc

    return _TraceSnapshot(
        borrow_inputs=payload["borrow_inputs"],
        parallel=payload["parallel"],
        phases=tuple(phases),
        report_json=report_json,
    )


def _phase_difference(
    before: _TraceSnapshotPhase,
    after: _TraceSnapshotPhase,
) -> CompilerTracePhaseDiff:
    if before.name != after.name:
        raise RuntimeError("validated compiler trace phase orders unexpectedly differ")
    unified_diff = "\n".join(
        difflib.unified_diff(
            before.text.splitlines(),
            after.text.splitlines(),
            fromfile=f"before/{before.name}",
            tofile=f"after/{after.name}",
            lineterm="",
        )
    )
    if before.text.endswith("\n") != after.text.endswith("\n"):
        newline_diff = (
            "@@ trailing-newline @@\n"
            f"-before: {before.text.endswith(chr(10))}\n"
            f"+after: {after.text.endswith(chr(10))}"
        )
        unified_diff = f"{unified_diff}\n{newline_diff}" if unified_diff else newline_diff
    return CompilerTracePhaseDiff(
        name=before.name,
        before_sha256=before.sha256,
        after_sha256=after.sha256,
        unified_diff=unified_diff,
    )


def _read_trace_file(path: str | Path, *, label: str) -> str:
    trace_path = Path(path)
    try:
        return trace_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise TraceSnapshotError(f"{label}: cannot read {trace_path}: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
