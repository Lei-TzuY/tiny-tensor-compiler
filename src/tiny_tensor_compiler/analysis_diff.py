from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .analysis import CompilerReport, StorageSlotReport

_DELTA_FORMAT = "tiny-tensor-compiler-report-delta"
_DELTA_VERSION = 1
_REPORT_FORMAT = "tiny-tensor-compiler-report"
_REPORT_VERSION = 1
_DTYPE_BYTES = {"i32": 4, "i64": 8, "f32": 4, "f64": 8}

_SCALAR_METRICS = (
    "input_count",
    "output_count",
    "logical_value_count",
    "logical_tensor_bytes",
    "physical_storage_count",
    "planned_owning_storage_bytes",
    "alias_value_count",
    "view_count",
    "pre_fusion_kernel_count",
    "post_fusion_kernel_count",
    "fused_kernel_count",
    "kernels_eliminated_by_fusion",
)

_REPORT_KEYS = {
    "function_name",
    "input_count",
    "output_count",
    "tensor_op_counts",
    "logical_value_count",
    "logical_tensor_bytes",
    "physical_storage_count",
    "planned_owning_storage_bytes",
    "storage_slots",
    "alias_value_count",
    "view_count",
    "effect_counts",
    "pre_fusion_kernel_count",
    "pre_fusion_kernel_counts",
    "post_fusion_kernel_count",
    "post_fusion_kernel_counts",
    "fused_kernel_count",
    "kernels_eliminated_by_fusion",
    "format",
    "version",
}


class CompilerReportValidationError(ValueError):
    """Raised when a serialized compiler report is malformed or internally inconsistent."""


@dataclass(frozen=True)
class ScalarDelta:
    metric: str
    before: int
    after: int
    delta: int


@dataclass(frozen=True)
class HistogramDelta:
    name: str
    before: int
    after: int
    delta: int


@dataclass(frozen=True)
class StorageSlotDelta:
    slot: int
    before: StorageSlotReport | None
    after: StorageSlotReport | None


@dataclass(frozen=True)
class CompilerReportDelta:
    """Deterministic field-level structural delta between two validated reports."""

    before_function_name: str
    after_function_name: str
    scalar_deltas: tuple[ScalarDelta, ...]
    tensor_op_deltas: tuple[HistogramDelta, ...]
    effect_deltas: tuple[HistogramDelta, ...]
    pre_fusion_kernel_deltas: tuple[HistogramDelta, ...]
    post_fusion_kernel_deltas: tuple[HistogramDelta, ...]
    storage_slot_deltas: tuple[StorageSlotDelta, ...]
    format: str = _DELTA_FORMAT
    version: int = _DELTA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "after_function_name": self.after_function_name,
            "before_function_name": self.before_function_name,
            "effect_deltas": [_histogram_delta_dict(item) for item in self.effect_deltas],
            "format": self.format,
            "post_fusion_kernel_deltas": [
                _histogram_delta_dict(item) for item in self.post_fusion_kernel_deltas
            ],
            "pre_fusion_kernel_deltas": [
                _histogram_delta_dict(item) for item in self.pre_fusion_kernel_deltas
            ],
            "scalar_deltas": [_scalar_delta_dict(item) for item in self.scalar_deltas],
            "storage_slot_deltas": [
                {
                    "after": _storage_slot_dict(item.after),
                    "before": _storage_slot_dict(item.before),
                    "slot": item.slot,
                }
                for item in self.storage_slot_deltas
            ],
            "tensor_op_deltas": [
                _histogram_delta_dict(item) for item in self.tensor_op_deltas
            ],
            "version": self.version,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class StructuralRegressionPolicy:
    """Baseline-relative structural limits; these are not runtime performance budgets."""

    max_planned_storage_bytes_increase: int = 0
    max_post_fusion_kernel_increase: int = 0

    def __post_init__(self) -> None:
        _validate_limit(
            self.max_planned_storage_bytes_increase,
            "max_planned_storage_bytes_increase",
        )
        _validate_limit(
            self.max_post_fusion_kernel_increase,
            "max_post_fusion_kernel_increase",
        )


@dataclass(frozen=True)
class StructuralRegression:
    metric: str
    allowed_increase: int
    actual_increase: int


def parse_compiler_report(document: str) -> CompilerReport:
    """Parse one strict version-1 compiler report and verify its structural invariants."""
    if not isinstance(document, str):
        raise TypeError("parse_compiler_report requires a JSON string")
    try:
        payload = json.loads(document, object_pairs_hook=_object_without_duplicates)
    except json.JSONDecodeError as exc:
        raise CompilerReportValidationError(f"invalid compiler report JSON: {exc.msg}") from exc

    root = _require_mapping(payload, "compiler report")
    _require_exact_keys(root, _REPORT_KEYS, "compiler report")
    if root["format"] != _REPORT_FORMAT:
        raise CompilerReportValidationError(
            f"unsupported compiler report format: {root['format']!r}"
        )
    version = _plain_nonnegative_int(root["version"], "compiler report version")
    if version != _REPORT_VERSION:
        raise CompilerReportValidationError(f"unsupported compiler report version: {version}")

    function_name = root["function_name"]
    if not isinstance(function_name, str):
        raise CompilerReportValidationError("function_name must be a string")

    tensor_op_counts = _decode_histogram(root["tensor_op_counts"], "tensor_op_counts")
    effect_counts = _decode_histogram(root["effect_counts"], "effect_counts")
    pre_kernel_counts = _decode_histogram(
        root["pre_fusion_kernel_counts"], "pre_fusion_kernel_counts"
    )
    post_kernel_counts = _decode_histogram(
        root["post_fusion_kernel_counts"], "post_fusion_kernel_counts"
    )
    storage_slots = _decode_storage_slots(root["storage_slots"])

    values = {
        key: _plain_nonnegative_int(root[key], key)
        for key in _SCALAR_METRICS
    }

    if values["physical_storage_count"] != len(storage_slots):
        raise CompilerReportValidationError(
            "physical_storage_count does not match the number of storage slots"
        )
    planned_bytes = sum(slot.byte_count for slot in storage_slots)
    if values["planned_owning_storage_bytes"] != planned_bytes:
        raise CompilerReportValidationError(
            "planned owning storage bytes do not match storage slot byte counts"
        )
    if values["pre_fusion_kernel_count"] != sum(count for _, count in pre_kernel_counts):
        raise CompilerReportValidationError(
            "pre_fusion_kernel_count does not match its kernel histogram"
        )
    if values["post_fusion_kernel_count"] != sum(count for _, count in post_kernel_counts):
        raise CompilerReportValidationError(
            "post_fusion_kernel_count does not match its kernel histogram"
        )
    if values["fused_kernel_count"] > values["post_fusion_kernel_count"]:
        raise CompilerReportValidationError(
            "fused_kernel_count cannot exceed post_fusion_kernel_count"
        )
    eliminated = values["pre_fusion_kernel_count"] - values["post_fusion_kernel_count"]
    if eliminated < 0 or values["kernels_eliminated_by_fusion"] != eliminated:
        raise CompilerReportValidationError(
            "kernels_eliminated_by_fusion does not match pre/post fusion kernel counts"
        )

    return CompilerReport(
        function_name=function_name,
        input_count=values["input_count"],
        output_count=values["output_count"],
        tensor_op_counts=tensor_op_counts,
        logical_value_count=values["logical_value_count"],
        logical_tensor_bytes=values["logical_tensor_bytes"],
        physical_storage_count=values["physical_storage_count"],
        planned_owning_storage_bytes=values["planned_owning_storage_bytes"],
        storage_slots=storage_slots,
        alias_value_count=values["alias_value_count"],
        view_count=values["view_count"],
        effect_counts=effect_counts,
        pre_fusion_kernel_count=values["pre_fusion_kernel_count"],
        pre_fusion_kernel_counts=pre_kernel_counts,
        post_fusion_kernel_count=values["post_fusion_kernel_count"],
        post_fusion_kernel_counts=post_kernel_counts,
        fused_kernel_count=values["fused_kernel_count"],
        kernels_eliminated_by_fusion=values["kernels_eliminated_by_fusion"],
    )


def compare_compiler_reports(
    before: CompilerReport,
    after: CompilerReport,
) -> CompilerReportDelta:
    """Return a deterministic structural delta without assigning performance meaning."""
    _require_report(before, "before")
    _require_report(after, "after")
    scalar_deltas = tuple(
        ScalarDelta(
            metric=metric,
            before=getattr(before, metric),
            after=getattr(after, metric),
            delta=getattr(after, metric) - getattr(before, metric),
        )
        for metric in _SCALAR_METRICS
    )
    return CompilerReportDelta(
        before_function_name=before.function_name,
        after_function_name=after.function_name,
        scalar_deltas=scalar_deltas,
        tensor_op_deltas=_histogram_delta(before.tensor_op_counts, after.tensor_op_counts),
        effect_deltas=_histogram_delta(before.effect_counts, after.effect_counts),
        pre_fusion_kernel_deltas=_histogram_delta(
            before.pre_fusion_kernel_counts,
            after.pre_fusion_kernel_counts,
        ),
        post_fusion_kernel_deltas=_histogram_delta(
            before.post_fusion_kernel_counts,
            after.post_fusion_kernel_counts,
        ),
        storage_slot_deltas=_storage_slot_delta(before.storage_slots, after.storage_slots),
    )


def evaluate_structural_regressions(
    delta: CompilerReportDelta,
    policy: StructuralRegressionPolicy,
) -> tuple[StructuralRegression, ...]:
    """Evaluate baseline-relative storage/kernel increases under an explicit policy."""
    if not isinstance(delta, CompilerReportDelta):
        raise TypeError("delta must be a CompilerReportDelta")
    if not isinstance(policy, StructuralRegressionPolicy):
        raise TypeError("policy must be a StructuralRegressionPolicy")

    scalar = {item.metric: item.delta for item in delta.scalar_deltas}
    checks = (
        (
            "planned_owning_storage_bytes",
            policy.max_planned_storage_bytes_increase,
        ),
        (
            "post_fusion_kernel_count",
            policy.max_post_fusion_kernel_increase,
        ),
    )
    regressions = []
    for metric, allowed in checks:
        actual = max(0, scalar[metric])
        if actual > allowed:
            regressions.append(
                StructuralRegression(
                    metric=metric,
                    allowed_increase=allowed,
                    actual_increase=actual,
                )
            )
    return tuple(regressions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare deterministic compiler reports under structural regression limits."
    )
    parser.add_argument("before", help="baseline CompilerReport JSON")
    parser.add_argument("after", help="candidate CompilerReport JSON")
    parser.add_argument(
        "--max-storage-increase",
        type=int,
        default=0,
        help="allowed increase in planned compiler-owned storage bytes (default: 0)",
    )
    parser.add_argument(
        "--max-kernel-increase",
        type=int,
        default=0,
        help="allowed increase in post-fusion kernel count (default: 0)",
    )
    args = parser.parse_args(argv)

    try:
        before = parse_compiler_report(Path(args.before).read_text(encoding="utf-8"))
        after = parse_compiler_report(Path(args.after).read_text(encoding="utf-8"))
    except (CompilerReportValidationError, OSError, UnicodeError) as exc:
        print(f"invalid compiler report: {exc}", file=sys.stderr)
        return 2

    try:
        policy = StructuralRegressionPolicy(
            max_planned_storage_bytes_increase=args.max_storage_increase,
            max_post_fusion_kernel_increase=args.max_kernel_increase,
        )
    except (TypeError, ValueError) as exc:
        print(f"invalid structural regression policy: {exc}", file=sys.stderr)
        return 2

    delta = compare_compiler_reports(before, after)
    regressions = evaluate_structural_regressions(delta, policy)
    if not regressions:
        print("structural regression gate: pass")
        print(delta.to_json())
        return 0

    print("structural regression gate: fail")
    for regression in regressions:
        print(
            f"{regression.metric}: increase {regression.actual_increase} "
            f"exceeds allowed {regression.allowed_increase}"
        )
    print(delta.to_json())
    return 1


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CompilerReportValidationError(f"duplicate field: {key!r}")
        result[key] = value
    return result


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CompilerReportValidationError(f"{context} must be an object")
    return value


def _require_exact_keys(record: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(record)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    details = []
    if missing:
        details.append(f"missing fields {missing}")
    if unexpected:
        details.append(f"unexpected fields {unexpected}")
    raise CompilerReportValidationError(f"{context} has " + "; ".join(details))


def _plain_nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompilerReportValidationError(f"{context} must be a non-negative integer")
    if value < 0:
        raise CompilerReportValidationError(f"{context} must be a non-negative integer")
    return value


def _decode_histogram(raw: Any, context: str) -> tuple[tuple[str, int], ...]:
    if not isinstance(raw, list):
        raise CompilerReportValidationError(f"{context} must be a list")
    result = []
    previous: str | None = None
    for index, item in enumerate(raw):
        if not isinstance(item, list) or len(item) != 2:
            raise CompilerReportValidationError(
                f"{context} item #{index} must be [name, count]"
            )
        name, raw_count = item
        if not isinstance(name, str) or not name:
            raise CompilerReportValidationError(f"{context} item #{index} name must be non-empty")
        count = _plain_nonnegative_int(raw_count, f"{context} item #{index} count")
        if count == 0:
            raise CompilerReportValidationError(f"{context} item #{index} count must be positive")
        if previous is not None and name <= previous:
            raise CompilerReportValidationError(f"{context} names must be strictly sorted and unique")
        result.append((name, count))
        previous = name
    return tuple(result)


def _decode_storage_slots(raw: Any) -> tuple[StorageSlotReport, ...]:
    if not isinstance(raw, list):
        raise CompilerReportValidationError("storage_slots must be a list")
    result = []
    for expected_slot, raw_slot in enumerate(raw):
        slot_record = _require_mapping(raw_slot, f"storage slot #{expected_slot}")
        _require_exact_keys(
            slot_record,
            {"slot", "shape", "dtype", "byte_count"},
            f"storage slot #{expected_slot}",
        )
        slot = _plain_nonnegative_int(slot_record["slot"], f"storage slot #{expected_slot} id")
        if slot != expected_slot:
            raise CompilerReportValidationError("storage slot ids must be dense starting at zero")
        raw_shape = slot_record["shape"]
        if not isinstance(raw_shape, list):
            raise CompilerReportValidationError(f"storage slot #{slot} shape must be a list")
        shape = tuple(
            _plain_nonnegative_int(dim, f"storage slot #{slot} shape dimension #{axis}")
            for axis, dim in enumerate(raw_shape)
        )
        dtype = slot_record["dtype"]
        if dtype not in _DTYPE_BYTES:
            raise CompilerReportValidationError(f"storage slot #{slot} has unsupported dtype")
        byte_count = _plain_nonnegative_int(
            slot_record["byte_count"], f"storage slot #{slot} byte_count"
        )
        expected_bytes = math.prod(shape) * _DTYPE_BYTES[dtype]
        if byte_count != expected_bytes:
            raise CompilerReportValidationError(
                f"storage slot byte count for p{slot} does not match shape/dtype"
            )
        result.append(
            StorageSlotReport(
                slot=slot,
                shape=shape,
                dtype=dtype,
                byte_count=byte_count,
            )
        )
    return tuple(result)


def _histogram_delta(
    before: tuple[tuple[str, int], ...],
    after: tuple[tuple[str, int], ...],
) -> tuple[HistogramDelta, ...]:
    before_map = dict(before)
    after_map = dict(after)
    result = []
    for name in sorted(before_map.keys() | after_map.keys()):
        before_count = before_map.get(name, 0)
        after_count = after_map.get(name, 0)
        if before_count == after_count:
            continue
        result.append(
            HistogramDelta(
                name=name,
                before=before_count,
                after=after_count,
                delta=after_count - before_count,
            )
        )
    return tuple(result)


def _storage_slot_delta(
    before: tuple[StorageSlotReport, ...],
    after: tuple[StorageSlotReport, ...],
) -> tuple[StorageSlotDelta, ...]:
    result = []
    for slot in range(max(len(before), len(after))):
        before_slot = before[slot] if slot < len(before) else None
        after_slot = after[slot] if slot < len(after) else None
        if before_slot == after_slot:
            continue
        result.append(StorageSlotDelta(slot=slot, before=before_slot, after=after_slot))
    return tuple(result)


def _require_report(report: CompilerReport, context: str) -> None:
    if not isinstance(report, CompilerReport):
        raise TypeError(f"{context} must be a CompilerReport")
    parse_compiler_report(report.to_json())


def _validate_limit(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _scalar_delta_dict(item: ScalarDelta) -> dict[str, Any]:
    return {
        "after": item.after,
        "before": item.before,
        "delta": item.delta,
        "metric": item.metric,
    }


def _histogram_delta_dict(item: HistogramDelta) -> dict[str, Any]:
    return {
        "after": item.after,
        "before": item.before,
        "delta": item.delta,
        "name": item.name,
    }


def _storage_slot_dict(slot: StorageSlotReport | None) -> dict[str, Any] | None:
    if slot is None:
        return None
    return {
        "byte_count": slot.byte_count,
        "dtype": slot.dtype,
        "shape": list(slot.shape),
        "slot": slot.slot,
    }


if __name__ == "__main__":
    raise SystemExit(main())
