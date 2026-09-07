import json

import pytest

from tiny_tensor_compiler.analysis import analyze_module
from tiny_tensor_compiler.analysis_diff import (
    CompilerReportValidationError,
    StructuralRegressionPolicy,
    compare_compiler_reports,
    evaluate_structural_regressions,
    main,
    parse_compiler_report,
)
from tiny_tensor_compiler.frontend import GraphBuilder


def _reports():
    baseline_builder = GraphBuilder("analysis-baseline")
    baseline_input = baseline_builder.input((8,), "int32")
    baseline = analyze_module(baseline_builder.finish(baseline_input))

    candidate_builder = GraphBuilder("analysis-candidate")
    candidate_input = candidate_builder.input((8,), "int32")
    candidate = analyze_module(candidate_builder.finish(candidate_input.relu()))
    return baseline, candidate


def test_strict_report_parser_round_trips_canonical_analysis_output():
    report, _ = _reports()

    parsed = parse_compiler_report(report.to_json())

    assert parsed == report
    assert parsed.to_json() == report.to_json()


def test_strict_report_parser_rejects_duplicate_unknown_and_inconsistent_fields():
    report, _ = _reports()
    document = report.to_json()

    duplicate = document[:-1] + ',"version":1}'
    with pytest.raises(CompilerReportValidationError, match="duplicate field"):
        parse_compiler_report(duplicate)

    unknown_payload = json.loads(document)
    unknown_payload["unexpected"] = 1
    with pytest.raises(CompilerReportValidationError, match="unexpected fields"):
        parse_compiler_report(json.dumps(unknown_payload))

    inconsistent_payload = json.loads(document)
    inconsistent_payload["planned_owning_storage_bytes"] += 1
    with pytest.raises(CompilerReportValidationError, match="planned owning storage bytes"):
        parse_compiler_report(json.dumps(inconsistent_payload))

    slot_payload = json.loads(document)
    slot_payload["storage_slots"][0]["byte_count"] += 4
    with pytest.raises(CompilerReportValidationError, match="storage slot byte count"):
        parse_compiler_report(json.dumps(slot_payload))


def test_report_delta_is_deterministic_and_tracks_histogram_and_storage_changes():
    baseline, candidate = _reports()

    delta = compare_compiler_reports(baseline, candidate)

    scalar = {item.metric: item.delta for item in delta.scalar_deltas}
    assert scalar["post_fusion_kernel_count"] == 1
    assert scalar["planned_owning_storage_bytes"] > 0
    assert scalar["physical_storage_count"] == 1

    tensor_ops = {item.name: item.delta for item in delta.tensor_op_deltas}
    assert tensor_ops == {"relu": 1}

    post_kernels = {item.name: item.delta for item in delta.post_fusion_kernel_deltas}
    assert post_kernels == {"relu": 1}

    assert delta.to_json() == compare_compiler_reports(baseline, candidate).to_json()


def test_structural_regression_policy_is_baseline_relative_and_fail_closed():
    baseline, candidate = _reports()
    delta = compare_compiler_reports(baseline, candidate)

    regressions = evaluate_structural_regressions(
        delta,
        StructuralRegressionPolicy(
            max_planned_storage_bytes_increase=0,
            max_post_fusion_kernel_increase=0,
        ),
    )

    assert tuple(item.metric for item in regressions) == (
        "planned_owning_storage_bytes",
        "post_fusion_kernel_count",
    )
    assert all(item.actual_increase > item.allowed_increase for item in regressions)

    assert not evaluate_structural_regressions(
        compare_compiler_reports(candidate, baseline),
        StructuralRegressionPolicy(),
    )

    with pytest.raises(TypeError, match="non-negative integer"):
        StructuralRegressionPolicy(max_post_fusion_kernel_increase=True)
    with pytest.raises(ValueError, match="non-negative integer"):
        StructuralRegressionPolicy(max_planned_storage_bytes_increase=-1)


def test_analysis_diff_cli_has_stable_success_regression_and_invalid_exit_codes(
    tmp_path, capsys
):
    baseline, candidate = _reports()
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    invalid_path = tmp_path / "invalid.json"
    baseline_path.write_text(baseline.to_json(), encoding="utf-8")
    candidate_path.write_text(candidate.to_json(), encoding="utf-8")
    invalid_path.write_text("{}", encoding="utf-8")

    assert main([str(candidate_path), str(baseline_path)]) == 0
    assert "structural regression gate: pass" in capsys.readouterr().out

    assert main([str(baseline_path), str(candidate_path)]) == 1
    regression_output = capsys.readouterr().out
    assert "structural regression gate: fail" in regression_output
    assert "planned_owning_storage_bytes" in regression_output
    assert "post_fusion_kernel_count" in regression_output

    assert main([str(invalid_path), str(candidate_path)]) == 2
    assert "invalid compiler report" in capsys.readouterr().err
