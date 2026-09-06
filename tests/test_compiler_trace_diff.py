import json

import pytest

from tiny_tensor_compiler.compiler_trace import trace_module
from tiny_tensor_compiler.frontend import GraphBuilder
from tiny_tensor_compiler.trace_diff import (
    TraceSnapshotError,
    compare_trace_files,
    compare_trace_json,
    main,
)


def _module():
    builder = GraphBuilder()
    left = builder.input((8,), dtype="float32")
    right = builder.input((8,), dtype="float32")
    return builder.finish((left + right).relu())


def test_equal_trace_content_ignores_json_formatting():
    compact = trace_module(_module()).to_json()
    pretty = json.dumps(json.loads(compact), ensure_ascii=False, indent=2)

    comparison = compare_trace_json(compact, pretty)

    assert comparison.equal is True
    assert comparison.first_divergent_phase is None
    assert comparison.changed_phases == ()
    assert comparison.configuration_changes == ()
    assert comparison.report_changed is False
    assert comparison.render() == "equal"


def test_parallel_trace_diff_localizes_generated_c():
    serial = trace_module(_module(), parallel=False)
    parallel = trace_module(_module(), parallel=True)

    comparison = compare_trace_json(serial.to_json(), parallel.to_json())

    assert comparison.equal is False
    assert comparison.configuration_changes == ("parallel",)
    assert comparison.report_changed is False
    assert comparison.first_divergent_phase == "generated_c"
    assert comparison.changed_phases == ("generated_c",)
    rendered = comparison.render()
    assert "first divergent phase: generated_c" in rendered
    assert "--- before/generated_c" in rendered
    assert "+++ after/generated_c" in rendered
    assert "#pragma omp parallel for schedule(static)" in rendered


def test_borrowed_input_trace_diff_localizes_execution_boundary():
    copied = trace_module(_module(), borrow_inputs=False)
    borrowed = trace_module(_module(), borrow_inputs=True)

    comparison = compare_trace_json(copied.to_json(), borrowed.to_json())

    assert comparison.configuration_changes == ("borrow_inputs",)
    assert comparison.report_changed is False
    assert comparison.first_divergent_phase == "execution_loop_ir"
    assert comparison.changed_phases == ("execution_loop_ir", "generated_c")
    assert "borrowed_inputs: none" in comparison.phase_diffs[0].unified_diff
    assert "borrowed_inputs: input0->p" in comparison.phase_diffs[0].unified_diff


def test_trace_diff_rejects_tampered_digest_and_invalid_structure():
    payload = json.loads(trace_module(_module()).to_json())
    payload["phases"][0]["text"] += "\ntampered"
    with pytest.raises(TraceSnapshotError, match="SHA-256"):
        compare_trace_json(json.dumps(payload), trace_module(_module()).to_json())

    payload = json.loads(trace_module(_module()).to_json())
    payload["phases"][0], payload["phases"][1] = payload["phases"][1], payload["phases"][0]
    with pytest.raises(TraceSnapshotError, match="phase order"):
        compare_trace_json(json.dumps(payload), trace_module(_module()).to_json())

    payload = json.loads(trace_module(_module()).to_json())
    payload["version"] = 2
    with pytest.raises(TraceSnapshotError, match="unsupported compiler trace version"):
        compare_trace_json(json.dumps(payload), trace_module(_module()).to_json())


def test_trace_diff_reports_tensor_ir_and_report_changes_for_different_modules():
    first_builder = GraphBuilder()
    first = first_builder.input((4,), dtype="float32")
    first_module = first_builder.finish(first.relu())

    second_builder = GraphBuilder()
    lhs = second_builder.input((4,), dtype="float32")
    rhs = second_builder.input((4,), dtype="float32")
    second_module = second_builder.finish((lhs + rhs).relu())

    comparison = compare_trace_json(
        trace_module(first_module).to_json(),
        trace_module(second_module).to_json(),
    )

    assert comparison.report_changed is True
    assert comparison.first_divergent_phase == "tensor_ir"
    assert comparison.changed_phases[0] == "tensor_ir"


def test_trace_diff_file_workflow_and_cli_exit_codes(tmp_path, capsys):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(trace_module(_module()).to_json(), encoding="utf-8")
    after.write_text(trace_module(_module()).to_json(), encoding="utf-8")

    comparison = compare_trace_files(before, after)
    assert comparison.equal is True
    assert main([str(before), str(after)]) == 0
    assert capsys.readouterr().out.strip() == "equal"

    after.write_text(trace_module(_module(), parallel=True).to_json(), encoding="utf-8")
    assert main([str(before), str(after)]) == 1
    different_output = capsys.readouterr().out
    assert "different" in different_output
    assert "first divergent phase: generated_c" in different_output

    tampered = json.loads(before.read_text(encoding="utf-8"))
    tampered["phases"][-1]["text"] += "\nchanged"
    after.write_text(json.dumps(tampered), encoding="utf-8")
    assert main([str(before), str(after)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid trace:" in captured.err
