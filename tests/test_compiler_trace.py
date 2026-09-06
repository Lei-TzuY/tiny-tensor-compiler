import hashlib
import json

import pytest

import tiny_tensor_compiler as ttc
from tiny_tensor_compiler.analysis import analyze_module
from tiny_tensor_compiler.compiler_trace import CompilerTracePhase, trace_module
from tiny_tensor_compiler.frontend import GraphBuilder
from tiny_tensor_compiler.ir import SymbolicDim

_EXPECTED_PHASES = (
    "tensor_ir",
    "buffer_ir",
    "memory_plan",
    "pre_fusion_loop_ir",
    "post_fusion_loop_ir",
    "execution_loop_ir",
    "generated_c",
)


def _module():
    builder = GraphBuilder()
    left = builder.input((8,), dtype="float32")
    right = builder.input((8,), dtype="float32")
    return builder.finish((left + right).relu())


def test_trace_captures_real_pipeline_with_exact_digests():
    module = _module()
    trace = trace_module(module)

    assert tuple(phase.name for phase in trace.phases) == _EXPECTED_PHASES
    assert trace.report == analyze_module(module)
    assert trace.borrow_inputs is False
    assert trace.parallel is False
    for phase in trace.phases:
        assert phase.sha256 == hashlib.sha256(phase.text.encode("utf-8")).hexdigest()
        assert len(phase.sha256) == 64

    payload = json.loads(trace.to_json())
    assert payload["format"] == "tiny-tensor-compiler-trace"
    assert payload["version"] == 1
    assert payload["phases"][0]["name"] == "tensor_ir"
    assert payload["report"] == json.loads(trace.report.to_json())


def test_trace_is_deterministic_for_independently_built_equivalent_modules():
    first = trace_module(_module())
    second = trace_module(_module())

    assert first == second
    assert first.to_json() == second.to_json()


def test_parallel_changes_only_generated_c_phase():
    serial = trace_module(_module(), parallel=False)
    parallel = trace_module(_module(), parallel=True)

    assert serial.report == parallel.report
    for name in _EXPECTED_PHASES[:-1]:
        assert serial.phase(name) == parallel.phase(name)
    assert serial.phase("generated_c").sha256 != parallel.phase("generated_c").sha256
    assert "#pragma omp parallel for schedule(static)" not in serial.phase("generated_c").text
    assert "#pragma omp parallel for schedule(static)" in parallel.phase("generated_c").text


def test_borrowing_changes_only_execution_and_codegen_phases():
    copied = trace_module(_module(), borrow_inputs=False)
    borrowed = trace_module(_module(), borrow_inputs=True)

    assert copied.report == borrowed.report
    for name in _EXPECTED_PHASES[:5]:
        assert copied.phase(name) == borrowed.phase(name)
    assert copied.phase("execution_loop_ir").sha256 != borrowed.phase("execution_loop_ir").sha256
    assert copied.phase("generated_c").sha256 != borrowed.phase("generated_c").sha256
    assert copied.phase("execution_loop_ir").text.startswith("borrowed_inputs: none\n")
    assert borrowed.phase("execution_loop_ir").text.startswith(
        "borrowed_inputs: input0->p"
    )


def test_trace_rejects_unspecialized_symbolic_modules_and_invalid_options():
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    value = builder.input((batch, 4), dtype="float32")
    module = builder.finish(value.relu())

    with pytest.raises(ValueError, match="requires concrete tensor shapes"):
        trace_module(module)
    with pytest.raises(TypeError, match="borrow_inputs must be a bool"):
        trace_module(_module(), borrow_inputs=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="parallel must be a bool"):
        trace_module(_module(), parallel=1)  # type: ignore[arg-type]


def test_trace_phase_lookup_fails_closed():
    phase = CompilerTracePhase.capture("one", "payload")
    assert phase.sha256 == hashlib.sha256(b"payload").hexdigest()

    trace = trace_module(_module())
    with pytest.raises(KeyError, match="no unique phase"):
        trace.phase("missing")


def test_trace_is_exported_from_package_root():
    assert ttc.CompilerTracePhase is CompilerTracePhase
    assert ttc.trace_module is trace_module
    assert isinstance(ttc.trace_module(_module()), ttc.CompilerTrace)
