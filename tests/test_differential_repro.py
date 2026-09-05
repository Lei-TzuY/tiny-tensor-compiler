from __future__ import annotations

import numpy as np
import pytest

from tiny_tensor_compiler import execute_reference
from tiny_tensor_compiler.differential import (
    generate_differential_case,
    run_differential_campaign,
)
from tiny_tensor_compiler.repro import (
    load_repro_case,
    replay_repro_case,
    repro_case_sha256,
)


def test_same_seed_generates_identical_canonical_repro_artifact():
    first = generate_differential_case(5)
    second = generate_differential_case(5)
    case = load_repro_case(first)

    assert first == second
    assert repro_case_sha256(first) == repro_case_sha256(second)
    assert [op.opcode for op in case.module.function.ops] == [
        "input",
        "input",
        "transpose",
        "transpose",
        "reverse",
        "mul",
        "reverse",
        "add",
        "return",
    ]
    assert case.inputs[0].shape == (4, 4)
    assert case.inputs[0].dtype == np.dtype(np.int32)


def test_generated_grammar_stays_on_stable_pure_non_reduction_surface():
    allowed = {
        "input",
        "add",
        "mul",
        "relu",
        "reverse",
        "transpose",
        "view",
        "reshape",
        "return",
    }

    for seed in range(16):
        case = load_repro_case(generate_differential_case(seed))
        assert {op.opcode for op in case.module.function.ops} <= allowed
        assert "sum" not in {op.opcode for op in case.module.function.ops}
        assert "copy_into" not in {op.opcode for op in case.module.function.ops}


def test_reference_candidate_campaign_is_deterministically_clean():
    def reference_runner(module, inputs):
        return execute_reference(module, inputs=inputs)

    first = run_differential_campaign(
        start_seed=0,
        cases=12,
        candidate_runner=reference_runner,
    )
    second = run_differential_campaign(
        start_seed=0,
        cases=12,
        candidate_runner=reference_runner,
    )

    assert first == second
    assert first.passed
    assert first.checked_cases == 12
    assert first.failure is None


def test_seed_four_replays_directly_through_native_backend():
    document = generate_differential_case(4)

    replay_repro_case(document, backend="native")


def test_native_campaign_executes_generated_views_and_elementwise_graphs():
    result = run_differential_campaign(start_seed=4, cases=3)

    signature = None if result.failure is None else result.failure.signature
    assert result.passed, f"unexpected differential failure: {signature}"
    assert result.checked_cases == 3
    assert result.failure is None


def test_mismatch_shrinks_deterministically_to_same_canonical_repro():
    def wrong_shape_runner(_module, _inputs):
        return np.array(0, dtype=np.int32)

    first = run_differential_campaign(
        start_seed=5,
        cases=1,
        candidate_runner=wrong_shape_runner,
    )
    second = run_differential_campaign(
        start_seed=5,
        cases=1,
        candidate_runner=wrong_shape_runner,
    )

    assert first == second
    assert not first.passed
    assert first.failure is not None
    failure = first.failure
    assert failure.seed == 5
    assert failure.signature == "mismatch:shape:0"
    assert failure.original_operation_count == 6
    assert failure.minimized_operation_count == 0
    assert failure.shrink_evaluations > 0
    assert failure.original_repro != failure.minimized_repro

    minimized = load_repro_case(failure.minimized_repro)
    assert [op.opcode for op in minimized.module.function.ops] == ["input", "input", "return"]
    assert minimized.inputs[0].shape == (0, 0)
    assert minimized.inputs[1].shape == (0, 0)
    assert repro_case_sha256(failure.minimized_repro) == repro_case_sha256(
        second.failure.minimized_repro
    )


def test_candidate_exception_is_a_stable_shrink_signature():
    def failing_runner(_module, _inputs):
        raise RuntimeError("synthetic candidate failure")

    result = run_differential_campaign(
        start_seed=4,
        cases=1,
        candidate_runner=failing_runner,
    )

    assert result.failure is not None
    assert result.failure.signature == "exception:builtins.RuntimeError"
    minimized = load_repro_case(result.failure.minimized_repro)
    assert [op.opcode for op in minimized.module.function.ops] == ["input", "input", "return"]
    assert minimized.inputs[0].shape == (0, 0)


def test_campaign_configuration_fails_closed():
    with pytest.raises(TypeError, match="seed"):
        generate_differential_case(True)
    with pytest.raises(ValueError, match="64-bit"):
        generate_differential_case(-1)
    with pytest.raises(TypeError, match="cases"):
        run_differential_campaign(start_seed=0, cases=1.5)
    with pytest.raises(ValueError, match="positive"):
        run_differential_campaign(start_seed=0, cases=0)
    with pytest.raises(ValueError, match="64-bit"):
        run_differential_campaign(start_seed=(1 << 64) - 1, cases=2)

    def reference_runner(module, inputs):
        return execute_reference(module, inputs=inputs)

    with pytest.raises(ValueError, match="default native"):
        run_differential_campaign(
            start_seed=0,
            cases=1,
            candidate_runner=reference_runner,
            parallel=True,
        )
