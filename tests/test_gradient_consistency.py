from __future__ import annotations

import numpy as np
import pytest

from tiny_tensor_compiler.gradient_consistency import (
    generate_gradient_consistency_case,
    run_gradient_consistency_campaign,
)
from tiny_tensor_compiler.repro import load_repro_case


def _zero_gradient_runner(_module, inputs):
    return np.zeros_like(inputs[0])


def _failing_gradient_runner(_module, _inputs):
    raise RuntimeError("synthetic deterministic gradient failure")


def test_gradient_consistency_campaign_passes_smooth_deterministic_cases():
    result = run_gradient_consistency_campaign(start_seed=0, cases=12)

    assert result.passed
    assert result.checked_cases == 12
    assert result.failure is None


def test_gradient_consistency_case_generation_is_canonical_and_replayable():
    first = generate_gradient_consistency_case(19)
    second = generate_gradient_consistency_case(19)

    assert first == second
    repro = load_repro_case(first)
    assert repro.module.function.name == "gradient_consistency"
    assert len(repro.inputs) == 2
    assert repro.inputs[0].dtype == np.dtype(np.float64)
    assert repro.inputs[0].shape == repro.inputs[1].shape


def test_gradient_consistency_failure_shrinks_to_minimal_repro():
    result = run_gradient_consistency_campaign(
        start_seed=7,
        cases=1,
        candidate_runner=_zero_gradient_runner,
    )

    assert not result.passed
    assert result.checked_cases == 1
    failure = result.failure
    assert failure is not None
    assert failure.signature == "mismatch:value"
    assert failure.original_operation_count >= 1
    assert failure.minimized_operation_count == 0
    assert failure.shrink_evaluations > 0

    minimized = load_repro_case(failure.minimized_repro)
    assert minimized.inputs[0].shape == (1, 1)
    assert [op.opcode for op in minimized.module.function.ops] == [
        "input",
        "input",
        "sum",
        "return",
    ]


def test_gradient_consistency_preserves_candidate_exception_signature():
    result = run_gradient_consistency_campaign(
        start_seed=3,
        cases=1,
        candidate_runner=_failing_gradient_runner,
    )

    failure = result.failure
    assert failure is not None
    assert failure.signature == "exception:builtins.RuntimeError"
    assert failure.minimized_operation_count == 0


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"start_seed": True, "cases": 1}, TypeError, "seed"),
        ({"start_seed": 0, "cases": True}, TypeError, "cases"),
        ({"start_seed": 0, "cases": 0}, ValueError, "positive"),
        ({"start_seed": 0, "cases": 1, "step": 0.0}, ValueError, "step"),
        ({"start_seed": 0, "cases": 1, "atol": -1.0}, ValueError, "atol"),
        ({"start_seed": 0, "cases": 1, "rtol": -1.0}, ValueError, "rtol"),
    ],
)
def test_gradient_consistency_configuration_fails_closed(kwargs, error, match):
    with pytest.raises(error, match=match):
        run_gradient_consistency_campaign(**kwargs)
