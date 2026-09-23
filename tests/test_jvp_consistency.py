from __future__ import annotations

import numpy as np
import pytest

from tiny_tensor_compiler.jvp_consistency import (
    generate_jvp_consistency_case,
    run_jvp_consistency_campaign,
)
from tiny_tensor_compiler.repro import load_repro_case


def _zero_jvp_runner(_module, _inputs, tangent):
    return np.zeros_like(tangent)


def _failing_vjp_runner(_module, _inputs, _cotangent):
    raise RuntimeError("synthetic deterministic VJP failure")


def test_jvp_consistency_campaign_passes_directional_and_duality_checks():
    result = run_jvp_consistency_campaign(start_seed=0, cases=16)

    assert result.passed
    assert result.checked_cases == 16
    assert result.failure is None


def test_jvp_consistency_case_generation_is_deterministic_and_replayable():
    first = generate_jvp_consistency_case(19)
    second = generate_jvp_consistency_case(19)

    assert first.seed == second.seed == 19
    assert first.primal_repro == second.primal_repro
    np.testing.assert_array_equal(first.tangent, second.tangent)
    np.testing.assert_array_equal(first.cotangent, second.cotangent)

    repro = load_repro_case(first.primal_repro)
    assert repro.module.function.name == "jvp_consistency"
    assert len(repro.inputs) == 2
    assert repro.inputs[0].dtype == np.dtype(np.float64)
    assert first.tangent.shape == repro.inputs[0].shape
    assert first.cotangent.shape == repro.expected_outputs[0].shape


def test_jvp_consistency_failure_shrinks_to_minimal_directional_repro():
    result = run_jvp_consistency_campaign(
        start_seed=7,
        cases=1,
        jvp_runner=_zero_jvp_runner,
    )

    assert not result.passed
    failure = result.failure
    assert failure is not None
    assert failure.signature == "finite-difference:mismatch:value"
    assert failure.original_operation_count >= 1
    assert failure.minimized_operation_count == 0
    assert failure.shrink_evaluations > 0

    minimized = failure.minimized_case
    repro = load_repro_case(minimized.primal_repro)
    assert repro.inputs[0].shape == (1, 1)
    assert minimized.tangent.shape == (1, 1)
    assert [op.opcode for op in repro.module.function.ops] == [
        "input",
        "input",
        "return",
    ]


def test_jvp_consistency_preserves_vjp_exception_signature():
    result = run_jvp_consistency_campaign(
        start_seed=3,
        cases=1,
        vjp_runner=_failing_vjp_runner,
    )

    failure = result.failure
    assert failure is not None
    assert failure.signature == "vjp:exception:builtins.RuntimeError"
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
def test_jvp_consistency_configuration_fails_closed(kwargs, error, match):
    with pytest.raises(error, match=match):
        run_jvp_consistency_campaign(**kwargs)
