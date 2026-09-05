from __future__ import annotations

import numpy as np
import pytest

from tiny_tensor_compiler import execute_reference
from tiny_tensor_compiler import verification_coverage as coverage


def test_measured_coverage_records_generator_lowering_layout_and_relation_features():
    observation = coverage.measure_structural_coverage(5)
    features = set(observation.features)

    assert observation.seed == 5
    assert observation.features == tuple(sorted(set(observation.features)))
    assert {
        "dtype:i32",
        "extent:multi",
        "generator-op:transpose",
        "generator-op:reverse0",
        "generator-transition:transpose->transpose",
        "tensor-op:transpose",
        "tensor-op:reverse",
        "loop:view",
        "layout:negative-stride",
        "layout:noncontiguous-view",
        "metamorphic-relation:double_reverse_axis1",
    } <= features


def test_greedy_selection_maximizes_new_features_then_uses_lower_seed(monkeypatch):
    observations = {
        10: coverage.StructuralCoverageObservation(10, ("a",)),
        11: coverage.StructuralCoverageObservation(11, ("a",)),
        12: coverage.StructuralCoverageObservation(12, ("b", "c")),
        13: coverage.StructuralCoverageObservation(13, ("d",)),
    }

    monkeypatch.setattr(
        coverage,
        "measure_structural_coverage",
        lambda seed: observations[seed],
    )

    selection = coverage.select_structural_coverage_seeds(
        start_seed=10,
        candidate_cases=4,
        budget=3,
    )

    assert selection.selected_seeds == (12, 10, 13)
    assert selection.candidate_features == ("a", "b", "c", "d")
    assert selection.covered_features == selection.candidate_features
    assert selection.uncovered_features == ()


def test_real_selection_is_deterministic_and_budgeted():
    first = coverage.select_structural_coverage_seeds(
        start_seed=0,
        candidate_cases=20,
        budget=5,
    )
    second = coverage.select_structural_coverage_seeds(
        start_seed=0,
        candidate_cases=20,
        budget=5,
    )

    assert first == second
    assert len(first.selected) == 5
    assert len(set(first.selected_seeds)) == 5
    assert all(0 <= seed < 20 for seed in first.selected_seeds)
    assert set(first.covered_features) <= set(first.candidate_features)
    assert tuple(sorted(first.candidate_features)) == first.candidate_features
    assert tuple(sorted(first.covered_features)) == first.covered_features


def test_coverage_guided_reference_oracles_execute_exact_selected_seed_order():
    def reference_candidate(module, inputs):
        return execute_reference(module, inputs=inputs)

    def reference_configuration(configuration, module, inputs):
        return execute_reference(module, inputs=inputs)

    differential = coverage.run_coverage_guided_differential_campaign(
        start_seed=0,
        candidate_cases=12,
        budget=3,
        candidate_runner=reference_candidate,
    )
    metamorphic = coverage.run_coverage_guided_metamorphic_campaign(
        start_seed=0,
        candidate_cases=12,
        budget=3,
        candidate_runner=reference_candidate,
    )
    configuration = coverage.run_coverage_guided_configuration_campaign(
        start_seed=0,
        candidate_cases=12,
        budget=3,
        configuration_runner=reference_configuration,
    )

    for result, oracle in (
        (differential, "differential"),
        (metamorphic, "metamorphic"),
        (configuration, "configuration"),
    ):
        assert result.oracle == oracle
        assert result.passed
        assert result.failure is None
        assert result.checked_seeds == result.selection.selected_seeds
        assert result.checked_cases == 3


def test_coverage_guided_failure_preserves_underlying_oracle_failure():
    def injected_failure(module, inputs):
        raise RuntimeError("injected candidate failure")

    result = coverage.run_coverage_guided_differential_campaign(
        start_seed=0,
        candidate_cases=8,
        budget=3,
        candidate_runner=injected_failure,
    )

    assert not result.passed
    assert result.failure is not None
    assert result.failure.seed == result.selection.selected_seeds[0]
    assert result.failure.signature == "exception:builtins.RuntimeError"
    assert result.checked_seeds == (result.selection.selected_seeds[0],)
    assert result.checked_cases == 1


def test_real_native_coverage_guided_configuration_campaign(tmp_path):
    result = coverage.run_coverage_guided_configuration_campaign(
        start_seed=0,
        candidate_cases=8,
        budget=1,
        cache_dir=tmp_path,
    )

    signature = None if result.failure is None else result.failure.signature
    assert result.passed, f"unexpected coverage-guided configuration failure: {signature}"
    assert result.checked_seeds == result.selection.selected_seeds
    assert result.checked_cases == 1


def test_selection_and_runner_configuration_fail_closed():
    with pytest.raises(TypeError, match="candidate_cases"):
        coverage.select_structural_coverage_seeds(
            start_seed=0,
            candidate_cases=True,
            budget=1,
        )
    with pytest.raises(ValueError, match="positive"):
        coverage.select_structural_coverage_seeds(
            start_seed=0,
            candidate_cases=0,
            budget=1,
        )
    with pytest.raises(TypeError, match="budget"):
        coverage.select_structural_coverage_seeds(
            start_seed=0,
            candidate_cases=1,
            budget=True,
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        coverage.select_structural_coverage_seeds(
            start_seed=0,
            candidate_cases=1,
            budget=2,
        )
    with pytest.raises(ValueError, match="64-bit"):
        coverage.select_structural_coverage_seeds(
            start_seed=(1 << 64) - 1,
            candidate_cases=2,
            budget=1,
        )

    def reference_candidate(module, inputs):
        return execute_reference(module, inputs=inputs)

    with pytest.raises(ValueError, match="default native"):
        coverage.run_coverage_guided_differential_campaign(
            start_seed=0,
            candidate_cases=2,
            budget=1,
            candidate_runner=reference_candidate,
            parallel=True,
        )


def test_observation_rejects_noncanonical_features():
    with pytest.raises(ValueError, match="sorted and unique"):
        coverage.StructuralCoverageObservation(0, ("b", "a"))
    with pytest.raises(ValueError, match="non-empty"):
        coverage.StructuralCoverageObservation(0, ("",))
    with pytest.raises(TypeError, match="text"):
        coverage.StructuralCoverageObservation(0, (np.int32(1),))
