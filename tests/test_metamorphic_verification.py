from __future__ import annotations

import numpy as np
import pytest

from tiny_tensor_compiler import execute_reference
from tiny_tensor_compiler.metamorphic import (
    METAMORPHIC_RELATIONS,
    generate_metamorphic_case,
    run_metamorphic_campaign,
)
from tiny_tensor_compiler.repro import load_repro_case, repro_case_sha256


def _single_output_bytes(document: str) -> tuple[tuple[int, ...], np.dtype, bytes]:
    case = load_repro_case(document)
    result = execute_reference(case.module, inputs=case.inputs)
    assert isinstance(result, np.ndarray)
    array = np.array(result, copy=True, order="C")
    return array.shape, array.dtype, array.tobytes(order="C")


def test_same_seed_generates_identical_relation_and_canonical_repro_pair():
    first = generate_metamorphic_case(11)
    second = generate_metamorphic_case(11)

    assert first == second
    assert first.relation == "reverse_transpose_commute"
    assert first.baseline_repro != first.transformed_repro
    assert repro_case_sha256(first.baseline_repro) == repro_case_sha256(second.baseline_repro)
    assert repro_case_sha256(first.transformed_repro) == repro_case_sha256(
        second.transformed_repro
    )


def test_relation_selector_covers_every_bounded_relation_with_reference_equivalence():
    expected_seeds = {
        0: "double_reverse_axis0",
        1: "identity_view",
        3: "double_reverse_axis1",
        7: "reshape_roundtrip",
        10: "double_transpose",
        11: "reverse_transpose_commute",
        23: "relu_idempotence",
    }

    seen = set()
    for seed, expected_relation in expected_seeds.items():
        case = generate_metamorphic_case(seed)
        assert case.relation == expected_relation
        seen.add(case.relation)
        assert _single_output_bytes(case.baseline_repro) == _single_output_bytes(
            case.transformed_repro
        )

    assert seen == set(METAMORPHIC_RELATIONS)


def test_generated_relations_stay_outside_reduction_and_mutation_surfaces():
    for seed in (0, 1, 3, 7, 10, 11, 23):
        case = generate_metamorphic_case(seed)
        for document in (case.baseline_repro, case.transformed_repro):
            module = load_repro_case(document).module
            opcodes = {op.opcode for op in module.function.ops}
            assert "sum" not in opcodes
            assert "copy_into" not in opcodes


def test_reference_candidate_campaign_is_deterministically_clean():
    def reference_runner(module, inputs):
        return execute_reference(module, inputs=inputs)

    first = run_metamorphic_campaign(
        start_seed=0,
        cases=24,
        candidate_runner=reference_runner,
    )
    second = run_metamorphic_campaign(
        start_seed=0,
        cases=24,
        candidate_runner=reference_runner,
    )

    assert first == second
    assert first.passed
    assert first.checked_cases == 24
    assert first.failure is None


def test_native_campaign_executes_multiple_metamorphic_relations():
    result = run_metamorphic_campaign(start_seed=0, cases=4)

    signature = None if result.failure is None else result.failure.signature
    assert result.passed, f"unexpected metamorphic failure: {signature}"
    assert result.checked_cases == 4
    assert result.failure is None


def test_relation_mismatch_shrinks_deterministically_to_same_repro_pair():
    def wrong_transformed_runner(module, inputs):
        result = execute_reference(module, inputs=inputs)
        if module.function.name.endswith("_transformed"):
            return np.array(0, dtype=np.int32)
        return result

    first = run_metamorphic_campaign(
        start_seed=11,
        cases=1,
        candidate_runner=wrong_transformed_runner,
    )
    second = run_metamorphic_campaign(
        start_seed=11,
        cases=1,
        candidate_runner=wrong_transformed_runner,
    )

    assert first == second
    assert not first.passed
    assert first.failure is not None
    failure = first.failure
    assert failure.seed == 11
    assert failure.relation == "reverse_transpose_commute"
    assert failure.signature == "metamorphic:reverse_transpose_commute:mismatch:shape:0"
    assert failure.original_operation_count > failure.minimized_operation_count
    assert failure.minimized_operation_count == 0
    assert failure.shrink_evaluations > 0
    assert repro_case_sha256(failure.minimized_baseline_repro) == repro_case_sha256(
        second.failure.minimized_baseline_repro
    )
    assert repro_case_sha256(failure.minimized_transformed_repro) == repro_case_sha256(
        second.failure.minimized_transformed_repro
    )


def test_transformed_candidate_exception_is_relation_aware_stable_signature():
    def failing_transformed_runner(module, inputs):
        if module.function.name.endswith("_transformed"):
            raise RuntimeError("synthetic transformed failure")
        return execute_reference(module, inputs=inputs)

    result = run_metamorphic_campaign(
        start_seed=0,
        cases=1,
        candidate_runner=failing_transformed_runner,
    )

    assert result.failure is not None
    assert result.failure.signature == (
        "metamorphic:double_reverse_axis0:transformed-exception:builtins.RuntimeError"
    )
    assert result.failure.minimized_operation_count == 0
    assert result.failure.shrink_evaluations > 0


def test_campaign_configuration_fails_closed():
    with pytest.raises(TypeError, match="seed"):
        generate_metamorphic_case(True)
    with pytest.raises(ValueError, match="64-bit"):
        generate_metamorphic_case(-1)
    with pytest.raises(TypeError, match="cases"):
        run_metamorphic_campaign(start_seed=0, cases=1.5)
    with pytest.raises(ValueError, match="positive"):
        run_metamorphic_campaign(start_seed=0, cases=0)
    with pytest.raises(ValueError, match="64-bit"):
        run_metamorphic_campaign(start_seed=(1 << 64) - 1, cases=2)

    def reference_runner(module, inputs):
        return execute_reference(module, inputs=inputs)

    with pytest.raises(ValueError, match="default native"):
        run_metamorphic_campaign(
            start_seed=0,
            cases=1,
            candidate_runner=reference_runner,
            parallel=True,
        )
