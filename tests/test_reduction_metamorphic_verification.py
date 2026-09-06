from __future__ import annotations

import numpy as np
import pytest

from tiny_tensor_compiler import execute_reference
from tiny_tensor_compiler.reduction_metamorphic import (
    REDUCTION_METAMORPHIC_RELATIONS,
    generate_reduction_metamorphic_case,
    run_reduction_metamorphic_campaign,
)
from tiny_tensor_compiler.repro import load_repro_case, repro_case_sha256


def _single_output_bytes(document: str) -> tuple[tuple[int, ...], np.dtype, bytes]:
    case = load_repro_case(document)
    result = execute_reference(case.module, inputs=case.inputs)
    assert isinstance(result, np.ndarray)
    array = np.array(result, copy=True, order="C")
    return array.shape, array.dtype, array.tobytes(order="C")


def test_same_seed_generates_identical_reduction_relation_and_repro_pair():
    first = generate_reduction_metamorphic_case(5)
    second = generate_reduction_metamorphic_case(5)

    assert first == second
    assert first.relation == "argmax_keepdims_view_equivalence"
    assert first.baseline_repro != first.transformed_repro
    assert repro_case_sha256(first.baseline_repro) == repro_case_sha256(second.baseline_repro)
    assert repro_case_sha256(first.transformed_repro) == repro_case_sha256(
        second.transformed_repro
    )


def test_selector_covers_every_reduction_relation_with_reference_bit_equivalence():
    expected_seeds = {
        0: "sum_axis1_transpose_map",
        1: "prod_all_axes_equivalence",
        3: "sum_all_axes_equivalence",
        5: "argmax_keepdims_view_equivalence",
        6: "sum_keepdims_view_equivalence",
        7: "prod_axis1_transpose_map",
        10: "argmax_axis1_transpose_map",
        11: "sum_reshape_invariance",
        15: "prod_keepdims_view_equivalence",
        16: "prod_reshape_invariance",
    }

    seen = set()
    for seed, expected_relation in expected_seeds.items():
        case = generate_reduction_metamorphic_case(seed)
        assert case.relation == expected_relation
        seen.add(case.relation)
        assert _single_output_bytes(case.baseline_repro) == _single_output_bytes(
            case.transformed_repro
        )

    assert seen == set(REDUCTION_METAMORPHIC_RELATIONS)


def test_relation_pairs_use_reductions_without_mutation_effects():
    for seed in (0, 1, 3, 5, 6, 7, 10, 11, 15, 16):
        case = generate_reduction_metamorphic_case(seed)
        for document in (case.baseline_repro, case.transformed_repro):
            module = load_repro_case(document).module
            opcodes = {op.opcode for op in module.function.ops}
            assert opcodes & {"sum", "prod", "argmax"}
            assert not opcodes & {"copy_into", "binary_inplace", "binary_into"}


def test_reference_candidate_campaign_is_deterministically_clean():
    def reference_runner(module, inputs):
        return execute_reference(module, inputs=inputs)

    first = run_reduction_metamorphic_campaign(
        start_seed=0,
        cases=24,
        candidate_runner=reference_runner,
    )
    second = run_reduction_metamorphic_campaign(
        start_seed=0,
        cases=24,
        candidate_runner=reference_runner,
    )

    assert first == second
    assert first.passed
    assert first.checked_cases == 24
    assert first.failure is None


def test_native_campaign_executes_monoid_and_index_reduction_relations():
    result = run_reduction_metamorphic_campaign(start_seed=5, cases=3)

    signature = None if result.failure is None else result.failure.signature
    assert result.passed, f"unexpected reduction metamorphic failure: {signature}"
    assert result.checked_cases == 3
    assert result.failure is None


def test_argmax_failure_shrinks_without_entering_empty_reduction_domain():
    def wrong_transformed_runner(module, inputs):
        result = execute_reference(module, inputs=inputs)
        if module.function.name.endswith("_transformed"):
            return np.array(0, dtype=np.int64)
        return result

    first = run_reduction_metamorphic_campaign(
        start_seed=5,
        cases=1,
        candidate_runner=wrong_transformed_runner,
    )
    second = run_reduction_metamorphic_campaign(
        start_seed=5,
        cases=1,
        candidate_runner=wrong_transformed_runner,
    )

    assert first == second
    assert not first.passed
    assert first.failure is not None
    failure = first.failure
    assert failure.relation == "argmax_keepdims_view_equivalence"
    assert failure.signature.startswith(
        "reduction-metamorphic:argmax_keepdims_view_equivalence:mismatch:"
    )
    assert failure.minimized_operation_count == 0
    minimized = load_repro_case(failure.minimized_baseline_repro)
    assert minimized.inputs[0].shape == (1, 1)
    assert failure.shrink_evaluations > 0


def test_candidate_exception_signature_is_relation_and_side_stable():
    def failing_transformed_runner(module, inputs):
        if module.function.name.endswith("_transformed"):
            raise RuntimeError("synthetic reduction failure with unstable details")
        return execute_reference(module, inputs=inputs)

    result = run_reduction_metamorphic_campaign(
        start_seed=1,
        cases=1,
        candidate_runner=failing_transformed_runner,
    )

    assert result.failure is not None
    assert result.failure.signature == (
        "reduction-metamorphic:prod_all_axes_equivalence:"
        "transformed-exception:builtins.RuntimeError"
    )
    assert result.failure.minimized_operation_count == 0
    assert result.failure.shrink_evaluations > 0


def test_campaign_configuration_fails_closed():
    with pytest.raises(TypeError, match="seed"):
        generate_reduction_metamorphic_case(True)
    with pytest.raises(ValueError, match="64-bit"):
        generate_reduction_metamorphic_case(-1)
    with pytest.raises(TypeError, match="cases"):
        run_reduction_metamorphic_campaign(start_seed=0, cases=1.5)
    with pytest.raises(ValueError, match="positive"):
        run_reduction_metamorphic_campaign(start_seed=0, cases=0)
    with pytest.raises(ValueError, match="64-bit"):
        run_reduction_metamorphic_campaign(start_seed=(1 << 64) - 1, cases=2)

    def reference_runner(module, inputs):
        return execute_reference(module, inputs=inputs)

    with pytest.raises(ValueError, match="default native"):
        run_reduction_metamorphic_campaign(
            start_seed=0,
            cases=1,
            candidate_runner=reference_runner,
            parallel=True,
        )
