from __future__ import annotations

import os
from dataclasses import dataclass, replace

import numpy as np

from .differential import (
    CandidateRunner,
    _CANDIDATE_FAILURE_EXCEPTIONS,
    _CaseSpec,
    _SplitMix64,
    _compare_results,
    _freeze_array,
    _generate_spec,
    _native_runner,
    _require_seed,
    _with_input,
    _with_side,
)
from .frontend import GraphBuilder, Tensor
from .ir import Module
from .repro import capture_repro_case
from .runtime import ExecutionResult, execute_reference

METAMORPHIC_RELATIONS = (
    "double_reverse_axis0",
    "double_reverse_axis1",
    "double_transpose",
    "identity_view",
    "reshape_roundtrip",
    "relu_idempotence",
    "reverse_transpose_commute",
)

# Separate relation selection from the differential case generator. Changing or
# extending relation selection must never perturb generate_differential_case(seed).
_RELATION_DOMAIN = 0xD1B54A32D192ED03


@dataclass(frozen=True)
class MetamorphicCase:
    """One deterministic semantic relation represented by two canonical repro cases."""

    seed: int
    relation: str
    baseline_repro: str
    transformed_repro: str


@dataclass(frozen=True)
class MetamorphicFailure:
    """One deterministic relation failure and its minimized repro pair."""

    seed: int
    relation: str
    signature: str
    original_baseline_repro: str
    original_transformed_repro: str
    minimized_baseline_repro: str
    minimized_transformed_repro: str
    original_operation_count: int
    minimized_operation_count: int
    shrink_evaluations: int


@dataclass(frozen=True)
class MetamorphicCampaignResult:
    """Result of one ordered deterministic metamorphic campaign."""

    start_seed: int
    requested_cases: int
    checked_cases: int
    failure: MetamorphicFailure | None

    @property
    def passed(self) -> bool:
        return self.failure is None


def generate_metamorphic_case(seed: int) -> MetamorphicCase:
    """Generate one deterministic relation pair without changing the base seed grammar."""
    normalized_seed = _require_seed(seed)
    spec = _generate_spec(normalized_seed)
    relation = _relation_for_seed(normalized_seed)
    return _capture_pair(normalized_seed, spec, relation)


def run_metamorphic_campaign(
    *,
    start_seed: int,
    cases: int,
    candidate_runner: CandidateRunner | None = None,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    parallel: bool = False,
) -> MetamorphicCampaignResult:
    """Run deterministic candidate-vs-candidate semantic relations and shrink the first failure."""
    first_seed = _require_seed(start_seed)
    if not isinstance(cases, int) or isinstance(cases, bool):
        raise TypeError("cases must be an integer")
    if cases <= 0:
        raise ValueError("cases must be positive")
    if first_seed + cases - 1 > (1 << 64) - 1:
        raise ValueError("seed campaign exceeds the 64-bit seed range")
    if candidate_runner is not None and (compiler is not None or cache_dir is not None or parallel):
        raise ValueError(
            "compiler, cache_dir, and parallel are only valid for the default native candidate"
        )

    runner = candidate_runner
    if runner is None:
        runner = _native_runner(compiler=compiler, cache_dir=cache_dir, parallel=parallel)

    for offset in range(cases):
        seed = first_seed + offset
        spec = _generate_spec(seed)
        relation = _relation_for_seed(seed)
        signature = _relation_failure_signature(spec, relation, runner)
        if signature is None:
            continue

        minimized, evaluations = _shrink_relation_failure(spec, relation, runner, signature)
        original = _capture_pair(seed, spec, relation)
        shrunk = _capture_pair(seed, minimized, relation)
        return MetamorphicCampaignResult(
            start_seed=first_seed,
            requested_cases=cases,
            checked_cases=offset + 1,
            failure=MetamorphicFailure(
                seed=seed,
                relation=relation,
                signature=signature,
                original_baseline_repro=original.baseline_repro,
                original_transformed_repro=original.transformed_repro,
                minimized_baseline_repro=shrunk.baseline_repro,
                minimized_transformed_repro=shrunk.transformed_repro,
                original_operation_count=len(spec.operations),
                minimized_operation_count=len(minimized.operations),
                shrink_evaluations=evaluations,
            ),
        )

    return MetamorphicCampaignResult(
        start_seed=first_seed,
        requested_cases=cases,
        checked_cases=cases,
        failure=None,
    )


def _relation_for_seed(seed: int) -> str:
    rng = _SplitMix64(seed ^ _RELATION_DOMAIN)
    return METAMORPHIC_RELATIONS[rng.index(len(METAMORPHIC_RELATIONS))]


def _capture_pair(seed: int, spec: _CaseSpec, relation: str) -> MetamorphicCase:
    baseline, transformed, inputs = _materialize_relation(spec, relation)
    return MetamorphicCase(
        seed=seed,
        relation=relation,
        baseline_repro=capture_repro_case(baseline, inputs=inputs),
        transformed_repro=capture_repro_case(transformed, inputs=inputs),
    )


def _materialize_relation(
    spec: _CaseSpec,
    relation: str,
) -> tuple[Module, Module, tuple[np.ndarray, np.ndarray]]:
    baseline_builder = GraphBuilder(f"metamorphic_{relation}_baseline")
    baseline_current = _generated_expression(baseline_builder, spec)
    baseline_result = _relation_baseline(baseline_current, spec, relation)
    baseline = baseline_builder.finish(baseline_result)

    transformed_builder = GraphBuilder(f"metamorphic_{relation}_transformed")
    transformed_current = _generated_expression(transformed_builder, spec)
    transformed_result = _relation_transformed(transformed_current, spec, relation)
    transformed = transformed_builder.finish(transformed_result)

    return baseline, transformed, spec.inputs


def _generated_expression(builder: GraphBuilder, spec: _CaseSpec) -> Tensor:
    lhs = builder.input((spec.side, spec.side), dtype=spec.dtype)
    rhs = builder.input((spec.side, spec.side), dtype=spec.dtype)
    current = lhs

    for opcode in spec.operations:
        if opcode == "add":
            current = current + rhs
        elif opcode == "mul":
            current = current * rhs
        elif opcode == "relu":
            current = current.relu()
        elif opcode == "reverse0":
            current = current.reverse(0)
        elif opcode == "reverse1":
            current = current.reverse(1)
        elif opcode == "transpose":
            current = current.transpose((1, 0))
        elif opcode == "view":
            current = current.view((spec.side, spec.side))
        elif opcode == "reshape":
            current = current.reshape((spec.side, spec.side))
        else:
            raise RuntimeError(f"unsupported generated metamorphic opcode: {opcode}")
    return current


def _relation_baseline(current: Tensor, spec: _CaseSpec, relation: str) -> Tensor:
    if relation in {"double_reverse_axis0", "double_reverse_axis1", "double_transpose"}:
        return current
    if relation == "identity_view":
        return current.reshape(current.type.shape)
    if relation == "reshape_roundtrip":
        return current.reshape((spec.side, spec.side))
    if relation == "relu_idempotence":
        return current.relu()
    if relation == "reverse_transpose_commute":
        return current.reverse(0).transpose((1, 0))
    raise ValueError(f"unknown metamorphic relation: {relation}")


def _relation_transformed(current: Tensor, spec: _CaseSpec, relation: str) -> Tensor:
    if relation == "double_reverse_axis0":
        return current.reverse(0).reverse(0)
    if relation == "double_reverse_axis1":
        return current.reverse(1).reverse(1)
    if relation == "double_transpose":
        return current.transpose((1, 0)).transpose((1, 0))
    if relation == "identity_view":
        copied = current.reshape(current.type.shape)
        return copied.view(copied.type.shape)
    if relation == "reshape_roundtrip":
        return current.reshape((spec.side * spec.side,)).reshape((spec.side, spec.side))
    if relation == "relu_idempotence":
        return current.relu().relu()
    if relation == "reverse_transpose_commute":
        return current.transpose((1, 0)).reverse(1)
    raise ValueError(f"unknown metamorphic relation: {relation}")


def _relation_failure_signature(
    spec: _CaseSpec,
    relation: str,
    runner: CandidateRunner,
) -> str | None:
    baseline, transformed, inputs = _materialize_relation(spec, relation)

    baseline_reference = execute_reference(baseline, inputs=inputs)
    transformed_reference = execute_reference(transformed, inputs=inputs)
    invalid_relation = _compare_results(baseline_reference, transformed_reference)
    if invalid_relation is not None:
        raise RuntimeError(
            f"metamorphic relation {relation} is invalid under reference semantics: "
            f"{invalid_relation}"
        )

    try:
        baseline_result = runner(baseline, inputs)
    except _CANDIDATE_FAILURE_EXCEPTIONS as exc:
        return _exception_signature(relation, "baseline", exc)

    try:
        transformed_result = runner(transformed, inputs)
    except _CANDIDATE_FAILURE_EXCEPTIONS as exc:
        return _exception_signature(relation, "transformed", exc)

    mismatch = _compare_results(baseline_result, transformed_result)
    if mismatch is None:
        return None
    return f"metamorphic:{relation}:{mismatch}"


def _exception_signature(relation: str, side: str, exc: BaseException) -> str:
    type_ = type(exc)
    return (
        f"metamorphic:{relation}:{side}-exception:"
        f"{type_.__module__}.{type_.__qualname__}"
    )


def _shrink_relation_failure(
    original: _CaseSpec,
    relation: str,
    runner: CandidateRunner,
    signature: str,
) -> tuple[_CaseSpec, int]:
    def preserves(candidate: _CaseSpec) -> bool:
        return _relation_failure_signature(candidate, relation, runner) == signature

    return _shrink_spec(original, preserves)


def _shrink_spec(
    original: _CaseSpec,
    preserves,
) -> tuple[_CaseSpec, int]:
    """Apply the differential shrink order while preserving a caller-defined property."""
    current = original
    evaluations = 0

    changed = True
    while changed and current.operations:
        changed = False
        for index in range(len(current.operations)):
            candidate = replace(
                current,
                operations=current.operations[:index] + current.operations[index + 1 :],
            )
            evaluations += 1
            if preserves(candidate):
                current = candidate
                changed = True
                break

    if current.side > 0:
        for side in range(current.side):
            candidate = _with_side(current, side)
            evaluations += 1
            if preserves(candidate):
                current = candidate
                break

    for input_index in range(2):
        zeroed = np.zeros_like(current.inputs[input_index])
        if np.array_equal(zeroed, current.inputs[input_index]):
            continue
        candidate = _with_input(current, input_index, zeroed)
        evaluations += 1
        if preserves(candidate):
            current = candidate

    for input_index in range(2):
        for flat_index in range(current.inputs[input_index].size):
            array = np.array(current.inputs[input_index], copy=True, order="C")
            flat = array.reshape(-1)
            if flat[flat_index] == 0:
                continue
            flat[flat_index] = 0
            candidate = _with_input(current, input_index, _freeze_array(array))
            evaluations += 1
            if preserves(candidate):
                current = candidate

    return current, evaluations
