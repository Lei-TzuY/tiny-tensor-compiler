from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .differential import (
    _CANDIDATE_FAILURE_EXCEPTIONS,
    CandidateRunner,
    _CaseSpec,
    _SplitMix64,
    _compare_results,
    _generate_spec,
    _native_runner,
    _require_seed,
)
from .ir import Module
from .metamorphic import _generated_expression, _shrink_spec
from .repro import capture_repro_case
from .runtime import execute_reference

REDUCTION_METAMORPHIC_RELATIONS = (
    "sum_reshape_invariance",
    "prod_reshape_invariance",
    "sum_all_axes_equivalence",
    "prod_all_axes_equivalence",
    "sum_axis1_transpose_map",
    "prod_axis1_transpose_map",
    "sum_keepdims_view_equivalence",
    "prod_keepdims_view_equivalence",
    "argmax_axis1_transpose_map",
    "argmax_keepdims_view_equivalence",
)

# Keep reduction-relation selection independent of both the base differential
# generator and the historical non-reduction metamorphic relation stream.
_RELATION_DOMAIN = 0xA24BAED4963EE407
_ARGMAX_RELATIONS = frozenset(
    {"argmax_axis1_transpose_map", "argmax_keepdims_view_equivalence"}
)


@dataclass(frozen=True)
class ReductionMetamorphicCase:
    """One deterministic reduction relation represented by canonical repro cases."""

    seed: int
    relation: str
    baseline_repro: str
    transformed_repro: str


@dataclass(frozen=True)
class ReductionMetamorphicFailure:
    """One deterministic reduction-relation failure and its minimized repro pair."""

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
class ReductionMetamorphicCampaignResult:
    """Result of one ordered deterministic reduction-metamorphic campaign."""

    start_seed: int
    requested_cases: int
    checked_cases: int
    failure: ReductionMetamorphicFailure | None

    @property
    def passed(self) -> bool:
        return self.failure is None


def generate_reduction_metamorphic_case(seed: int) -> ReductionMetamorphicCase:
    """Generate one deterministic bit-exact reduction relation pair."""
    normalized_seed = _require_seed(seed)
    spec = _generate_spec(normalized_seed)
    relation = _relation_for_spec(normalized_seed, spec)
    return _capture_pair(normalized_seed, spec, relation)


def run_reduction_metamorphic_campaign(
    *,
    start_seed: int,
    cases: int,
    candidate_runner: CandidateRunner | None = None,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    parallel: bool = False,
) -> ReductionMetamorphicCampaignResult:
    """Run reduction semantic relations and shrink the first exact same-signature failure."""
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
        relation = _relation_for_spec(seed, spec)
        signature = _relation_failure_signature(spec, relation, runner)
        if signature is None:
            continue

        minimized, evaluations = _shrink_relation_failure(spec, relation, runner, signature)
        original = _capture_pair(seed, spec, relation)
        shrunk = _capture_pair(seed, minimized, relation)
        return ReductionMetamorphicCampaignResult(
            start_seed=first_seed,
            requested_cases=cases,
            checked_cases=offset + 1,
            failure=ReductionMetamorphicFailure(
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

    return ReductionMetamorphicCampaignResult(
        start_seed=first_seed,
        requested_cases=cases,
        checked_cases=cases,
        failure=None,
    )


def _relation_for_spec(seed: int, spec: _CaseSpec) -> str:
    relations = REDUCTION_METAMORPHIC_RELATIONS
    if spec.side == 0:
        relations = tuple(relation for relation in relations if relation not in _ARGMAX_RELATIONS)
    rng = _SplitMix64(seed ^ _RELATION_DOMAIN)
    return relations[rng.index(len(relations))]


def _capture_pair(seed: int, spec: _CaseSpec, relation: str) -> ReductionMetamorphicCase:
    baseline, transformed, inputs = _materialize_relation(spec, relation)
    return ReductionMetamorphicCase(
        seed=seed,
        relation=relation,
        baseline_repro=capture_repro_case(baseline, inputs=inputs),
        transformed_repro=capture_repro_case(transformed, inputs=inputs),
    )


def _materialize_relation(
    spec: _CaseSpec,
    relation: str,
) -> tuple[Module, Module, tuple[np.ndarray, np.ndarray]]:
    from .frontend import GraphBuilder

    baseline_builder = GraphBuilder(f"reduction_metamorphic_{relation}_baseline")
    baseline_current = _generated_expression(baseline_builder, spec)
    baseline_result = _relation_baseline(baseline_current, spec, relation)
    baseline = baseline_builder.finish(baseline_result)

    transformed_builder = GraphBuilder(f"reduction_metamorphic_{relation}_transformed")
    transformed_current = _generated_expression(transformed_builder, spec)
    transformed_result = _relation_transformed(transformed_current, spec, relation)
    transformed = transformed_builder.finish(transformed_result)

    return baseline, transformed, spec.inputs


def _relation_baseline(current, spec: _CaseSpec, relation: str):
    del spec
    if relation in {"sum_reshape_invariance", "sum_all_axes_equivalence"}:
        return current.sum()
    if relation in {"prod_reshape_invariance", "prod_all_axes_equivalence"}:
        return current.prod()
    if relation in {"sum_axis1_transpose_map", "sum_keepdims_view_equivalence"}:
        return current.sum(axis=1)
    if relation in {"prod_axis1_transpose_map", "prod_keepdims_view_equivalence"}:
        return current.prod(axis=1)
    if relation in _ARGMAX_RELATIONS:
        return current.argmax(axis=1)
    raise ValueError(f"unknown reduction metamorphic relation: {relation}")


def _relation_transformed(current, spec: _CaseSpec, relation: str):
    if relation == "sum_reshape_invariance":
        return current.reshape((spec.side * spec.side,)).sum()
    if relation == "prod_reshape_invariance":
        return current.reshape((spec.side * spec.side,)).prod()
    if relation == "sum_all_axes_equivalence":
        return current.sum(axis=(0, 1))
    if relation == "prod_all_axes_equivalence":
        return current.prod(axis=(0, 1))
    if relation == "sum_axis1_transpose_map":
        return current.transpose((1, 0)).sum(axis=0)
    if relation == "prod_axis1_transpose_map":
        return current.transpose((1, 0)).prod(axis=0)
    if relation == "sum_keepdims_view_equivalence":
        return current.sum(axis=1, keepdims=True).view((spec.side,))
    if relation == "prod_keepdims_view_equivalence":
        return current.prod(axis=1, keepdims=True).view((spec.side,))
    if relation == "argmax_axis1_transpose_map":
        return current.transpose((1, 0)).argmax(axis=0)
    if relation == "argmax_keepdims_view_equivalence":
        return current.argmax(axis=1, keepdims=True).view((spec.side,))
    raise ValueError(f"unknown reduction metamorphic relation: {relation}")


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
            f"reduction metamorphic relation {relation} is invalid under reference semantics: "
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
    return f"reduction-metamorphic:{relation}:{mismatch}"


def _exception_signature(relation: str, side: str, exc: BaseException) -> str:
    type_ = type(exc)
    return (
        f"reduction-metamorphic:{relation}:{side}-exception:"
        f"{type_.__module__}.{type_.__qualname__}"
    )


def _shrink_relation_failure(
    original: _CaseSpec,
    relation: str,
    runner: CandidateRunner,
    signature: str,
) -> tuple[_CaseSpec, int]:
    def preserves(candidate: _CaseSpec) -> bool:
        if relation in _ARGMAX_RELATIONS and candidate.side == 0:
            return False
        return _relation_failure_signature(candidate, relation, runner) == signature

    return _shrink_spec(original, preserves)
