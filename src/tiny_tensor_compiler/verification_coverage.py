from __future__ import annotations

import os
from dataclasses import dataclass
from itertools import pairwise

from .configuration_metamorphic import (
    ConfigurationMetamorphicFailure,
    ConfigurationRunner,
    run_configuration_metamorphic_campaign,
)
from .differential import (
    CandidateRunner,
    DifferentialFailure,
    _generate_spec,
    _materialize,
    _require_seed,
    run_differential_campaign,
)
from .fusion_planner import fuse_elementwise
from .loop_ir import fused_expression_for_kernel, lower_to_loops
from .lowering import lower_to_cpu
from .metamorphic import (
    MetamorphicFailure,
    _relation_for_seed,
    run_metamorphic_campaign,
)

_UINT64_MAX = (1 << 64) - 1

CoverageFailure = DifferentialFailure | MetamorphicFailure | ConfigurationMetamorphicFailure


@dataclass(frozen=True)
class StructuralCoverageObservation:
    """Deterministic measured structural feature set for one generated verification seed."""

    seed: int
    features: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_seed(self.seed)
        if not isinstance(self.features, tuple):
            raise TypeError("structural coverage features must be a tuple")
        if any(not isinstance(feature, str) for feature in self.features):
            raise TypeError("structural coverage features must contain text values")
        if any(not feature for feature in self.features):
            raise ValueError("structural coverage feature names must be non-empty")
        if tuple(sorted(set(self.features))) != self.features:
            raise ValueError("structural coverage features must be sorted and unique")


@dataclass(frozen=True)
class StructuralCoverageSelection:
    """One deterministic fixed-budget selection from a measured candidate seed range."""

    start_seed: int
    candidate_cases: int
    budget: int
    selected: tuple[StructuralCoverageObservation, ...]
    candidate_features: tuple[str, ...]
    covered_features: tuple[str, ...]

    @property
    def selected_seeds(self) -> tuple[int, ...]:
        return tuple(observation.seed for observation in self.selected)

    @property
    def uncovered_features(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.candidate_features) - set(self.covered_features)))


@dataclass(frozen=True)
class CoverageGuidedCampaignResult:
    """Result of running one existing oracle over a measured fixed-budget seed selection."""

    oracle: str
    selection: StructuralCoverageSelection
    checked_seeds: tuple[int, ...]
    failure: CoverageFailure | None

    def __post_init__(self) -> None:
        if self.oracle not in {"differential", "metamorphic", "configuration"}:
            raise ValueError(f"unsupported coverage-guided oracle: {self.oracle}")
        expected_prefix = self.selection.selected_seeds[: len(self.checked_seeds)]
        if self.checked_seeds != expected_prefix:
            raise ValueError("checked seeds must be a prefix of the selected seed order")
        if self.failure is None and self.checked_seeds != self.selection.selected_seeds:
            raise ValueError("passing coverage-guided campaigns must check every selected seed")

    @property
    def passed(self) -> bool:
        return self.failure is None

    @property
    def checked_cases(self) -> int:
        return len(self.checked_seeds)


def measure_structural_coverage(seed: int) -> StructuralCoverageObservation:
    """Measure stable verification features from the generated grammar and lowered compiler IR."""
    normalized_seed = _require_seed(seed)
    spec = _generate_spec(normalized_seed)
    module, _ = _materialize(spec)
    loops = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))

    features: set[str] = {
        f"dtype:{spec.dtype.value}",
        f"metamorphic-relation:{_relation_for_seed(normalized_seed)}",
    }
    if spec.side == 0:
        features.add("extent:zero")
    elif spec.side == 1:
        features.add("extent:singleton")
    else:
        features.add("extent:multi")

    for opcode in spec.operations:
        features.add(f"generator-op:{opcode}")
    for lhs, rhs in pairwise(spec.operations):
        features.add(f"generator-transition:{lhs}->{rhs}")

    for op in module.function.ops:
        if op.opcode not in {"input", "return"}:
            features.add(f"tensor-op:{op.opcode}")

    layouts = loops.value_layouts
    types = loops.value_types
    for view in loops.views:
        features.add("loop:view")
        layout = layouts[view.output]
        type_ = types[view.output]
        if any(stride < 0 for stride in layout.strides):
            features.add("layout:negative-stride")
        if layout.offset:
            features.add("layout:nonzero-offset")
        if layout.is_contiguous(type_.shape):
            features.add("layout:contiguous-view")
        else:
            features.add("layout:noncontiguous-view")

    if loops.copies:
        features.add("loop:copy-into")

    for kernel in loops.kernels:
        features.add("loop:kernel")
        if not kernel.iteration_shape:
            features.add("kernel-extent:scalar")
        elif any(dim == 0 for dim in kernel.iteration_shape):
            features.add("kernel-extent:zero")
        else:
            features.add("kernel-extent:nonzero")

        expression = fused_expression_for_kernel(kernel)
        if expression is None:
            features.add(f"loop-kernel:{kernel.opcode}")
        else:
            features.add(f"fusion:{expression.family}")
            if expression.terminal_relu:
                features.add("fusion:terminal-relu")
            for step in expression.steps:
                features.add(f"fusion-step:{step.opcode}")

        if kernel.inputs:
            input_layouts = [layouts[buffer] for buffer in kernel.inputs]
            input_types = [types[buffer] for buffer in kernel.inputs]
            if all(
                layout.is_contiguous(type_.shape)
                for layout, type_ in zip(input_layouts, input_types, strict=True)
            ):
                features.add("kernel-input:all-contiguous")
            else:
                features.add("kernel-input:noncontiguous")
            if any(
                stride < 0
                for layout in input_layouts
                for stride in layout.strides
            ):
                features.add("kernel-input:negative-stride")

    return StructuralCoverageObservation(
        seed=normalized_seed,
        features=tuple(sorted(features)),
    )


def select_structural_coverage_seeds(
    *,
    start_seed: int,
    candidate_cases: int,
    budget: int,
) -> StructuralCoverageSelection:
    """Select a fixed seed budget by deterministic greedy measured-feature gain."""
    first_seed = _require_selection_range(start_seed, candidate_cases, budget)
    observations = tuple(
        measure_structural_coverage(first_seed + offset)
        for offset in range(candidate_cases)
    )
    candidate_features = _union_features(observations)

    remaining = list(observations)
    selected: list[StructuralCoverageObservation] = []
    covered: set[str] = set()
    for _ in range(budget):
        best = min(
            remaining,
            key=lambda observation: (
                -len(set(observation.features) - covered),
                observation.seed,
            ),
        )
        remaining.remove(best)
        selected.append(best)
        covered.update(best.features)

    return StructuralCoverageSelection(
        start_seed=first_seed,
        candidate_cases=candidate_cases,
        budget=budget,
        selected=tuple(selected),
        candidate_features=candidate_features,
        covered_features=tuple(sorted(covered)),
    )


def run_coverage_guided_differential_campaign(
    *,
    start_seed: int,
    candidate_cases: int,
    budget: int,
    candidate_runner: CandidateRunner | None = None,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    parallel: bool = False,
) -> CoverageGuidedCampaignResult:
    """Run the existing reference-vs-candidate oracle on selected measured seeds."""
    if candidate_runner is not None and (compiler is not None or cache_dir is not None or parallel):
        raise ValueError(
            "compiler, cache_dir, and parallel are only valid for the default native candidate"
        )
    selection = select_structural_coverage_seeds(
        start_seed=start_seed,
        candidate_cases=candidate_cases,
        budget=budget,
    )
    checked: list[int] = []
    for seed in selection.selected_seeds:
        result = run_differential_campaign(
            start_seed=seed,
            cases=1,
            candidate_runner=candidate_runner,
            compiler=compiler,
            cache_dir=cache_dir,
            parallel=parallel,
        )
        checked.append(seed)
        if result.failure is not None:
            return CoverageGuidedCampaignResult(
                oracle="differential",
                selection=selection,
                checked_seeds=tuple(checked),
                failure=result.failure,
            )
    return CoverageGuidedCampaignResult(
        oracle="differential",
        selection=selection,
        checked_seeds=tuple(checked),
        failure=None,
    )


def run_coverage_guided_metamorphic_campaign(
    *,
    start_seed: int,
    candidate_cases: int,
    budget: int,
    candidate_runner: CandidateRunner | None = None,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    parallel: bool = False,
) -> CoverageGuidedCampaignResult:
    """Run the existing IR metamorphic oracle on selected measured seeds."""
    if candidate_runner is not None and (compiler is not None or cache_dir is not None or parallel):
        raise ValueError(
            "compiler, cache_dir, and parallel are only valid for the default native candidate"
        )
    selection = select_structural_coverage_seeds(
        start_seed=start_seed,
        candidate_cases=candidate_cases,
        budget=budget,
    )
    checked: list[int] = []
    for seed in selection.selected_seeds:
        result = run_metamorphic_campaign(
            start_seed=seed,
            cases=1,
            candidate_runner=candidate_runner,
            compiler=compiler,
            cache_dir=cache_dir,
            parallel=parallel,
        )
        checked.append(seed)
        if result.failure is not None:
            return CoverageGuidedCampaignResult(
                oracle="metamorphic",
                selection=selection,
                checked_seeds=tuple(checked),
                failure=result.failure,
            )
    return CoverageGuidedCampaignResult(
        oracle="metamorphic",
        selection=selection,
        checked_seeds=tuple(checked),
        failure=None,
    )


def run_coverage_guided_configuration_campaign(
    *,
    start_seed: int,
    candidate_cases: int,
    budget: int,
    configuration_runner: ConfigurationRunner | None = None,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> CoverageGuidedCampaignResult:
    """Run cross-configuration native metamorphism on selected measured seeds."""
    if configuration_runner is not None and (compiler is not None or cache_dir is not None):
        raise ValueError(
            "compiler and cache_dir are only valid for the default native configuration runner"
        )
    selection = select_structural_coverage_seeds(
        start_seed=start_seed,
        candidate_cases=candidate_cases,
        budget=budget,
    )
    checked: list[int] = []
    for seed in selection.selected_seeds:
        result = run_configuration_metamorphic_campaign(
            start_seed=seed,
            cases=1,
            configuration_runner=configuration_runner,
            compiler=compiler,
            cache_dir=cache_dir,
        )
        checked.append(seed)
        if result.failure is not None:
            return CoverageGuidedCampaignResult(
                oracle="configuration",
                selection=selection,
                checked_seeds=tuple(checked),
                failure=result.failure,
            )
    return CoverageGuidedCampaignResult(
        oracle="configuration",
        selection=selection,
        checked_seeds=tuple(checked),
        failure=None,
    )


def _require_selection_range(start_seed: int, candidate_cases: int, budget: int) -> int:
    first_seed = _require_seed(start_seed)
    if not isinstance(candidate_cases, int) or isinstance(candidate_cases, bool):
        raise TypeError("candidate_cases must be an integer")
    if candidate_cases <= 0:
        raise ValueError("candidate_cases must be positive")
    if first_seed + candidate_cases - 1 > _UINT64_MAX:
        raise ValueError("candidate seed range exceeds the 64-bit seed range")
    if not isinstance(budget, int) or isinstance(budget, bool):
        raise TypeError("budget must be an integer")
    if budget <= 0:
        raise ValueError("budget must be positive")
    if budget > candidate_cases:
        raise ValueError("budget cannot exceed candidate_cases")
    return first_seed


def _union_features(
    observations: tuple[StructuralCoverageObservation, ...],
) -> tuple[str, ...]:
    return tuple(sorted({feature for observation in observations for feature in observation.features}))
