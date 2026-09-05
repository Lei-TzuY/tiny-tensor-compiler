from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .compiler import compile_module
from .differential import (
    _CANDIDATE_FAILURE_EXCEPTIONS,
    _CaseSpec,
    _compare_results,
    _generate_spec,
    _materialize,
    _require_seed,
)
from .ir import Module
from .metamorphic import _shrink_spec
from .repro import capture_repro_case
from .runtime import ExecutionResult


@dataclass(frozen=True)
class CompilerConfiguration:
    """One explicitly named native C compiler used by the cross-compiler oracle."""

    name: str
    compiler: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("compiler configuration name must be non-empty text")
        if not isinstance(self.compiler, str) or not self.compiler:
            raise ValueError("compiler command must be non-empty text")


CROSS_COMPILER_CONFIGURATIONS = (
    CompilerConfiguration("gcc", "gcc"),
    CompilerConfiguration("clang", "clang"),
)

CompilerRunner = Callable[
    [CompilerConfiguration, Module, tuple[np.ndarray, ...]],
    ExecutionResult,
]


@dataclass(frozen=True)
class CrossCompilerMetamorphicFailure:
    """One deterministic compiler divergence and its minimized canonical repro."""

    seed: int
    signature: str
    baseline_compiler: str
    failing_compiler: str
    original_repro: str
    minimized_repro: str
    original_operation_count: int
    minimized_operation_count: int
    shrink_evaluations: int


@dataclass(frozen=True)
class CrossCompilerMetamorphicCampaignResult:
    """Result of one ordered deterministic same-host cross-compiler campaign."""

    start_seed: int
    requested_cases: int
    checked_cases: int
    failure: CrossCompilerMetamorphicFailure | None

    @property
    def passed(self) -> bool:
        return self.failure is None


@dataclass(frozen=True)
class _FailureObservation:
    signature: str
    failing_compiler: str


def run_cross_compiler_metamorphic_campaign(
    *,
    start_seed: int,
    cases: int,
    compiler_runner: CompilerRunner | None = None,
    configurations: tuple[CompilerConfiguration, CompilerConfiguration] = CROSS_COMPILER_CONFIGURATIONS,
    cache_dir: str | os.PathLike[str] | None = None,
) -> CrossCompilerMetamorphicCampaignResult:
    """Require one generated program to agree across two independent native compilers."""
    first_seed = _require_seed(start_seed)
    if not isinstance(cases, int) or isinstance(cases, bool):
        raise TypeError("cases must be an integer")
    if cases <= 0:
        raise ValueError("cases must be positive")
    if first_seed + cases - 1 > (1 << 64) - 1:
        raise ValueError("seed campaign exceeds the 64-bit seed range")
    _validate_configurations(configurations)
    if compiler_runner is not None and cache_dir is not None:
        raise ValueError("cache_dir is only valid for the default native compiler runner")

    runner = compiler_runner
    if runner is None:
        runner = _native_compiler_runner(cache_dir=cache_dir)

    for offset in range(cases):
        seed = first_seed + offset
        spec = _generate_spec(seed)
        observation = _compiler_failure(spec, runner, configurations)
        if observation is None:
            continue

        minimized, evaluations = _shrink_spec(
            spec,
            lambda candidate, expected=observation: _same_failure(
                candidate,
                runner,
                configurations,
                expected,
            ),
        )
        original_module, original_inputs = _materialize(spec)
        minimized_module, minimized_inputs = _materialize(minimized)
        baseline_name = configurations[0].name
        return CrossCompilerMetamorphicCampaignResult(
            start_seed=first_seed,
            requested_cases=cases,
            checked_cases=offset + 1,
            failure=CrossCompilerMetamorphicFailure(
                seed=seed,
                signature=observation.signature,
                baseline_compiler=baseline_name,
                failing_compiler=observation.failing_compiler,
                original_repro=capture_repro_case(original_module, inputs=original_inputs),
                minimized_repro=capture_repro_case(minimized_module, inputs=minimized_inputs),
                original_operation_count=len(spec.operations),
                minimized_operation_count=len(minimized.operations),
                shrink_evaluations=evaluations,
            ),
        )

    return CrossCompilerMetamorphicCampaignResult(
        start_seed=first_seed,
        requested_cases=cases,
        checked_cases=cases,
        failure=None,
    )


def _validate_configurations(
    configurations: tuple[CompilerConfiguration, CompilerConfiguration],
) -> None:
    if not isinstance(configurations, tuple) or len(configurations) != 2:
        raise ValueError("cross-compiler verification requires exactly two compiler configurations")
    baseline, candidate = configurations
    if not isinstance(baseline, CompilerConfiguration) or not isinstance(
        candidate,
        CompilerConfiguration,
    ):
        raise TypeError("compiler configurations must be CompilerConfiguration values")
    if baseline.name == candidate.name:
        raise ValueError("cross-compiler configuration names must be distinct")
    if baseline.compiler == candidate.compiler:
        raise ValueError("cross-compiler commands must be distinct")


def _native_compiler_runner(
    *,
    cache_dir: str | os.PathLike[str] | None,
) -> CompilerRunner:
    def run(
        configuration: CompilerConfiguration,
        module: Module,
        inputs: tuple[np.ndarray, ...],
    ) -> ExecutionResult:
        executable = compile_module(
            module,
            compiler=configuration.compiler,
            cache_dir=cache_dir,
        )
        return executable(inputs=inputs)

    return run


def _compiler_failure(
    spec: _CaseSpec,
    runner: CompilerRunner,
    configurations: tuple[CompilerConfiguration, CompilerConfiguration],
) -> _FailureObservation | None:
    module, inputs = _materialize(spec)
    baseline_configuration, candidate_configuration = configurations

    try:
        baseline = runner(baseline_configuration, module, inputs)
    except _CANDIDATE_FAILURE_EXCEPTIONS as exc:
        return _FailureObservation(
            signature=_exception_signature(
                baseline_configuration.name,
                baseline_configuration.name,
                exc,
            ),
            failing_compiler=baseline_configuration.name,
        )

    try:
        candidate = runner(candidate_configuration, module, inputs)
    except _CANDIDATE_FAILURE_EXCEPTIONS as exc:
        return _FailureObservation(
            signature=_exception_signature(
                baseline_configuration.name,
                candidate_configuration.name,
                exc,
            ),
            failing_compiler=candidate_configuration.name,
        )

    mismatch = _compare_results(candidate, baseline)
    if mismatch is None:
        return None
    return _FailureObservation(
        signature=(
            f"compiler:{baseline_configuration.name}->{candidate_configuration.name}:"
            f"{mismatch}"
        ),
        failing_compiler=candidate_configuration.name,
    )


def _same_failure(
    spec: _CaseSpec,
    runner: CompilerRunner,
    configurations: tuple[CompilerConfiguration, CompilerConfiguration],
    expected: _FailureObservation,
) -> bool:
    return _compiler_failure(spec, runner, configurations) == expected


def _exception_signature(
    baseline_compiler: str,
    failing_compiler: str,
    exc: BaseException,
) -> str:
    type_ = type(exc)
    return (
        f"compiler:{baseline_compiler}->{failing_compiler}:exception:"
        f"{type_.__module__}.{type_.__qualname__}"
    )
