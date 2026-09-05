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
class NativeConfiguration:
    """One verified native execution configuration used by the metamorphic oracle."""

    name: str
    parallel: bool
    borrow_inputs: bool

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("native configuration name must be non-empty text")
        if not isinstance(self.parallel, bool):
            raise TypeError("native configuration parallel must be bool")
        if not isinstance(self.borrow_inputs, bool):
            raise TypeError("native configuration borrow_inputs must be bool")


NATIVE_CONFIGURATIONS = (
    NativeConfiguration("serial-copied", parallel=False, borrow_inputs=False),
    NativeConfiguration("parallel-copied", parallel=True, borrow_inputs=False),
    NativeConfiguration("serial-borrowed", parallel=False, borrow_inputs=True),
    NativeConfiguration("parallel-borrowed", parallel=True, borrow_inputs=True),
)

ConfigurationRunner = Callable[
    [NativeConfiguration, Module, tuple[np.ndarray, ...]],
    ExecutionResult,
]


@dataclass(frozen=True)
class ConfigurationMetamorphicFailure:
    """One deterministic cross-configuration failure and its minimized repro."""

    seed: int
    signature: str
    baseline_configuration: str
    failing_configuration: str
    original_repro: str
    minimized_repro: str
    original_operation_count: int
    minimized_operation_count: int
    shrink_evaluations: int


@dataclass(frozen=True)
class ConfigurationMetamorphicCampaignResult:
    """Result of one ordered deterministic cross-configuration native campaign."""

    start_seed: int
    requested_cases: int
    checked_cases: int
    failure: ConfigurationMetamorphicFailure | None

    @property
    def passed(self) -> bool:
        return self.failure is None


@dataclass(frozen=True)
class _FailureObservation:
    signature: str
    failing_configuration: str


def run_configuration_metamorphic_campaign(
    *,
    start_seed: int,
    cases: int,
    configuration_runner: ConfigurationRunner | None = None,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> ConfigurationMetamorphicCampaignResult:
    """Require one generated program to agree across verified native configurations."""
    first_seed = _require_seed(start_seed)
    if not isinstance(cases, int) or isinstance(cases, bool):
        raise TypeError("cases must be an integer")
    if cases <= 0:
        raise ValueError("cases must be positive")
    if first_seed + cases - 1 > (1 << 64) - 1:
        raise ValueError("seed campaign exceeds the 64-bit seed range")
    if configuration_runner is not None and (compiler is not None or cache_dir is not None):
        raise ValueError(
            "compiler and cache_dir are only valid for the default native configuration runner"
        )

    runner = configuration_runner
    if runner is None:
        runner = _native_configuration_runner(compiler=compiler, cache_dir=cache_dir)

    for offset in range(cases):
        seed = first_seed + offset
        spec = _generate_spec(seed)
        observation = _configuration_failure(spec, runner)
        if observation is None:
            continue

        minimized, evaluations = _shrink_spec(
            spec,
            lambda candidate, expected=observation: _same_failure(candidate, runner, expected),
        )
        original_module, original_inputs = _materialize(spec)
        minimized_module, minimized_inputs = _materialize(minimized)
        baseline_name = NATIVE_CONFIGURATIONS[0].name
        return ConfigurationMetamorphicCampaignResult(
            start_seed=first_seed,
            requested_cases=cases,
            checked_cases=offset + 1,
            failure=ConfigurationMetamorphicFailure(
                seed=seed,
                signature=observation.signature,
                baseline_configuration=baseline_name,
                failing_configuration=observation.failing_configuration,
                original_repro=capture_repro_case(original_module, inputs=original_inputs),
                minimized_repro=capture_repro_case(minimized_module, inputs=minimized_inputs),
                original_operation_count=len(spec.operations),
                minimized_operation_count=len(minimized.operations),
                shrink_evaluations=evaluations,
            ),
        )

    return ConfigurationMetamorphicCampaignResult(
        start_seed=first_seed,
        requested_cases=cases,
        checked_cases=cases,
        failure=None,
    )


def _native_configuration_runner(
    *,
    compiler: str | None,
    cache_dir: str | os.PathLike[str] | None,
) -> ConfigurationRunner:
    def run(
        configuration: NativeConfiguration,
        module: Module,
        inputs: tuple[np.ndarray, ...],
    ) -> ExecutionResult:
        executable = compile_module(
            module,
            compiler=compiler,
            cache_dir=cache_dir,
            borrow_inputs=configuration.borrow_inputs,
            parallel=configuration.parallel,
        )
        return executable(inputs=inputs)

    return run


def _configuration_failure(
    spec: _CaseSpec,
    runner: ConfigurationRunner,
) -> _FailureObservation | None:
    module, inputs = _materialize(spec)
    baseline_configuration = NATIVE_CONFIGURATIONS[0]

    try:
        baseline = runner(baseline_configuration, module, inputs)
    except _CANDIDATE_FAILURE_EXCEPTIONS as exc:
        return _FailureObservation(
            signature=_exception_signature(
                baseline_configuration.name,
                baseline_configuration.name,
                exc,
            ),
            failing_configuration=baseline_configuration.name,
        )

    for configuration in NATIVE_CONFIGURATIONS[1:]:
        try:
            candidate = runner(configuration, module, inputs)
        except _CANDIDATE_FAILURE_EXCEPTIONS as exc:
            return _FailureObservation(
                signature=_exception_signature(
                    baseline_configuration.name,
                    configuration.name,
                    exc,
                ),
                failing_configuration=configuration.name,
            )

        mismatch = _compare_results(candidate, baseline)
        if mismatch is not None:
            return _FailureObservation(
                signature=(
                    f"configuration:{baseline_configuration.name}->{configuration.name}:"
                    f"{mismatch}"
                ),
                failing_configuration=configuration.name,
            )

    return None


def _same_failure(
    spec: _CaseSpec,
    runner: ConfigurationRunner,
    expected: _FailureObservation,
) -> bool:
    return _configuration_failure(spec, runner) == expected


def _exception_signature(
    baseline_configuration: str,
    failing_configuration: str,
    exc: BaseException,
) -> str:
    type_ = type(exc)
    return (
        f"configuration:{baseline_configuration}->{failing_configuration}:exception:"
        f"{type_.__module__}.{type_.__qualname__}"
    )
