from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any

import numpy as np

from .autodiff import differentiate_module
from .differential import (
    _CANDIDATE_FAILURE_EXCEPTIONS,
    _require_seed,
    _SplitMix64,
)
from .frontend import GraphBuilder
from .ir import DType, Module
from .repro import capture_repro_case
from .runtime import execute_reference

_UINT64_MAX = (1 << 64) - 1

_OPERATIONS = (
    "add_rhs",
    "mul_rhs",
    "square",
    "transpose",
    "reverse0",
    "reverse1",
)
_ARITHMETIC_OPERATIONS = ("add_rhs", "mul_rhs", "square")
_FLOAT_VALUES = (
    np.float64(-0.75),
    np.float64(-0.5),
    np.float64(-0.25),
    np.float64(0.0),
    np.float64(0.25),
    np.float64(0.5),
    np.float64(0.75),
)

GradientRunner = Callable[[Module, tuple[np.ndarray, ...]], Any]


@dataclass(frozen=True)
class GradientConsistencyFailure:
    """One deterministic analytic-vs-numeric gradient divergence and minimized repro."""

    seed: int
    signature: str
    original_repro: str
    minimized_repro: str
    original_operation_count: int
    minimized_operation_count: int
    shrink_evaluations: int


@dataclass(frozen=True)
class GradientConsistencyCampaignResult:
    """Result of one ordered deterministic finite-difference campaign."""

    start_seed: int
    requested_cases: int
    checked_cases: int
    failure: GradientConsistencyFailure | None

    @property
    def passed(self) -> bool:
        return self.failure is None


@dataclass(frozen=True)
class _GradientCaseSpec:
    side: int
    operations: tuple[str, ...]
    inputs: tuple[np.ndarray, np.ndarray]


def generate_gradient_consistency_case(seed: int) -> str:
    """Generate one canonical smooth scalar-loss repro artifact from a 64-bit seed."""
    normalized_seed = _require_seed(seed)
    spec = _generate_spec(normalized_seed)
    module, inputs = _materialize(spec)
    return capture_repro_case(module, inputs=inputs)


def run_gradient_consistency_campaign(
    *,
    start_seed: int,
    cases: int,
    candidate_runner: GradientRunner | None = None,
    step: float = 1.0e-6,
    atol: float = 5.0e-6,
    rtol: float = 5.0e-6,
) -> GradientConsistencyCampaignResult:
    """Compare reverse-mode gradients with central finite differences and shrink failures."""
    first_seed = _require_seed(start_seed)
    if not isinstance(cases, int) or isinstance(cases, bool):
        raise TypeError("cases must be an integer")
    if cases <= 0:
        raise ValueError("cases must be positive")
    if first_seed + cases - 1 > _UINT64_MAX:
        raise ValueError("seed campaign exceeds the 64-bit seed range")

    normalized_step = _require_positive_real("step", step)
    normalized_atol = _require_non_negative_real("atol", atol)
    normalized_rtol = _require_non_negative_real("rtol", rtol)

    if candidate_runner is not None and not callable(candidate_runner):
        raise TypeError("candidate_runner must be callable")
    runner = candidate_runner or _reference_autodiff_runner

    for offset in range(cases):
        seed = first_seed + offset
        spec = _generate_spec(seed)
        signature = _failure_signature(
            spec,
            runner,
            step=normalized_step,
            atol=normalized_atol,
            rtol=normalized_rtol,
        )
        if signature is None:
            continue

        minimized, evaluations = _shrink_failure(
            spec,
            runner,
            signature,
            step=normalized_step,
            atol=normalized_atol,
            rtol=normalized_rtol,
        )
        original_module, original_inputs = _materialize(spec)
        minimized_module, minimized_inputs = _materialize(minimized)
        return GradientConsistencyCampaignResult(
            start_seed=first_seed,
            requested_cases=cases,
            checked_cases=offset + 1,
            failure=GradientConsistencyFailure(
                seed=seed,
                signature=signature,
                original_repro=capture_repro_case(original_module, inputs=original_inputs),
                minimized_repro=capture_repro_case(minimized_module, inputs=minimized_inputs),
                original_operation_count=len(spec.operations),
                minimized_operation_count=len(minimized.operations),
                shrink_evaluations=evaluations,
            ),
        )

    return GradientConsistencyCampaignResult(
        start_seed=first_seed,
        requested_cases=cases,
        checked_cases=cases,
        failure=None,
    )


def _reference_autodiff_runner(
    module: Module,
    inputs: tuple[np.ndarray, ...],
) -> np.ndarray:
    differentiated = differentiate_module(module, wrt=(0,))
    result = execute_reference(differentiated, inputs=inputs)
    if isinstance(result, tuple):
        raise TypeError("single-input gradient unexpectedly returned multiple outputs")
    return result


def _generate_spec(seed: int) -> _GradientCaseSpec:
    rng = _SplitMix64(seed)
    side = 1 + rng.index(3)
    operation_count = 1 + rng.index(5)
    operations = [_ARITHMETIC_OPERATIONS[rng.index(len(_ARITHMETIC_OPERATIONS))]]
    operations.extend(
        _OPERATIONS[rng.index(len(_OPERATIONS))]
        for _ in range(operation_count - 1)
    )
    inputs = (
        _generate_input(rng, side),
        _generate_input(rng, side),
    )
    return _GradientCaseSpec(
        side=side,
        operations=tuple(operations),
        inputs=inputs,
    )


def _generate_input(rng: _SplitMix64, side: int) -> np.ndarray:
    values = [_FLOAT_VALUES[rng.index(len(_FLOAT_VALUES))] for _ in range(side * side)]
    return _freeze_array(np.asarray(values, dtype=np.float64).reshape((side, side)))


def _materialize(
    spec: _GradientCaseSpec,
) -> tuple[Module, tuple[np.ndarray, np.ndarray]]:
    builder = GraphBuilder("gradient_consistency")
    value = builder.input((spec.side, spec.side), DType.FLOAT64)
    rhs = builder.input((spec.side, spec.side), DType.FLOAT64)
    current = value

    for opcode in spec.operations:
        if opcode == "add_rhs":
            current = current + rhs
        elif opcode == "mul_rhs":
            current = current * rhs
        elif opcode == "square":
            current = current * current
        elif opcode == "transpose":
            current = current.transpose((1, 0))
        elif opcode == "reverse0":
            current = current.reverse(0)
        elif opcode == "reverse1":
            current = current.reverse(1)
        else:
            raise RuntimeError(f"unsupported gradient consistency opcode: {opcode}")

    return builder.finish(current.sum()), spec.inputs


def _failure_signature(
    spec: _GradientCaseSpec,
    runner: GradientRunner,
    *,
    step: float,
    atol: float,
    rtol: float,
) -> str | None:
    module, inputs = _materialize(spec)
    expected = _finite_difference_gradient(module, inputs, step=step)
    runner_inputs = tuple(np.array(value, copy=True) for value in inputs)

    try:
        actual = runner(module, runner_inputs)
    except _CANDIDATE_FAILURE_EXCEPTIONS as exc:
        type_ = type(exc)
        return f"exception:{type_.__module__}.{type_.__qualname__}"

    if isinstance(actual, tuple):
        return "mismatch:output-count"
    actual_array = np.asarray(actual)
    if actual_array.shape != expected.shape:
        return "mismatch:shape"
    if actual_array.dtype != expected.dtype:
        return "mismatch:dtype"
    if not np.allclose(
        actual_array,
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=False,
    ):
        return "mismatch:value"
    return None


def _finite_difference_gradient(
    module: Module,
    inputs: tuple[np.ndarray, np.ndarray],
    *,
    step: float,
) -> np.ndarray:
    variable, rhs = inputs
    gradient = np.empty_like(variable, dtype=np.float64)

    for index in np.ndindex(variable.shape):
        plus = np.array(variable, copy=True)
        minus = np.array(variable, copy=True)
        plus[index] += step
        minus[index] -= step
        plus_value = _scalar_loss(module, (plus, rhs))
        minus_value = _scalar_loss(module, (minus, rhs))
        gradient[index] = (plus_value - minus_value) / (2.0 * step)

    return gradient


def _scalar_loss(
    module: Module,
    inputs: tuple[np.ndarray, np.ndarray],
) -> float:
    result = execute_reference(module, inputs=inputs)
    if isinstance(result, tuple):
        raise TypeError("gradient consistency case unexpectedly returned multiple outputs")
    array = np.asarray(result)
    if array.shape != ():
        raise RuntimeError("gradient consistency case unexpectedly returned a non-scalar loss")
    return float(array)


def _shrink_failure(
    original: _GradientCaseSpec,
    runner: GradientRunner,
    signature: str,
    *,
    step: float,
    atol: float,
    rtol: float,
) -> tuple[_GradientCaseSpec, int]:
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
            if _failure_signature(
                candidate,
                runner,
                step=step,
                atol=atol,
                rtol=rtol,
            ) == signature:
                current = candidate
                changed = True
                break

    if current.side > 1:
        for side in range(1, current.side):
            candidate = _with_side(current, side)
            evaluations += 1
            if _failure_signature(
                candidate,
                runner,
                step=step,
                atol=atol,
                rtol=rtol,
            ) == signature:
                current = candidate
                break

    for input_index in range(2):
        zeroed = np.zeros_like(current.inputs[input_index])
        if np.array_equal(zeroed, current.inputs[input_index]):
            continue
        candidate = _with_input(current, input_index, zeroed)
        evaluations += 1
        if _failure_signature(
            candidate,
            runner,
            step=step,
            atol=atol,
            rtol=rtol,
        ) == signature:
            current = candidate

    for input_index in range(2):
        for flat_index in range(current.inputs[input_index].size):
            array = np.array(current.inputs[input_index], copy=True)
            flat = array.reshape(-1)
            if flat[flat_index] == 0:
                continue
            flat[flat_index] = 0
            candidate = _with_input(current, input_index, array)
            evaluations += 1
            if _failure_signature(
                candidate,
                runner,
                step=step,
                atol=atol,
                rtol=rtol,
            ) == signature:
                current = candidate

    return current, evaluations


def _with_side(spec: _GradientCaseSpec, side: int) -> _GradientCaseSpec:
    inputs = tuple(
        _freeze_array(np.asarray(value[:side, :side], dtype=np.float64))
        for value in spec.inputs
    )
    return replace(spec, side=side, inputs=inputs)


def _with_input(
    spec: _GradientCaseSpec,
    input_index: int,
    value: np.ndarray,
) -> _GradientCaseSpec:
    inputs = list(spec.inputs)
    inputs[input_index] = _freeze_array(np.asarray(value, dtype=np.float64))
    return replace(spec, inputs=(inputs[0], inputs[1]))


def _freeze_array(value: np.ndarray) -> np.ndarray:
    frozen = np.array(value, dtype=np.float64, order="C", copy=True)
    frozen.setflags(write=False)
    return frozen


def _require_positive_real(name: str, value: Real) -> float:
    normalized = _require_finite_real(name, value)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _require_non_negative_real(name: str, value: Real) -> float:
    normalized = _require_finite_real(name, value)
    if normalized < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def _require_finite_real(name: str, value: Real) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized
