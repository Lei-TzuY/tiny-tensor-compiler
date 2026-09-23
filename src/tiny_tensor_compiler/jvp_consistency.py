from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .autodiff import jacobian_vector_product_module, vector_jacobian_product_module
from .differential import _CANDIDATE_FAILURE_EXCEPTIONS, _require_seed, _SplitMix64
from .gradient_consistency import (
    _FLOAT_VALUES,
    _freeze_array,
    _generate_spec,
    _GradientCaseSpec,
    _materialize_tensor,
    _require_non_negative_real,
    _require_positive_real,
    _with_input,
    _with_side,
)
from .ir import Module
from .repro import capture_repro_case
from .runtime import execute_reference

_UINT64_MAX = (1 << 64) - 1

JVPRunner = Callable[[Module, tuple[np.ndarray, ...], np.ndarray], Any]
VJPRunner = Callable[[Module, tuple[np.ndarray, ...], np.ndarray], Any]


@dataclass(frozen=True)
class JVPConsistencyCase:
    """One deterministic smooth primal repro plus directional evidence inputs."""

    seed: int
    primal_repro: str
    tangent: np.ndarray
    cotangent: np.ndarray


@dataclass(frozen=True)
class JVPConsistencyFailure:
    """One deterministic JVP correctness divergence and minimized repro case."""

    seed: int
    signature: str
    original_case: JVPConsistencyCase
    minimized_case: JVPConsistencyCase
    original_operation_count: int
    minimized_operation_count: int
    shrink_evaluations: int


@dataclass(frozen=True)
class JVPConsistencyCampaignResult:
    """Result of directional finite-difference plus JVP/VJP duality checks."""

    start_seed: int
    requested_cases: int
    checked_cases: int
    failure: JVPConsistencyFailure | None

    @property
    def passed(self) -> bool:
        return self.failure is None


@dataclass(frozen=True)
class _JVPCaseSpec:
    primal: _GradientCaseSpec
    tangent: np.ndarray
    cotangent: np.ndarray


def generate_jvp_consistency_case(seed: int) -> JVPConsistencyCase:
    """Generate one deterministic smooth tensor-output JVP/VJP evidence case."""
    normalized_seed = _require_seed(seed)
    return _case_artifact(normalized_seed, _generate_case_spec(normalized_seed))


def run_jvp_consistency_campaign(
    *,
    start_seed: int,
    cases: int,
    jvp_runner: JVPRunner | None = None,
    vjp_runner: VJPRunner | None = None,
    step: float = 1.0e-6,
    atol: float = 5.0e-6,
    rtol: float = 5.0e-6,
) -> JVPConsistencyCampaignResult:
    """Check analytic JVPs against directional differences and VJP duality."""
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

    if jvp_runner is not None and not callable(jvp_runner):
        raise TypeError("jvp_runner must be callable")
    if vjp_runner is not None and not callable(vjp_runner):
        raise TypeError("vjp_runner must be callable")

    resolved_jvp = jvp_runner or _reference_jvp_runner
    resolved_vjp = vjp_runner or _reference_vjp_runner

    for offset in range(cases):
        seed = first_seed + offset
        spec = _generate_case_spec(seed)
        signature = _failure_signature(
            spec,
            resolved_jvp,
            resolved_vjp,
            step=normalized_step,
            atol=normalized_atol,
            rtol=normalized_rtol,
        )
        if signature is None:
            continue

        minimized, evaluations = _shrink_failure(
            spec,
            resolved_jvp,
            resolved_vjp,
            signature,
            step=normalized_step,
            atol=normalized_atol,
            rtol=normalized_rtol,
        )
        return JVPConsistencyCampaignResult(
            start_seed=first_seed,
            requested_cases=cases,
            checked_cases=offset + 1,
            failure=JVPConsistencyFailure(
                seed=seed,
                signature=signature,
                original_case=_case_artifact(seed, spec),
                minimized_case=_case_artifact(seed, minimized),
                original_operation_count=len(spec.primal.operations),
                minimized_operation_count=len(minimized.primal.operations),
                shrink_evaluations=evaluations,
            ),
        )

    return JVPConsistencyCampaignResult(
        start_seed=first_seed,
        requested_cases=cases,
        checked_cases=cases,
        failure=None,
    )


def _generate_case_spec(seed: int) -> _JVPCaseSpec:
    primal = _generate_spec(seed)
    rng = _SplitMix64(seed ^ 0xA0761D6478BD642F)
    tangent = _generate_direction(rng, primal.side)
    cotangent = _generate_direction(rng, primal.side)
    return _JVPCaseSpec(
        primal=primal,
        tangent=tangent,
        cotangent=cotangent,
    )


def _generate_direction(rng: _SplitMix64, side: int) -> np.ndarray:
    values = [
        _FLOAT_VALUES[rng.index(len(_FLOAT_VALUES))]
        for _ in range(side * side)
    ]
    array = np.asarray(values, dtype=np.float64).reshape((side, side))
    if not np.any(array):
        array[0, 0] = np.float64(0.5)
    return _freeze_array(array)


def _case_artifact(seed: int, spec: _JVPCaseSpec) -> JVPConsistencyCase:
    module, inputs = _materialize_tensor(
        spec.primal,
        function_name="jvp_consistency",
    )
    return JVPConsistencyCase(
        seed=seed,
        primal_repro=capture_repro_case(module, inputs=inputs),
        tangent=_freeze_array(spec.tangent),
        cotangent=_freeze_array(spec.cotangent),
    )


def _reference_jvp_runner(
    module: Module,
    inputs: tuple[np.ndarray, ...],
    tangent: np.ndarray,
) -> np.ndarray:
    transformed = jacobian_vector_product_module(module, wrt=(0,))
    result = execute_reference(transformed, inputs=inputs + (tangent,))
    if isinstance(result, tuple):
        raise TypeError("single-output JVP unexpectedly returned multiple outputs")
    return result


def _reference_vjp_runner(
    module: Module,
    inputs: tuple[np.ndarray, ...],
    cotangent: np.ndarray,
) -> np.ndarray:
    transformed = vector_jacobian_product_module(module, wrt=(0,))
    result = execute_reference(transformed, inputs=inputs + (cotangent,))
    if isinstance(result, tuple):
        raise TypeError("single-input VJP unexpectedly returned multiple outputs")
    return result


def _failure_signature(
    spec: _JVPCaseSpec,
    jvp_runner: JVPRunner,
    vjp_runner: VJPRunner,
    *,
    step: float,
    atol: float,
    rtol: float,
) -> str | None:
    module, inputs = _materialize_tensor(
        spec.primal,
        function_name="jvp_consistency",
    )
    expected = _directional_finite_difference(
        module,
        inputs,
        spec.tangent,
        step=step,
    )
    runner_inputs = tuple(np.array(value, copy=True) for value in inputs)

    try:
        actual = jvp_runner(
            module,
            runner_inputs,
            np.array(spec.tangent, copy=True),
        )
    except _CANDIDATE_FAILURE_EXCEPTIONS as exc:
        type_ = type(exc)
        return f"jvp:exception:{type_.__module__}.{type_.__qualname__}"

    if isinstance(actual, tuple):
        return "jvp:mismatch:output-count"
    actual_array = np.asarray(actual)
    structural = _array_mismatch(actual_array, expected)
    if structural is not None:
        return f"jvp:mismatch:{structural}"
    if not np.allclose(
        actual_array,
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=False,
    ):
        return "finite-difference:mismatch:value"

    try:
        transposed = vjp_runner(
            module,
            runner_inputs,
            np.array(spec.cotangent, copy=True),
        )
    except _CANDIDATE_FAILURE_EXCEPTIONS as exc:
        type_ = type(exc)
        return f"vjp:exception:{type_.__module__}.{type_.__qualname__}"

    if isinstance(transposed, tuple):
        return "vjp:mismatch:output-count"
    transposed_array = np.asarray(transposed)
    structural = _array_mismatch(transposed_array, spec.tangent)
    if structural is not None:
        return f"vjp:mismatch:{structural}"

    lhs = float(
        np.sum(
            actual_array * np.asarray(spec.cotangent),
            dtype=np.float64,
        )
    )
    rhs = float(
        np.sum(
            np.asarray(spec.tangent) * transposed_array,
            dtype=np.float64,
        )
    )
    if not np.isclose(lhs, rhs, rtol=rtol, atol=atol, equal_nan=False):
        return "duality:mismatch:value"
    return None


def _array_mismatch(actual: np.ndarray, expected: np.ndarray) -> str | None:
    if actual.shape != expected.shape:
        return "shape"
    if actual.dtype != expected.dtype:
        return "dtype"
    return None


def _directional_finite_difference(
    module: Module,
    inputs: tuple[np.ndarray, np.ndarray],
    tangent: np.ndarray,
    *,
    step: float,
) -> np.ndarray:
    variable, rhs = inputs
    plus = np.asarray(variable, dtype=np.float64) + step * np.asarray(tangent)
    minus = np.asarray(variable, dtype=np.float64) - step * np.asarray(tangent)
    plus_value = _tensor_output(module, (plus, rhs))
    minus_value = _tensor_output(module, (minus, rhs))
    return np.asarray((plus_value - minus_value) / (2.0 * step), dtype=np.float64)


def _tensor_output(
    module: Module,
    inputs: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    result = execute_reference(module, inputs=inputs)
    if isinstance(result, tuple):
        raise TypeError("JVP consistency case unexpectedly returned multiple outputs")
    return np.asarray(result)


def _shrink_failure(
    original: _JVPCaseSpec,
    jvp_runner: JVPRunner,
    vjp_runner: VJPRunner,
    signature: str,
    *,
    step: float,
    atol: float,
    rtol: float,
) -> tuple[_JVPCaseSpec, int]:
    current = original
    evaluations = 0

    changed = True
    while changed and current.primal.operations:
        changed = False
        for index in range(len(current.primal.operations)):
            primal = replace(
                current.primal,
                operations=(
                    current.primal.operations[:index]
                    + current.primal.operations[index + 1 :]
                ),
            )
            candidate = replace(current, primal=primal)
            evaluations += 1
            if _failure_signature(
                candidate,
                jvp_runner,
                vjp_runner,
                step=step,
                atol=atol,
                rtol=rtol,
            ) == signature:
                current = candidate
                changed = True
                break

    if current.primal.side > 1:
        for side in range(1, current.primal.side):
            candidate = _with_case_side(current, side)
            evaluations += 1
            if _failure_signature(
                candidate,
                jvp_runner,
                vjp_runner,
                step=step,
                atol=atol,
                rtol=rtol,
            ) == signature:
                current = candidate
                break

    for input_index in range(2):
        zeroed = np.zeros_like(current.primal.inputs[input_index])
        candidate = replace(
            current,
            primal=_with_input(current.primal, input_index, zeroed),
        )
        evaluations += 1
        if _failure_signature(
            candidate,
            jvp_runner,
            vjp_runner,
            step=step,
            atol=atol,
            rtol=rtol,
        ) == signature:
            current = candidate

    for input_index in range(2):
        for flat_index in range(current.primal.inputs[input_index].size):
            array = np.array(current.primal.inputs[input_index], copy=True)
            flat = array.reshape(-1)
            if flat[flat_index] == 0:
                continue
            flat[flat_index] = 0
            candidate = replace(
                current,
                primal=_with_input(current.primal, input_index, array),
            )
            evaluations += 1
            if _failure_signature(
                candidate,
                jvp_runner,
                vjp_runner,
                step=step,
                atol=atol,
                rtol=rtol,
            ) == signature:
                current = candidate

    for field in ("tangent", "cotangent"):
        current, added = _shrink_direction(
            current,
            field,
            jvp_runner,
            vjp_runner,
            signature,
            step=step,
            atol=atol,
            rtol=rtol,
        )
        evaluations += added

    return current, evaluations


def _with_case_side(spec: _JVPCaseSpec, side: int) -> _JVPCaseSpec:
    return replace(
        spec,
        primal=_with_side(spec.primal, side),
        tangent=_freeze_array(spec.tangent[:side, :side]),
        cotangent=_freeze_array(spec.cotangent[:side, :side]),
    )


def _shrink_direction(
    spec: _JVPCaseSpec,
    field: str,
    jvp_runner: JVPRunner,
    vjp_runner: VJPRunner,
    signature: str,
    *,
    step: float,
    atol: float,
    rtol: float,
) -> tuple[_JVPCaseSpec, int]:
    current = spec
    evaluations = 0

    value = np.asarray(getattr(current, field))
    zeroed = np.zeros_like(value)
    candidate = replace(current, **{field: _freeze_array(zeroed)})
    evaluations += 1
    if _failure_signature(
        candidate,
        jvp_runner,
        vjp_runner,
        step=step,
        atol=atol,
        rtol=rtol,
    ) == signature:
        current = candidate

    for flat_index in range(np.asarray(getattr(current, field)).size):
        array = np.array(getattr(current, field), copy=True)
        flat = array.reshape(-1)
        if flat[flat_index] == 0:
            continue
        flat[flat_index] = 0
        candidate = replace(current, **{field: _freeze_array(array)})
        evaluations += 1
        if _failure_signature(
            candidate,
            jvp_runner,
            vjp_runner,
            step=step,
            atol=atol,
            rtol=rtol,
        ) == signature:
            current = candidate

    return current, evaluations
