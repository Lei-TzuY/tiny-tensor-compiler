from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .compiler import compile_module
from .frontend import GraphBuilder
from .ir import DType, Module
from .repro import capture_repro_case
from .runtime import ExecutionResult, execute_reference

_UINT64_MAX = (1 << 64) - 1
_MASK64 = _UINT64_MAX

_OPERATIONS = (
    "add",
    "mul",
    "relu",
    "reverse0",
    "reverse1",
    "transpose",
    "view",
    "reshape",
)

_CANONICAL_DTYPES = {
    DType.INT32: np.dtype("<i4"),
    DType.INT64: np.dtype("<i8"),
    DType.FLOAT32: np.dtype("<f4"),
}

_INT32_VALUES = (
    np.int32(-(1 << 31)),
    np.int32(-3),
    np.int32(-1),
    np.int32(0),
    np.int32(1),
    np.int32(3),
    np.int32((1 << 31) - 1),
)

_FLOAT32_VALUES = (
    np.float32(-4.0),
    np.float32(-2.0),
    np.float32(-1.0),
    np.float32(-0.0),
    np.float32(0.0),
    np.float32(0.5),
    np.float32(1.0),
    np.float32(2.0),
    np.float32(4.0),
)

CandidateRunner = Callable[[Module, tuple[np.ndarray, ...]], ExecutionResult]

_CANDIDATE_FAILURE_EXCEPTIONS = (
    AssertionError,
    ArithmeticError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True)
class DifferentialFailure:
    """One deterministic candidate failure and its minimized canonical repro artifact."""

    seed: int
    signature: str
    original_repro: str
    minimized_repro: str
    original_operation_count: int
    minimized_operation_count: int
    shrink_evaluations: int


@dataclass(frozen=True)
class DifferentialCampaignResult:
    """Result of one ordered deterministic seed campaign."""

    start_seed: int
    requested_cases: int
    checked_cases: int
    failure: DifferentialFailure | None

    @property
    def passed(self) -> bool:
        return self.failure is None


@dataclass(frozen=True)
class _CaseSpec:
    dtype: DType
    side: int
    operations: tuple[str, ...]
    inputs: tuple[np.ndarray, np.ndarray]


@dataclass
class _SplitMix64:
    state: int

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
        return (value ^ (value >> 31)) & _MASK64

    def index(self, size: int) -> int:
        if size <= 0:
            raise ValueError("random choice size must be positive")
        return self.next_u64() % size


def generate_differential_case(seed: int) -> str:
    """Generate one canonical bounded pure-graph repro artifact from a 64-bit seed."""
    normalized_seed = _require_seed(seed)
    spec = _generate_spec(normalized_seed)
    module, inputs = _materialize(spec)
    return capture_repro_case(module, inputs=inputs)


def run_differential_campaign(
    *,
    start_seed: int,
    cases: int,
    candidate_runner: CandidateRunner | None = None,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    parallel: bool = False,
) -> DifferentialCampaignResult:
    """Compare deterministic generated cases and shrink the first reproducible failure."""
    first_seed = _require_seed(start_seed)
    if not isinstance(cases, int) or isinstance(cases, bool):
        raise TypeError("cases must be an integer")
    if cases <= 0:
        raise ValueError("cases must be positive")
    if first_seed + cases - 1 > _UINT64_MAX:
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
        signature = _failure_signature(spec, runner)
        if signature is None:
            continue

        minimized, evaluations = _shrink_failure(spec, runner, signature)
        original_module, original_inputs = _materialize(spec)
        minimized_module, minimized_inputs = _materialize(minimized)
        return DifferentialCampaignResult(
            start_seed=first_seed,
            requested_cases=cases,
            checked_cases=offset + 1,
            failure=DifferentialFailure(
                seed=seed,
                signature=signature,
                original_repro=capture_repro_case(original_module, inputs=original_inputs),
                minimized_repro=capture_repro_case(minimized_module, inputs=minimized_inputs),
                original_operation_count=len(spec.operations),
                minimized_operation_count=len(minimized.operations),
                shrink_evaluations=evaluations,
            ),
        )

    return DifferentialCampaignResult(
        start_seed=first_seed,
        requested_cases=cases,
        checked_cases=cases,
        failure=None,
    )


def _native_runner(
    *,
    compiler: str | None,
    cache_dir: str | os.PathLike[str] | None,
    parallel: bool,
) -> CandidateRunner:
    def run(module: Module, inputs: tuple[np.ndarray, ...]) -> ExecutionResult:
        executable = compile_module(
            module,
            compiler=compiler,
            cache_dir=cache_dir,
            parallel=parallel,
        )
        return executable(inputs=inputs)

    return run


def _require_seed(seed: int) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if seed < 0 or seed > _UINT64_MAX:
        raise ValueError("seed must fit in an unsigned 64-bit integer")
    return seed


def _generate_spec(seed: int) -> _CaseSpec:
    rng = _SplitMix64(seed)
    dtype = (DType.INT32, DType.FLOAT32)[rng.index(2)]
    side = rng.index(5)
    operation_count = 1 + rng.index(6)
    operations = tuple(_OPERATIONS[rng.index(len(_OPERATIONS))] for _ in range(operation_count))
    inputs = (
        _generate_input(rng, dtype, side),
        _generate_input(rng, dtype, side),
    )
    return _CaseSpec(dtype=dtype, side=side, operations=operations, inputs=inputs)


def _generate_input(rng: _SplitMix64, dtype: DType, side: int) -> np.ndarray:
    pool = _INT32_VALUES if dtype is DType.INT32 else _FLOAT32_VALUES
    values = [pool[rng.index(len(pool))] for _ in range(side * side)]
    array = np.asarray(values, dtype=dtype.to_numpy()).reshape((side, side))
    return _freeze_array(array)


def _materialize(spec: _CaseSpec) -> tuple[Module, tuple[np.ndarray, np.ndarray]]:
    builder = GraphBuilder("differential")
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
            raise RuntimeError(f"unsupported generated differential opcode: {opcode}")

    return builder.finish(current), spec.inputs


def _failure_signature(spec: _CaseSpec, runner: CandidateRunner) -> str | None:
    module, inputs = _materialize(spec)
    expected = execute_reference(module, inputs=inputs)
    try:
        actual = runner(module, inputs)
    except _CANDIDATE_FAILURE_EXCEPTIONS as exc:
        type_ = type(exc)
        return f"exception:{type_.__module__}.{type_.__qualname__}"
    return _compare_results(actual, expected)


def _compare_results(actual: ExecutionResult, expected: ExecutionResult) -> str | None:
    actual_outputs = actual if isinstance(actual, tuple) else (actual,)
    expected_outputs = expected if isinstance(expected, tuple) else (expected,)
    if len(actual_outputs) != len(expected_outputs):
        return "mismatch:output-count"

    for index, (actual_output, expected_output) in enumerate(
        zip(actual_outputs, expected_outputs, strict=True)
    ):
        actual_array = np.asarray(actual_output)
        expected_array = np.asarray(expected_output)
        if actual_array.shape != expected_array.shape:
            return f"mismatch:shape:{index}"
        if actual_array.dtype != expected_array.dtype:
            return f"mismatch:dtype:{index}"
        if _canonical_bytes(actual_array) != _canonical_bytes(expected_array):
            return f"mismatch:bytes:{index}"
    return None


def _canonical_bytes(value: np.ndarray) -> bytes:
    array = np.asarray(value)
    dtype = DType.from_numpy(array.dtype)
    try:
        canonical_dtype = _CANONICAL_DTYPES[dtype]
    except KeyError as exc:
        raise TypeError(f"differential output dtype is unsupported: {dtype.value}") from exc
    canonical = np.array(array, dtype=canonical_dtype, order="C", copy=True).reshape(array.shape)
    return canonical.tobytes(order="C")


def _shrink_failure(
    original: _CaseSpec,
    runner: CandidateRunner,
    signature: str,
) -> tuple[_CaseSpec, int]:
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
            if _failure_signature(candidate, runner) == signature:
                current = candidate
                changed = True
                break

    if current.side > 0:
        for side in range(current.side):
            candidate = _with_side(current, side)
            evaluations += 1
            if _failure_signature(candidate, runner) == signature:
                current = candidate
                break

    for input_index in range(2):
        zeroed = np.zeros_like(current.inputs[input_index])
        if np.array_equal(zeroed, current.inputs[input_index]):
            continue
        candidate = _with_input(current, input_index, zeroed)
        evaluations += 1
        if _failure_signature(candidate, runner) == signature:
            current = candidate

    for input_index in range(2):
        for flat_index in range(current.inputs[input_index].size):
            array = np.array(current.inputs[input_index], copy=True, order="C")
            flat = array.reshape(-1)
            if flat[flat_index] == 0:
                continue
            flat[flat_index] = 0
            candidate = _with_input(current, input_index, array)
            evaluations += 1
            if _failure_signature(candidate, runner) == signature:
                current = candidate

    return current, evaluations


def _with_side(spec: _CaseSpec, side: int) -> _CaseSpec:
    inputs = tuple(_freeze_array(value[:side, :side]) for value in spec.inputs)
    return _CaseSpec(
        dtype=spec.dtype,
        side=side,
        operations=spec.operations,
        inputs=(inputs[0], inputs[1]),
    )


def _with_input(spec: _CaseSpec, index: int, value: np.ndarray) -> _CaseSpec:
    inputs = list(spec.inputs)
    inputs[index] = _freeze_array(value)
    return replace(spec, inputs=(inputs[0], inputs[1]))


def _freeze_array(value: Any) -> np.ndarray:
    array = np.array(value, copy=True, order="C")
    array.setflags(write=False)
    return array
