from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .repro import (
    ReproMismatchError,
    capture_repro_case,
    load_repro_case,
    replay_repro_case,
)

ReproPredicate = Callable[[str], bool]


class ReproMinimizationError(ValueError):
    """Raised when a repro cannot enter the deterministic minimization contract."""


@dataclass(frozen=True)
class ReproMinimizationResult:
    """Best predicate-preserving runtime-value reduction found deterministically."""

    document: str
    evaluations: int
    original_nonzero: int
    minimized_nonzero: int
    exhausted: bool

    @property
    def changed(self) -> bool:
        return self.minimized_nonzero < self.original_nonzero


def minimize_repro_case(
    document: str,
    predicate: ReproPredicate,
    *,
    max_evaluations: int | None = None,
) -> ReproMinimizationResult:
    """Greedily zero C-order input chunks while a caller predicate still reproduces.

    Every candidate is rebuilt through ``capture_repro_case`` so its expected reference
    outputs always correspond to the candidate inputs. Shapes, dtypes, module IR, and
    input declaration order are never changed by this bounded minimizer.
    """
    if not callable(predicate):
        raise TypeError("predicate must be callable")
    _validate_evaluation_budget(max_evaluations)

    case = load_repro_case(document)
    inputs = [np.array(value, copy=True, order="C") for value in case.inputs]
    current = capture_repro_case(case.module, inputs)
    original_nonzero = _nonzero_count(inputs)
    evaluations = 1
    if not _evaluate_predicate(predicate, current):
        raise ReproMinimizationError("original repro does not satisfy the minimization predicate")

    for input_index in range(len(inputs)):
        size = inputs[input_index].size
        if size == 0:
            continue
        block_size = max(1, size // 2)
        while True:
            for start in range(0, size, block_size):
                end = min(size, start + block_size)
                flat = inputs[input_index].reshape(-1)
                if not np.any(flat[start:end] != 0):
                    continue
                if max_evaluations is not None and evaluations >= max_evaluations:
                    return _result(
                        current,
                        evaluations,
                        original_nonzero,
                        inputs,
                        exhausted=True,
                    )

                candidate_input = np.array(inputs[input_index], copy=True, order="C")
                candidate_input.reshape(-1)[start:end] = 0
                candidate_inputs = list(inputs)
                candidate_inputs[input_index] = candidate_input
                candidate = capture_repro_case(case.module, candidate_inputs)
                evaluations += 1
                if _evaluate_predicate(predicate, candidate):
                    inputs[input_index] = candidate_input
                    current = candidate

            if block_size == 1:
                break
            block_size = max(1, block_size // 2)

    return _result(
        current,
        evaluations,
        original_nonzero,
        inputs,
        exhausted=False,
    )


def minimize_native_mismatch(
    document: str,
    *,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    parallel: bool = False,
    max_evaluations: int | None = None,
) -> ReproMinimizationResult:
    """Minimize input support while an exact native-vs-reference mismatch persists."""
    if not isinstance(parallel, bool):
        raise TypeError("parallel must be a bool")

    def reproduces(candidate: str) -> bool:
        try:
            replay_repro_case(
                candidate,
                backend="native",
                compiler=compiler,
                cache_dir=cache_dir,
                parallel=parallel,
            )
        except ReproMismatchError:
            return True
        return False

    return minimize_repro_case(
        document,
        reproduces,
        max_evaluations=max_evaluations,
    )


def _validate_evaluation_budget(max_evaluations: int | None) -> None:
    if max_evaluations is None:
        return
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int):
        raise TypeError("max_evaluations must be a positive integer or None")
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be at least 1")


def _evaluate_predicate(predicate: ReproPredicate, document: str) -> bool:
    result = predicate(document)
    if type(result) is not bool:
        raise TypeError("predicate must return bool")
    return result


def _nonzero_count(inputs: list[np.ndarray]) -> int:
    return sum(int(np.count_nonzero(value)) for value in inputs)


def _result(
    document: str,
    evaluations: int,
    original_nonzero: int,
    inputs: list[np.ndarray],
    *,
    exhausted: bool,
) -> ReproMinimizationResult:
    return ReproMinimizationResult(
        document=document,
        evaluations=evaluations,
        original_nonzero=original_nonzero,
        minimized_nonzero=_nonzero_count(inputs),
        exhausted=exhausted,
    )
