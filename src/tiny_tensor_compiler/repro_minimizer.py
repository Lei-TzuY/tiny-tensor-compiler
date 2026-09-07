from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .ir import Function, Module, Operation, Value
from .reduction import REDUCTION_OPCODES
from .serialization import IRSerializationError, deserialize_module, serialize_module
from .verifier import verify

_PURE_OPCODES = frozenset(
    {
        "const",
        "add",
        "mul",
        "relu",
        *REDUCTION_OPCODES,
        "reshape",
        "view",
        "slice",
        "reverse",
        "transpose",
    }
)
_EFFECT_OPCODES = frozenset({"copy_into", "binary_into", "binary_inplace"})


class ReproMinimizationError(ValueError):
    """Raised when a deterministic repro reduction cannot be performed safely."""


class InitialReproductionMissing(ReproMinimizationError):
    """Raised when the original module does not satisfy the reproduction predicate."""


class PredicateExecutionError(ReproMinimizationError):
    """Raised when an external reproduction predicate cannot produce a verdict."""


@dataclass(frozen=True)
class ReproMinimizationResult:
    """Canonical one-minimal return-root reduction of one verified concrete module."""

    module_json: str
    original_return_count: int
    minimized_return_count: int
    attempts: int
    accepted_reductions: int

    @property
    def module(self) -> Module:
        return deserialize_module(self.module_json)


def minimize_return_roots(
    module: Module,
    predicate: Callable[[Module], bool],
) -> ReproMinimizationResult:
    """Greedily remove returned roots while a caller-supplied repro predicate remains true.

    Each candidate is rebuilt from the backward SSA dependency closure of the selected
    return values while retaining all declared inputs.  This reassigns canonical SSA ids
    instead of mutating the original module in place.  The current phase deliberately
    accepts only concrete modules over the known pure operation surface; mutation/effect
    operations and unknown opcodes fail closed.

    The result is deterministic and one-minimal with respect to single return-root
    removal in right-to-left order.  It is not a claim of globally minimum IR size.
    """
    if not isinstance(module, Module):
        raise TypeError("minimize_return_roots requires a Module")
    if not callable(predicate):
        raise TypeError("predicate must be callable")

    canonical = serialize_module(module)
    validated = deserialize_module(canonical)
    _validate_reducible_module(validated)
    original_return_count = len(_return_op(validated).operands)

    if not _predicate_holds(canonical, predicate):
        raise InitialReproductionMissing("initial module does not satisfy reproduction predicate")

    current_json = canonical
    attempts = 0
    accepted_reductions = 0

    while True:
        current = deserialize_module(current_json)
        return_count = len(_return_op(current).operands)
        if return_count <= 1:
            break

        reduced = False
        for removed_index in reversed(range(return_count)):
            selected = tuple(index for index in range(return_count) if index != removed_index)
            candidate = _slice_to_return_roots(current, selected)
            candidate_json = serialize_module(candidate)
            attempts += 1
            if not _predicate_holds(candidate_json, predicate):
                continue
            current_json = candidate_json
            accepted_reductions += 1
            reduced = True
            break

        if not reduced:
            break

    minimized = deserialize_module(current_json)
    return ReproMinimizationResult(
        module_json=current_json,
        original_return_count=original_return_count,
        minimized_return_count=len(_return_op(minimized).operands),
        attempts=attempts,
        accepted_reductions=accepted_reductions,
    )


def main(argv: list[str] | None = None) -> int:
    """Minimize serialized tensor IR with an explicit cross-process predicate command."""
    parser = argparse.ArgumentParser(
        description="Deterministically minimize multi-output compiler repro modules by return root."
    )
    parser.add_argument("module", help="path to canonical serialized tensor IR")
    parser.add_argument("output", help="path to write minimized canonical tensor IR")
    parser.add_argument(
        "--predicate",
        nargs=argparse.REMAINDER,
        help=(
            "predicate command; candidate module path is appended as the final argument; "
            "exit 0 means reproduced, 1 means not reproduced, any other code is an error"
        ),
    )
    args = parser.parse_args(argv)

    if not args.predicate:
        parser.error("--predicate requires a command")

    try:
        module_document = Path(args.module).read_text(encoding="utf-8")
        module = deserialize_module(module_document)
        predicate = _external_predicate(tuple(args.predicate))
        result = minimize_return_roots(module, predicate)
        Path(args.output).write_text(result.module_json, encoding="utf-8")
    except InitialReproductionMissing as exc:
        print(f"not reproduced: {exc}", file=sys.stderr)
        return 1
    except (
        IRSerializationError,
        PredicateExecutionError,
        ReproMinimizationError,
        OSError,
        UnicodeError,
    ) as exc:
        print(f"repro minimization failed: {exc}", file=sys.stderr)
        return 2

    print(
        f"returns: {result.original_return_count} -> {result.minimized_return_count}; "
        f"attempts: {result.attempts}; accepted reductions: {result.accepted_reductions}"
    )
    return 0


def _validate_reducible_module(module: Module) -> None:
    verify(module)
    for op in module.function.ops:
        if op.opcode in {"input", "return"}:
            continue
        if op.opcode in _EFFECT_OPCODES:
            raise ReproMinimizationError(
                f"effectful opcode {op.opcode!r} is outside return-root minimization"
            )
        if op.opcode not in _PURE_OPCODES:
            raise ReproMinimizationError(
                f"unsupported opcode {op.opcode!r} is outside return-root minimization"
            )
        if any(not result.type.is_static for result in op.results):
            raise ReproMinimizationError("repro minimization requires concrete tensor shapes")

    for op in module.function.ops:
        for result in op.results:
            if not result.type.is_static:
                raise ReproMinimizationError("repro minimization requires concrete tensor shapes")

    _return_op(module)


def _return_op(module: Module) -> Operation:
    returns = [op for op in module.function.ops if op.opcode == "return"]
    if len(returns) != 1:
        raise ReproMinimizationError("repro minimization requires exactly one return operation")
    result = returns[0]
    if result is not module.function.ops[-1]:
        raise ReproMinimizationError("repro minimization requires the return operation to be final")
    if not result.operands:
        raise ReproMinimizationError("repro minimization requires at least one returned value")
    return result


def _predicate_holds(document: str, predicate: Callable[[Module], bool]) -> bool:
    candidate = deserialize_module(document)
    verdict = predicate(candidate)
    if type(verdict) is not bool:
        raise TypeError("predicate must return a bool")
    return verdict


def _slice_to_return_roots(module: Module, selected_indices: Sequence[int]) -> Module:
    return_op = _return_op(module)
    if not selected_indices:
        raise ReproMinimizationError("return-root reduction must keep at least one result")
    if len(set(selected_indices)) != len(selected_indices):
        raise ReproMinimizationError("return-root reduction indices must be unique")
    if any(index < 0 or index >= len(return_op.operands) for index in selected_indices):
        raise ReproMinimizationError("return-root reduction index is out of range")

    selected_values = tuple(return_op.operands[index] for index in selected_indices)
    needed_ops = _dependency_closure(selected_values)
    rebuilt = Function(module.function.name)
    values: dict[Value, Value] = {}

    for op in module.function.ops:
        if op.opcode == "return":
            continue
        if op.opcode != "input" and op not in needed_ops:
            continue
        try:
            operands = [values[operand] for operand in op.operands]
        except KeyError as exc:
            raise RuntimeError("internal error: repro dependency closure is incomplete") from exc
        cloned = rebuilt.add_op(
            op.opcode,
            operands=operands,
            result_types=[result.type for result in op.results],
            attrs=copy.deepcopy(op.attrs),
        )
        for source, destination in zip(op.results, cloned.results, strict=True):
            values[source] = destination

    try:
        rebuilt_returns = [values[value] for value in selected_values]
    except KeyError as exc:
        raise RuntimeError("internal error: selected repro root was not rebuilt") from exc
    rebuilt.add_op("return", operands=rebuilt_returns)
    result = Module(rebuilt)
    verify(result)
    return result


def _dependency_closure(values: Sequence[Value]) -> frozenset[Operation]:
    needed: set[Operation] = set()
    pending = list(values)
    while pending:
        value = pending.pop()
        producer = value.producer
        if producer is None or producer.opcode == "input" or producer in needed:
            continue
        needed.add(producer)
        pending.extend(producer.operands)
    return frozenset(needed)


def _external_predicate(command: tuple[str, ...]) -> Callable[[Module], bool]:
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise PredicateExecutionError("predicate command must contain non-empty argv strings")

    def predicate(module: Module) -> bool:
        with tempfile.TemporaryDirectory(prefix="tiny-tensor-repro-") as directory:
            candidate = Path(directory) / "candidate.json"
            candidate.write_text(serialize_module(module), encoding="utf-8")
            try:
                completed = subprocess.run(
                    [*command, str(candidate)],
                    check=False,
                )
            except OSError as exc:
                raise PredicateExecutionError(f"cannot execute predicate command: {exc}") from exc
        if completed.returncode == 0:
            return True
        if completed.returncode == 1:
            return False
        raise PredicateExecutionError(
            f"predicate command exited {completed.returncode}; expected 0 or 1"
        )

    return predicate


if __name__ == "__main__":
    raise SystemExit(main())
