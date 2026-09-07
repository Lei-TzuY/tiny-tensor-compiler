from __future__ import annotations

import argparse
import copy
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
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


@dataclass(frozen=True)
class OperationMinimizationResult:
    """Canonical one-minimal exact-type operation-substitution reduction."""

    module_json: str
    original_operation_count: int
    minimized_operation_count: int
    attempts: int
    accepted_reductions: int

    @property
    def module(self) -> Module:
        return deserialize_module(self.module_json)


@dataclass(frozen=True)
class EffectMinimizationResult:
    """Canonical one-minimal storage-generation rollback reduction."""

    module_json: str
    original_effect_count: int
    minimized_effect_count: int
    attempts: int
    accepted_reductions: int

    @property
    def module(self) -> Module:
        return deserialize_module(self.module_json)


def minimize_return_roots(
    module: Module,
    predicate: Callable[[Module], bool],
    *,
    allow_effects: bool = False,
) -> ReproMinimizationResult:
    """Greedily remove returned roots while a caller-supplied repro predicate remains true.

    Each candidate is rebuilt from the backward SSA dependency closure of the selected
    return values while retaining all declared inputs.  This reassigns canonical SSA ids
    instead of mutating the original module in place.  By default mutation/effect
    operations fail closed.  ``allow_effects=True`` permits the known generation-producing
    mutation operations so the verified SSA dependency closure can retain exactly the
    effect generations required by the selected returns.

    The result is deterministic and one-minimal with respect to single return-root
    removal in right-to-left order.  It is not a claim of globally minimum IR size.
    """
    if not isinstance(module, Module):
        raise TypeError("minimize_return_roots requires a Module")
    if not callable(predicate):
        raise TypeError("predicate must be callable")
    if not isinstance(allow_effects, bool):
        raise TypeError("allow_effects must be a bool")

    canonical = serialize_module(module)
    validated = deserialize_module(canonical)
    _validate_reducible_module(validated, allow_effects=allow_effects)
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


def minimize_operations(
    module: Module,
    predicate: Callable[[Module], bool],
    *,
    allow_effects: bool = False,
) -> OperationMinimizationResult:
    """Greedily substitute pure operations with exact-typed operands under a predicate.

    Only a single-result known-pure operation whose result type exactly equals one of its
    operand types is a candidate.  Candidates are considered deterministically from the
    end of the function toward the beginning and from the rightmost operand toward the
    leftmost.  Every accepted candidate is rebuilt as fresh SSA, preserves every declared
    input and its dense index, and is reverified before the trusted predicate observes it.

    By default mutation/effect operations still fail closed.  With ``allow_effects=True``
    the known effects are preserved as non-candidates while pure dependencies around them
    may be reduced.  A substitution that would violate storage-generation or alias
    freshness is rejected by the ordinary verifier before the predicate is called.

    The result is one-minimal with respect to one additional exact-type operand
    substitution under this deterministic order.  It is not a superoptimizer and does
    not claim a globally minimum operation count or semantic equivalence without the
    caller-supplied reproduction predicate.
    """
    if not isinstance(module, Module):
        raise TypeError("minimize_operations requires a Module")
    if not callable(predicate):
        raise TypeError("predicate must be callable")
    if not isinstance(allow_effects, bool):
        raise TypeError("allow_effects must be a bool")

    canonical = serialize_module(module)
    validated = deserialize_module(canonical)
    _validate_reducible_module(validated, allow_effects=allow_effects)
    original_operation_count = _pure_operation_count(validated)

    if not _predicate_holds(canonical, predicate):
        raise InitialReproductionMissing("initial module does not satisfy reproduction predicate")

    current_json = canonical
    attempts = 0
    accepted_reductions = 0

    while True:
        current = deserialize_module(current_json)
        reduced = False
        for op_index, operand_index in _operation_substitutions(current):
            attempts += 1
            try:
                candidate = _substitute_operation_with_operand(current, op_index, operand_index)
            except ValueError:
                if allow_effects:
                    continue
                raise
            candidate_json = serialize_module(candidate)
            if not _predicate_holds(candidate_json, predicate):
                continue
            current_json = candidate_json
            accepted_reductions += 1
            reduced = True
            break

        if not reduced:
            break

    minimized = deserialize_module(current_json)
    return OperationMinimizationResult(
        module_json=current_json,
        original_operation_count=original_operation_count,
        minimized_operation_count=_pure_operation_count(minimized),
        attempts=attempts,
        accepted_reductions=accepted_reductions,
    )


def minimize_effects(
    module: Module,
    predicate: Callable[[Module], bool],
) -> EffectMinimizationResult:
    """Greedily remove verified mutation generations under an explicit repro predicate.

    One candidate removes a single known effect operation and maps that operation's fresh
    full-root generation result back to its exact-typed pre-write root operand.  The whole
    module is rebuilt as fresh SSA and passed through the ordinary verifier.  Candidates
    that violate storage-generation, alias-freshness, dominance, or any other verifier
    invariant are rejected before the trusted predicate observes them.

    Acceptance means only that the caller-supplied predicate still reproduces after this
    bounded generation rollback.  It is not a claim that dropping the write is generally
    semantics-preserving, and it does not reorder effects or synthesize replacements.
    """
    if not isinstance(module, Module):
        raise TypeError("minimize_effects requires a Module")
    if not callable(predicate):
        raise TypeError("predicate must be callable")

    canonical = serialize_module(module)
    validated = deserialize_module(canonical)
    _validate_reducible_module(validated, allow_effects=True)
    original_effect_count = _effect_operation_count(validated)

    if not _predicate_holds(canonical, predicate):
        raise InitialReproductionMissing("initial module does not satisfy reproduction predicate")

    current_json = canonical
    attempts = 0
    accepted_reductions = 0

    while True:
        current = deserialize_module(current_json)
        reduced = False
        for op_index in _effect_substitutions(current):
            attempts += 1
            try:
                candidate = _substitute_effect_with_root(current, op_index)
            except ValueError:
                continue
            candidate_json = serialize_module(candidate)
            if not _predicate_holds(candidate_json, predicate):
                continue
            current_json = candidate_json
            accepted_reductions += 1
            reduced = True
            break

        if not reduced:
            break

    minimized = deserialize_module(current_json)
    return EffectMinimizationResult(
        module_json=current_json,
        original_effect_count=original_effect_count,
        minimized_effect_count=_effect_operation_count(minimized),
        attempts=attempts,
        accepted_reductions=accepted_reductions,
    )


def main(argv: list[str] | None = None) -> int:
    """Minimize serialized tensor IR with an explicit cross-process predicate command."""
    parser = argparse.ArgumentParser(
        description="Deterministically minimize compiler repro modules with a trusted predicate."
    )
    parser.add_argument("module", help="path to canonical serialized tensor IR")
    parser.add_argument("output", help="path to write minimized canonical tensor IR")
    parser.add_argument(
        "--reduce-operations",
        action="store_true",
        help="after return-root reduction, minimize exact-type pure operations by substitution",
    )
    parser.add_argument(
        "--reduce-effects",
        action="store_true",
        help=(
            "allow verified mutation generations during reduction and then minimize them "
            "by predicate-guarded generation rollback"
        ),
    )
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

    operation_result: OperationMinimizationResult | None = None
    effect_result: EffectMinimizationResult | None = None
    try:
        module_document = Path(args.module).read_text(encoding="utf-8")
        module = deserialize_module(module_document)
        predicate = _external_predicate(tuple(args.predicate))
        result = minimize_return_roots(
            module,
            predicate,
            allow_effects=args.reduce_effects,
        )
        current = result.module
        output_json = result.module_json
        if args.reduce_operations:
            operation_result = minimize_operations(
                current,
                predicate,
                allow_effects=args.reduce_effects,
            )
            current = operation_result.module
            output_json = operation_result.module_json
        if args.reduce_effects:
            effect_result = minimize_effects(current, predicate)
            output_json = effect_result.module_json
        Path(args.output).write_text(output_json, encoding="utf-8")
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

    summary = (
        f"returns: {result.original_return_count} -> {result.minimized_return_count}; "
        f"attempts: {result.attempts}; accepted reductions: {result.accepted_reductions}"
    )
    if operation_result is not None:
        summary += (
            f"; operations: {operation_result.original_operation_count} -> "
            f"{operation_result.minimized_operation_count}; operation attempts: "
            f"{operation_result.attempts}; operation reductions: "
            f"{operation_result.accepted_reductions}"
        )
    if effect_result is not None:
        summary += (
            f"; effects: {effect_result.original_effect_count} -> "
            f"{effect_result.minimized_effect_count}; effect attempts: "
            f"{effect_result.attempts}; effect reductions: "
            f"{effect_result.accepted_reductions}"
        )
    print(summary)
    return 0


def _validate_reducible_module(module: Module, *, allow_effects: bool = False) -> None:
    verify(module)
    for op in module.function.ops:
        if op.opcode in {"input", "return"}:
            continue
        if op.opcode in _EFFECT_OPCODES:
            if not allow_effects:
                raise ReproMinimizationError(
                    f"effectful opcode {op.opcode!r} is outside repro minimization"
                )
        elif op.opcode not in _PURE_OPCODES:
            raise ReproMinimizationError(
                f"unsupported opcode {op.opcode!r} is outside repro minimization"
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


def _operation_substitutions(module: Module) -> Iterator[tuple[int, int]]:
    ops = module.function.ops
    for op_index in reversed(range(len(ops))):
        op = ops[op_index]
        if op.opcode not in _PURE_OPCODES or op.opcode == "const" or len(op.results) != 1:
            continue
        result_type = op.results[0].type
        for operand_index in reversed(range(len(op.operands))):
            if op.operands[operand_index].type == result_type:
                yield op_index, operand_index


def _substitute_operation_with_operand(
    module: Module,
    op_index: int,
    operand_index: int,
) -> Module:
    ops = module.function.ops
    if op_index < 0 or op_index >= len(ops):
        raise ReproMinimizationError("operation reduction index is out of range")
    target = ops[op_index]
    if target.opcode not in _PURE_OPCODES or target.opcode == "const" or len(target.results) != 1:
        raise ReproMinimizationError("operation is not eligible for operand substitution")
    if operand_index < 0 or operand_index >= len(target.operands):
        raise ReproMinimizationError("operation operand reduction index is out of range")
    replacement = target.operands[operand_index]
    if replacement.type != target.results[0].type:
        raise ReproMinimizationError("operation substitution requires an exact result/operand type")

    rebuilt = Function(module.function.name)
    values: dict[Value, Value] = {}
    for index, op in enumerate(ops):
        if op.opcode == "return":
            try:
                returns = [values[operand] for operand in op.operands]
            except KeyError as exc:
                raise RuntimeError("internal error: repro operation substitution lost a return") from exc
            rebuilt.add_op("return", operands=returns)
            continue

        if index == op_index:
            try:
                values[op.results[0]] = values[replacement]
            except KeyError as exc:
                raise RuntimeError(
                    "internal error: repro operation replacement does not dominate target"
                ) from exc
            continue

        try:
            operands = [values[operand] for operand in op.operands]
        except KeyError as exc:
            raise RuntimeError("internal error: repro operation substitution lost an operand") from exc
        cloned = rebuilt.add_op(
            op.opcode,
            operands=operands,
            result_types=[result.type for result in op.results],
            attrs=copy.deepcopy(op.attrs),
        )
        for source, destination in zip(op.results, cloned.results, strict=True):
            values[source] = destination

    result = Module(rebuilt)
    verify(result)
    return result


def _effect_substitutions(module: Module) -> Iterator[int]:
    for op_index in reversed(range(len(module.function.ops))):
        if module.function.ops[op_index].opcode in _EFFECT_OPCODES:
            yield op_index


def _substitute_effect_with_root(module: Module, op_index: int) -> Module:
    ops = module.function.ops
    if op_index < 0 or op_index >= len(ops):
        raise ReproMinimizationError("effect reduction index is out of range")
    target = ops[op_index]
    if target.opcode not in _EFFECT_OPCODES or len(target.results) != 1 or not target.operands:
        raise ReproMinimizationError("operation is not an eligible mutation generation")
    root = target.operands[0]
    if root.type != target.results[0].type:
        raise ReproMinimizationError("effect rollback requires an exact result/root type")

    rebuilt = Function(module.function.name)
    values: dict[Value, Value] = {}
    for index, op in enumerate(ops):
        if op.opcode == "return":
            try:
                returns = [values[operand] for operand in op.operands]
            except KeyError as exc:
                raise RuntimeError("internal error: repro effect rollback lost a return") from exc
            rebuilt.add_op("return", operands=returns)
            continue

        if index == op_index:
            try:
                values[op.results[0]] = values[root]
            except KeyError as exc:
                raise RuntimeError(
                    "internal error: repro effect root does not dominate mutation"
                ) from exc
            continue

        try:
            operands = [values[operand] for operand in op.operands]
        except KeyError as exc:
            raise RuntimeError("internal error: repro effect rollback lost an operand") from exc
        cloned = rebuilt.add_op(
            op.opcode,
            operands=operands,
            result_types=[result.type for result in op.results],
            attrs=copy.deepcopy(op.attrs),
        )
        for source, destination in zip(op.results, cloned.results, strict=True):
            values[source] = destination

    result = Module(rebuilt)
    verify(result)
    return result


def _pure_operation_count(module: Module) -> int:
    return sum(op.opcode in _PURE_OPCODES for op in module.function.ops)


def _effect_operation_count(module: Module) -> int:
    return sum(op.opcode in _EFFECT_OPCODES for op in module.function.ops)


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
