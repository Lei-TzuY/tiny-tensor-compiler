# Deterministic repro minimization

`tiny_tensor_compiler.repro_minimizer` reduces concrete verifier-valid tensor IR only when a caller supplies an explicit reproduction predicate. The reducer is diagnostic tooling, not a compiler optimization pass: a candidate is accepted because the trusted predicate says the failure still reproduces, not because the reducer claims the rewritten program is semantically equivalent.

## Return-root reduction

`minimize_return_roots(module, predicate)` is the first reduction layer. It canonicalizes and verifies the input module, requires the original module to reproduce, then tries to remove one returned root at a time in deterministic right-to-left order.

Each candidate is rebuilt from the backward SSA dependency closure of the retained returns while preserving every declared input and its dense runtime index. The rebuilt candidate is verified before the predicate sees it. The result is one-minimal with respect to one additional return-root removal under that deterministic order; it is not a globally minimum IR claim.

By default this API still fails closed when the module contains mutation/effect operations. Passing `allow_effects=True` admits only the known generation-producing effects (`copy_into`, `binary_into`, and `binary_inplace`). The same backward dependency closure then retains every effect generation required by a selected return and may prune an entire unreachable effect chain when a returned root is removed. The ordinary verifier remains the generation/freshness proof for every rebuilt candidate.

## Exact-type operation reduction

`minimize_operations(module, predicate)` adds a second, deliberately bounded reduction relation. A candidate operation is eligible only when all of the following hold:

- the module is concrete and passes the normal verifier;
- the operation is on the known pure repro-rebuild surface;
- the operation has exactly one result;
- the operation is not `input`, `return`, or `const`;
- one existing operand has exactly the same complete `TensorType` as the result.

Eligible operations are considered from the end of the function toward the beginning. Within one operation, operands are considered from right to left. A trial substitutes the operation result with that already-dominating operand, rebuilds the whole function as fresh SSA, preserves all declared inputs and attributes on untouched operations, and reruns `verify()` before invoking the predicate.

If the predicate rejects the candidate, the current module is unchanged and the reducer tries the next deterministic candidate. If the predicate accepts it, the canonical rebuilt candidate becomes the new current module and candidate enumeration restarts. Reduction stops only when no single additional exact-type operand substitution preserves the predicate.

`OperationMinimizationResult` reports the canonical minimized module JSON, original/minimized pure-operation counts, predicate attempts, and accepted reductions. The result is one-minimal only with respect to this exact-type operand-substitution relation and deterministic order. It is not a superoptimizer, does not synthesize constants, does not coerce types or shapes, and does not prove semantic equivalence independently of the supplied predicate.

The default API continues to reject effectful modules. With `allow_effects=True`, known effects remain present as non-candidates while pure dependencies around them may be trial-reduced. If a pure substitution would invalidate an effect root, alias generation, stale-view rule, dominance relation, or any other verifier invariant, that candidate is discarded before the reproduction predicate runs.

## Effect-generation reduction

`minimize_effects(module, predicate)` adds a third reduction relation that is explicitly tied to the compiler's storage-generation model.

Every supported mutation operation produces a fresh full-root generation SSA result whose complete `TensorType` is identical to its pre-write root operand. An effect-reduction trial removes one known effect and maps that fresh result back to the exact-typed pre-write root. The entire module is then rebuilt as fresh SSA and passed through the ordinary verifier.

Effects are considered from the end of the function toward the beginning. A candidate that violates generation freshness, storage-root ownership, alias lifetime, dominance, type/layout constraints, or any other verifier invariant is rejected before the predicate. A verifier-valid candidate is accepted only when the trusted predicate still reproduces the failure. Candidate enumeration then restarts from the new canonical module until no additional single generation rollback succeeds.

`EffectMinimizationResult` reports canonical minimized JSON, original/minimized known-effect counts, predicate attempts, and accepted reductions. This is one-minimal only under the exact relation "replace one effect's fresh generation result by its pre-write root" and the deterministic order. It is not a proof that the removed write is generally dead or semantics-preserving.

This phase deliberately does **not** reorder effects, delete arbitrary effect operands, synthesize replacement values, infer alias independence, delete/renumber inputs, or perform compiler optimization. The verifier proves only that the rebuilt IR remains valid under the existing storage-generation model; the reproduction predicate remains the semantic oracle for whether removing that mutation generation preserves the failure.

## CLI composition

The existing external-predicate CLI remains shell-free:

```bash
python -m tiny_tensor_compiler.repro_minimizer \
  module.json minimized.json \
  --predicate python reproduce.py
```

Without effect flags, the historical behavior is unchanged: effectful input is rejected. Passing `--reduce-operations` composes return-root reduction followed by exact-type pure-operation reduction using the same external predicate contract.

```bash
python -m tiny_tensor_compiler.repro_minimizer \
  module.json minimized.json \
  --reduce-operations \
  --predicate python reproduce.py
```

Passing `--reduce-effects` explicitly admits the known mutation surface for return-root reduction, permits pure operation reduction around effects when `--reduce-operations` is also present, and finally applies generation rollback with `minimize_effects()`.

```bash
python -m tiny_tensor_compiler.repro_minimizer \
  module.json minimized.json \
  --reduce-operations \
  --reduce-effects \
  --predicate python reproduce.py
```

For every predicate invocation, the reducer writes a fresh canonical candidate module to a temporary file and appends that path as the final argv element. No shell is used. Predicate exit code `0` means reproduced, `1` means not reproduced, and any other exit code aborts as predicate infrastructure failure.

The minimizer CLI exits `0` after successful reduction, `1` when the initial module does not reproduce, and `2` for malformed or unsupported IR and predicate execution failures.

## Fail-closed boundary

Default return-root and operation reduction still reject mutation/effect operations, unknown operation kinds, unspecialized symbolic result types, malformed return structure, and verifier-invalid candidates. Effect-aware mode must be requested explicitly and admits only the three known mutation opcodes whose generation contract is already checked by the ordinary verifier. Unknown or newly introduced effects remain unsupported rather than being guessed safe.

Operation reduction additionally refuses operands whose complete `TensorType` differs from the result, including otherwise broadcast-compatible operands. Effect rollback requires the fresh effect result and its pre-write root to have exactly the same complete `TensorType`.

All input declarations are retained even when a reduced repro no longer depends on them. This preserves the runtime-input ABI and dense input indices so a reproduction harness can keep the same invocation contract while the internal failure case shrinks.

Input deletion or renumbering requires a separate ABI-preservation model. Constant synthesis, tensor-extent shrinking, attribute delta debugging, effect reordering, and globally minimum IR search are also separate future dimensions; they are not silently folded into these reducers.

## Evidence boundary

Determinism means identical canonical input plus identical predicate behavior yields the same candidate order and minimized canonical JSON. One-minimality means no single additional reduction from the selected supported relation succeeds at termination. Neither property means the result is globally smallest.

The reproduction predicate is the semantic oracle for operation substitution and effect rollback. A predicate that accepts an unrelated program can therefore produce an unrelated minimized repro; this is intentional and explicit. The verifier protects IR well-formedness, type/layout invariants, storage-root generation, and alias freshness, while the caller-owned predicate defines preservation of the failure being investigated.

Core implementation/tests on exact head `ab170a423afbfaec0c917d04815ac6ac8737d0b0` passed the Ubuntu/Windows × Python 3.11/3.13 CI matrix in run `34077477458`. This evidence establishes deterministic verifier-backed reduction behavior; it is not a claim that removing effects preserves general program semantics outside the supplied reproduction predicate.
