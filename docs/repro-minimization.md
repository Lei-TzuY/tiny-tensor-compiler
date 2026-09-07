# Deterministic repro minimization

`tiny_tensor_compiler.repro_minimizer` reduces concrete verifier-valid tensor IR only when a caller supplies an explicit reproduction predicate. The reducer is diagnostic tooling, not a compiler optimization pass: a candidate is accepted because the trusted predicate says the failure still reproduces, not because the reducer claims the rewritten program is semantically equivalent.

## Return-root reduction

`minimize_return_roots(module, predicate)` is the first reduction layer. It canonicalizes and verifies the input module, requires the original module to reproduce, then tries to remove one returned root at a time in deterministic right-to-left order.

Each candidate is rebuilt from the backward SSA dependency closure of the retained returns while preserving every declared input and its dense runtime index. The rebuilt candidate is verified before the predicate sees it. The result is one-minimal with respect to one additional return-root removal under that deterministic order; it is not a globally minimum IR claim.

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

## CLI composition

The existing external-predicate CLI remains shell-free:

```bash
python -m tiny_tensor_compiler.repro_minimizer \
  module.json minimized.json \
  --predicate python reproduce.py
```

Passing `--reduce-operations` composes both reduction layers: return-root reduction runs first, then exact-type operation reduction runs on the resulting verified module using the same external predicate contract.

```bash
python -m tiny_tensor_compiler.repro_minimizer \
  module.json minimized.json \
  --reduce-operations \
  --predicate python reproduce.py
```

For every predicate invocation, the reducer writes a fresh canonical candidate module to a temporary file and appends that path as the final argv element. No shell is used. Predicate exit code `0` means reproduced, `1` means not reproduced, and any other exit code aborts as predicate infrastructure failure.

The minimizer CLI exits `0` after successful reduction, `1` when the initial module does not reproduce, and `2` for malformed or unsupported IR and predicate execution failures.

## Fail-closed boundary

Both reduction layers deliberately reject mutation/effect operations such as `copy_into`, `binary_into`, and `binary_inplace`, unknown operation kinds, unspecialized symbolic result types, malformed return structure, and any candidate that fails ordinary IR verification. Operation reduction additionally refuses operands whose complete `TensorType` differs from the result, including otherwise broadcast-compatible operands.

All input declarations are retained even when a reduced repro no longer depends on them. This preserves the runtime-input ABI and dense input indices so a reproduction harness can keep the same invocation contract while the internal failure case shrinks.

Effect-aware reduction requires a separate generation/dependence proof. Input deletion or renumbering requires an explicit ABI-preservation model. Constant synthesis, tensor-extent shrinking, attribute delta debugging, and globally minimum IR search are also separate future dimensions; they are not silently folded into this reducer.

## Evidence boundary

Determinism means identical canonical input plus identical predicate behavior yields the same candidate order and minimized canonical JSON. One-minimality means no single additional reduction from the supported relation succeeds at termination. Neither property means the result is globally smallest.

The reproduction predicate is the semantic oracle for operation substitution. A predicate that accepts an unrelated program can therefore produce an unrelated minimized repro; this is intentional and explicit. The verifier protects IR well-formedness and type/layout invariants, while the caller-owned predicate defines preservation of the failure being investigated.
