# Deterministic compiler repro artifacts

`tiny_tensor_compiler.repro_artifact` packages one concrete verified tensor module, the compile configuration that affects deterministic tracing, and the expected compiler trace into one fail-closed JSON document. The workflow is for cross-process compiler-drift reproduction and first-divergence diagnosis; it does not invoke a native compiler.

## Capture and replay

Capture from canonical serialized tensor IR:

```bash
python -m tiny_tensor_compiler.repro_artifact capture module.json repro.json
```

The options that affect the traced execution pipeline are explicit:

```bash
python -m tiny_tensor_compiler.repro_artifact capture \
  module.json repro.json --borrow-inputs --parallel
```

Replay the artifact through the current compiler:

```bash
python -m tiny_tensor_compiler.repro_artifact replay repro.json
```

Replay has a stable automation-oriented exit contract:

- exit `0`: the artifact is valid and the current trace reproduces exactly;
- exit `1`: the artifact is valid but the current compiler trace differs;
- exit `2`: the artifact, embedded module, or embedded trace is unreadable or fails validation.

A valid mismatch reuses the existing trace-diff model and reports the first divergent compiler phase plus the ordered changed phases and deterministic textual differences. Replay therefore localizes compiler drift without compiling or loading generated native code.

Python callers can use `capture_repro_artifact()`, `deserialize_repro_artifact()`, and `replay_repro_artifact()` from `tiny_tensor_compiler.repro_artifact`.

## Version-1 schema

The canonical top-level document contains exactly:

- `format`: `tiny-tensor-compiler-repro`;
- `version`: integer `1`;
- `config`: exactly `borrow_inputs` and `parallel` booleans;
- `module`: canonical versioned tensor-IR JSON;
- `module_sha256`: SHA-256 of the exact UTF-8 `module` string;
- `trace`: canonical compiler-trace JSON;
- `trace_sha256`: SHA-256 of the exact UTF-8 `trace` string;
- `payload_sha256`: SHA-256 of the canonical compact JSON object containing every preceding field except `payload_sha256` itself.

Artifact serialization uses sorted keys, compact separators, UTF-8 text, and no NaN JSON values. Unknown/missing fields, duplicate JSON object keys, unsupported format/version values, malformed digests, non-canonical embedded content, and configuration/type mismatches are rejected rather than normalized silently.

## Fail-closed internal consistency

Deserialization verifies all three integrity layers before replay:

1. the whole-payload digest must match the canonical core document;
2. the embedded module and trace digests must match their exact strings;
3. the embedded module must deserialize and reserialize byte-for-byte canonically, while the embedded trace must pass the existing trace schema, phase-order, per-phase SHA-256, and report validation.

The trace configuration must equal the artifact configuration. Its first `tensor_ir` phase must also be exactly the embedded canonical module document. This prevents a caller from combining an independently valid module with an unrelated independently valid trace and treating the pair as one repro case.

Replay validates the artifact again even when a caller passes an already constructed `CompilerReproArtifact` object. It then deserializes the stored module, reruns the real `trace_module()` pipeline with the captured `borrow_inputs` / `parallel` configuration, and delegates comparison to `compare_trace_json()`.

Unspecialized symbolic modules remain outside the capture boundary because deterministic compiler traces require concrete Buffer IR, Loop IR, and generated C. Callers must capture a concrete specialization if the source program is symbolic.

## Deterministic return-root minimization

`tiny_tensor_compiler.repro_minimizer` adds a bounded reducer for concrete multi-output modules when the caller can supply an explicit reproduction predicate. It does not pretend that a version-1 artifact contains an oracle for how an older compiler would have traced a different, reduced module.

Python callers use `minimize_return_roots(module, predicate)`. The reducer first canonicalizes and verifies the module, requires the initial module to satisfy the predicate, then greedily tries single return-root removals in deterministic right-to-left order. Every candidate is rebuilt into a fresh function from the backward SSA dependency closure of the retained return values while preserving all declared inputs and their dense runtime indices. Fresh rebuilding gives canonical SSA ids instead of relying on in-place erasure.

The returned `ReproMinimizationResult` records the canonical minimized module JSON, original/minimized return counts, predicate attempts, and accepted reductions. The result is **one-minimal with respect to another single return-root removal under that deterministic order**. It is not a claim of globally minimum operation count, input count, tensor extent, or serialized byte size.

The CLI accepts an external predicate command without invoking a shell:

```bash
python -m tiny_tensor_compiler.repro_minimizer \
  module.json minimized.json \
  --predicate python reproduce.py
```

The temporary canonical candidate-module path is appended as the final argument to the predicate command. Predicate exit codes are fail-closed:

- exit `0`: candidate still reproduces;
- exit `1`: candidate does not reproduce;
- any other exit code: predicate infrastructure error, abort minimization.

The minimizer CLI itself exits `0` after a successful minimization, `1` when the initial module does not reproduce, and `2` for malformed IR, unsupported reduction surfaces, or predicate execution failures.

This first minimization phase deliberately retains all input declarations to preserve the runtime-input ABI and only reduces returned roots plus pure dependencies that become unreachable. Concrete known-pure expression/view/reduction operations are rebuildable. Mutation/effect operations such as `copy_into`, `binary_into`, and `binary_inplace`, unknown opcodes, unspecialized symbolic shapes, malformed return structure, and empty-result candidates fail closed rather than being rewritten speculatively. General operation-level delta debugging and effect-aware repro reduction require a separate correctness model.

## Evidence boundary

The three SHA-256 values are corruption/tamper-detection checks for the repro document. They are not publisher signatures, identities, timestamps, rollback protection, transparency evidence, or supply-chain attestations. Those concerns remain on the repository's native-bundle trust surfaces.

An exact replay is also not a proof of semantic equivalence. It proves that the captured compiler representations through `generated_c` reproduce byte-for-byte under the recorded trace configuration. Verifier, differential, metamorphic, native-execution, and conformance tests remain the evidence for executable semantics.

The workflow intentionally stops before native compiler invocation. It does not capture compiler executable identity, host flags beyond the traced configuration, shared-library bytes, persistent-cache state, native loader behavior, or runtime outputs. Extending the artifact across that boundary would require a distinct evidence model rather than silently broadening version 1.

## Phase boundary

The compiler-reproduction surface now includes canonical capture/replay, first-divergence localization, and deterministic predicate-driven return-root minimization. Further minimizer work should only proceed when it can safely reduce another semantic dimension with an explicit preservation rule—for example operation-level reduction under a trusted external oracle or effect-aware reduction with generation/dependence proofs. Farming return-order variants, extra counters, or cosmetic CLI switches is not the next milestone.
