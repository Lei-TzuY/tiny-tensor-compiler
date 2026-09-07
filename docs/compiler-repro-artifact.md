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

## Evidence boundary

The three SHA-256 values are corruption/tamper-detection checks for the repro document. They are not publisher signatures, identities, timestamps, rollback protection, transparency evidence, or supply-chain attestations. Those concerns remain on the repository's native-bundle trust surfaces.

An exact replay is also not a proof of semantic equivalence. It proves that the captured compiler representations through `generated_c` reproduce byte-for-byte under the recorded trace configuration. Verifier, differential, metamorphic, native-execution, and conformance tests remain the evidence for executable semantics.

The workflow intentionally stops before native compiler invocation. It does not capture compiler executable identity, host flags beyond the traced configuration, shared-library bytes, persistent-cache state, native loader behavior, or runtime outputs. Extending the artifact across that boundary would require a distinct evidence model rather than silently broadening version 1.

## Phase boundary

This phase closes the bounded compiler-reproduction workflow built on canonical IR serialization, deterministic phase traces, and first-divergence comparison. Further observability work should only continue if it adds a qualitatively new executable diagnostic capability, such as independently replayable runtime-input/output evidence or deterministic minimization of a valid reproducer. Adding more checksums, presentation switches, or duplicate trace metadata is not the next milestone.
