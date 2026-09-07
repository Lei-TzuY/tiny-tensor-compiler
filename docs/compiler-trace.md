# Deterministic compiler phase traces

`trace_module()` exposes the compiler's concrete lowering path as exact text snapshots plus SHA-256 fingerprints. It is an observability and regression-diagnosis surface: the trace runs the real lowering, memory-planning, fusion, input-binding, and C-generation code paths without invoking an external native compiler.

```python
from tiny_tensor_compiler import trace_module

trace = trace_module(module, borrow_inputs=True, parallel=False)
for phase in trace.phases:
    print(phase.name, phase.sha256)

snapshot = trace.to_json()
```

## Phase order

A trace contains these phases in deterministic order:

1. `tensor_ir` — canonical serialized verified tensor IR.
2. `buffer_ir` — virtual-buffer CPU IR after ordinary lowering.
3. `memory_plan` — the deterministic physical-storage assignment/lifetime plan.
4. `pre_fusion_loop_ir` — explicit Loop IR before elementwise fusion.
5. `post_fusion_loop_ir` — Loop IR after the repository's sole fusion planner.
6. `execution_loop_ir` — the executable Loop IR, including explicit borrowed-input bindings when `borrow_inputs=True`.
7. `generated_c` — deterministic C source for the selected execution configuration.

Each `CompilerTracePhase` stores the exact UTF-8 `text` and `sha256(text.encode("utf-8"))`. `CompilerTrace.to_json()` uses sorted keys and compact separators so two equal traces serialize byte-for-byte identically.

The trace also embeds the ordinary `CompilerReport` returned by `analyze_module()`. Reports remain the compact structural summary; phase traces add exact snapshots for locating where two compilations begin to diverge.

## Configuration boundaries

Trace options are deliberately narrow and make their first affected phase explicit.

- `parallel=True` leaves tensor IR, Buffer IR, the memory plan, and both Loop IR snapshots unchanged. It changes only generated C scheduling/source.
- `borrow_inputs=True` leaves the first five phases unchanged. It changes `execution_loop_ir` by recording the verified borrowed-input bindings and consequently changes generated C input materialization.
- Unspecialized symbolic modules are rejected. Callers must first produce a concrete specialization, preserving the existing rule that Buffer IR, Loop IR, and generated C are concrete-shape compiler layers.

A trace never compiles or loads a shared library. Native compiler choice, subprocess timeout, persistent-cache leases, bundle publication, and host-toolchain execution therefore remain outside this API.

## Deterministic comparison and repro diagnosis

Stored v1 trace snapshots can be compared directly without rerunning the compiler. The comparison workflow validates both files before reporting a difference:

```bash
python -m tiny_tensor_compiler.trace_diff before.json after.json
```

The command has a stable automation-oriented exit contract:

- exit `0`: the validated snapshots are equal;
- exit `1`: the snapshots are valid but differ;
- exit `2`: either input is unreadable or fails the trace format, schema, phase-order, or SHA-256 self-digest contract.

Comparison is based on parsed trace content, so insignificant JSON whitespace or indentation does not create a false difference. Valid differences report configuration changes, whether the embedded `CompilerReport` changed, the first divergent compiler phase, the complete ordered set of changed phases, and deterministic unified diffs for changed phase text.

The first-divergence result follows the trace's real pipeline order. For example:

- comparing otherwise identical serial and `parallel=True` traces first diverges at `generated_c`;
- comparing copied-input and `borrow_inputs=True` traces first diverges at `execution_loop_ir`, with generated C changing afterward.

The stored JSON is treated as untrusted diagnostic input. A phase whose text no longer matches its recorded SHA-256, a reordered/missing phase sequence, an unsupported format/version, malformed top-level fields, or invalid report data is rejected rather than rendered as an ordinary compiler diff.

Python callers can use `compare_trace_json()` or `compare_trace_files()` from `tiny_tensor_compiler.trace_diff` and inspect the structured `CompilerTraceComparison` / `CompilerTracePhaseDiff` values. The package root intentionally remains focused on trace capture; comparison is a tooling submodule and does not alter compilation semantics.

## Repro artifact integration

`tiny_tensor_compiler.repro_artifact` builds directly on this trace contract. A versioned repro document stores one canonical serialized concrete module, the trace-affecting `borrow_inputs` / `parallel` configuration, the expected canonical trace, per-component SHA-256 digests, and a whole-document digest. Replay validates the artifact, deserializes and reverifies the module, reruns `trace_module()`, then delegates first-divergence localization to the same trace-diff implementation.

```bash
python -m tiny_tensor_compiler.repro_artifact capture module.json repro.json
python -m tiny_tensor_compiler.repro_artifact replay repro.json
```

See `docs/compiler-repro-artifact.md` for the exact schema, fail-closed checks, CLI exit contract, and evidence boundary.

## Stability and evidence boundary

Phase fingerprints are intended for deterministic regression snapshots, differential debugging, and localizing compiler drift. They are not a compatibility promise across compiler versions: an intentional lowering or code-generation change is expected to change the affected phase text and digest.

The SHA-256 values identify the captured text only. They are not publisher signatures, supply-chain attestations, native-artifact trust decisions, or proofs of semantic equivalence. Likewise, equality of two traces is strong evidence that the captured compiler representations are byte-for-byte identical, but it is not a substitute for verifier, differential, native-execution, or conformance tests.

A successful comparison or repro replay also does not certify the native compiler or generated artifact because the trace boundary intentionally stops at generated C. Host compiler invocation, timeout/process-tree behavior, cache leases, signed bundles, and native loading retain their own validation surfaces.

The compiler-report, trace, and repro APIs are complementary:

- use `analyze_module()` / `CompilerReport` for stable structural facts and admission-policy inputs;
- use `trace_module()` / `CompilerTrace` to capture exact phase-by-phase compiler state;
- use `tiny_tensor_compiler.trace_diff` when two stored snapshots need deterministic first-divergence localization;
- use `tiny_tensor_compiler.repro_artifact` when the original canonical Module plus trace-affecting configuration must travel with the expected trace and be replayed fail-closed.

## Phase boundary

The deterministic trace-comparison layer and the bounded compiler-repro artifact workflow are now both executable. Further observability work should only continue when it adds a qualitatively new diagnostic capability—such as independently replayable runtime-input/output evidence or deterministic minimization of a valid reproducer—rather than more trace fields, checksums, presentation switches, or statistics-only surfaces.
