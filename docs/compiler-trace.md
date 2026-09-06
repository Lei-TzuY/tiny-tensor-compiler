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

## Stability and evidence boundary

Phase fingerprints are intended for deterministic regression snapshots, differential debugging, and localizing compiler drift. They are not a compatibility promise across compiler versions: an intentional lowering or code-generation change is expected to change the affected phase text and digest.

The SHA-256 values identify the captured text only. They are not publisher signatures, supply-chain attestations, native-artifact trust decisions, or proofs of semantic equivalence. Likewise, equality of two traces is strong evidence that the captured compiler representations are byte-for-byte identical, but it is not a substitute for verifier, differential, native-execution, or conformance tests.

The compiler-report and trace APIs are complementary:

- use `analyze_module()` / `CompilerReport` for stable structural facts and admission-policy inputs;
- use `trace_module()` / `CompilerTrace` when an exact phase-by-phase snapshot is needed to explain or lock down a compiler transformation.

## Phase boundary

This milestone closes the first deterministic compiler-tracing layer without changing execution semantics or optimizer policy. Future observability work should build on these canonical snapshots—for example targeted trace comparison or reproducibility tooling—only when it adds an executable diagnostic workflow rather than another statistics-only surface.
