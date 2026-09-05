# Native OpenMP scheduling

This document defines the first bounded parallel-execution phase. The goal is to add a real native scheduling capability without changing tensor semantics, reordering kernels, or making an unsupported performance claim.

## Entry points

Parallel native execution is opt-in through the existing public surfaces:

```python
compile_native(loop_program, parallel=True)
execute_native(loop_program, inputs=inputs, parallel=True)
compile_module(module, parallel=True)
compile_dynamic_module(module, parallel=True)
```

`parallel=False` remains the default. Dynamic executables freeze the selected mode and forward it independently to every concrete symbolic specialization.

## Scheduling boundary

Parallelism starts only after the existing tensor, buffer, memory-planning, loop-IR, and fusion correctness boundaries have run. The OpenMP emitter therefore receives verified concrete Loop IR with explicit physical buffers and index maps.

For each general-C `LoopKernel`:

- scalar and zero-extent kernels remain serial;
- a linearized non-empty kernel schedules its row-major `n` loop with `#pragma omp parallel for schedule(static)`;
- a non-empty broadcast/nested kernel schedules only the outer `i0` loop and leaves inner loops nested inside that iteration;
- each parallel loop retains the default implicit OpenMP barrier;
- input-copy operations and terminal output-copy operations remain serial;
- explicit SSE2 kernels remain on their existing serial vector-loop plus scalar-tail implementation in this phase.

The per-kernel implicit barrier is part of the correctness contract. A later kernel cannot begin consuming a physical buffer until every scheduled iteration of its producer has completed. This phase does not use `nowait`, task graphs, asynchronous returns, kernel reordering, or speculative overlap.

Loop IR already prohibits a kernel output from aliasing an input read by that same kernel. Parallel scheduling does not weaken that invariant or create a separate alias-analysis path.

## Compiler integration

The generated source uses ordinary OpenMP pragmas and keeps the existing C ABI unchanged. The native compiler command is extended only in parallel mode:

- GCC-style toolchains: `-fopenmp`
- MSVC: `/openmp`

MSVC requires the OpenMP induction variable to be declared outside the canonical `for` header. The parallel emitter therefore produces forms such as:

```c
int64_t n;
#pragma omp parallel for schedule(static)
for (n = 0; n < count; ++n) {
    ...
}
```

and similarly externalizes `i0` for broadcast outer loops. The serial emitter is unchanged.

The OpenMP flag and parallel generated source naturally participate in the existing native artifact identity, so a serial artifact cannot be confused with its parallel counterpart.

## Windows generated-DLL lifetime

The first MSVC regression exposed a native lifetime hazard rather than an arithmetic error: unloading a generated OpenMP DLL immediately after use could race with runtime worker threads that still retained outlined loop code, producing a Windows access violation during `FreeLibrary` / staging-directory cleanup.

Windows OpenMP artifacts therefore have a stricter lifetime than ordinary serial artifacts:

- a compiled OpenMP artifact is removed from the ordinary unloadable process cache;
- the artifact is retained in a process-pinned registry for the rest of the Python process;
- `clear_native_cache()` does not unload that generated OpenMP DLL;
- the staging directory receives a PID marker;
- a later process may best-effort remove a marked directory only when the recorded owner PID is no longer live.

Serial Windows artifacts keep the existing release/reacquire behavior. The process-pinned policy is limited to generated OpenMP DLLs and exists to preserve executable-code lifetime, not to broaden the general cache contract.

## Evidence boundary

The regression suite exercises:

- serial-vs-parallel generated-source separation;
- linearized and broadcast-loop OpenMP structure;
- scalar, zero-extent, and explicit-SSE2 fallback boundaries;
- package-root `compile_native(..., parallel=True)` and `execute_native(..., parallel=True)`;
- high-level concrete compilation;
- symbolic specialization and specialization reuse;
- broadcast + ordered multi-output + verified borrowed inputs;
- reusable parallel execution before and after `clear_native_cache()`;
- GCC-style and MSVC native compile/load/execute paths in CI.

This establishes executable scheduling semantics and cross-platform correctness. It does **not** establish that parallel mode is faster for a particular tensor size or workload. No thread-count policy, grain-size threshold, controlled speedup benchmark, or profitability model is claimed by this phase.

## Phase boundary

This phase stops at one barriered native loop-scheduling policy. The next milestone should not be a stream of pragma or thread-count micro-tweaks without controlled benchmark evidence. Higher-value promotion candidates are a shape-transform/reshape subsystem with new symbolic semantics, a second executable ISA/backend that can justify an ISA-neutral vector plan, or an accelerator backend.