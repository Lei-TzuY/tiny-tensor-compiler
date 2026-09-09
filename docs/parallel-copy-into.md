# Verified parallel `copy_into`

This phase extends the existing opt-in native OpenMP scheduler to verified `copy_into` effects without changing the writable-effect IR or the historical serial backend.

## Scheduling contract

`parallel=False` continues to emit the existing deterministic serial copy loop byte-for-byte through `emit_copy_into()`.

With `parallel=True`, `copy_into` is eligible for `#pragma omp parallel for schedule(static)` only when all of the following hold:

- the target is non-scalar;
- the target has a non-zero element count;
- the verified `StorageLayout` conservatively proves that every logical target index reaches a distinct backing-storage element.

Only the outer target axis is scheduled. The existing serial emitter still owns destination addressing, signed strides, source broadcasting, and source-layout indexing. The OpenMP region keeps its implicit barrier, so the fresh storage generation produced by the effect is complete before any later kernel, view, effect, or return executes.

If injectivity cannot be proven, code generation deliberately keeps the serial copy. This preserves the existing set of verifier-legal low-level `copy_into` programs instead of tightening the verifier merely to make parallel code generation easier.

## Layout and source semantics

Parallel scheduling composes with the existing copy semantics rather than replacing them:

- lower-rank and scalar exact-dtype sources retain their canonical `IndexMap` broadcast mapping;
- positive-stride, transposed, and negative-stride target layouts retain their root-relative address calculation;
- borrowed runtime sources remain read-only;
- high-level same-root sources are still snapshotted into distinct owning storage before the low-level effect;
- every actual Loop IR `copy_into` source therefore remains on a different storage root from the destination.

The scheduler does not introduce `memmove`, traversal-direction semantics, a new alias-analysis path, or a second writable-effect representation.

## Cross-platform execution

The effect scheduler reuses the compiler-neutral OpenMP canonical loop form established by the native parallel phase: the induction variable is declared before the pragma and the loop header assigns it. GCC-style toolchains compile the existing `-fopenmp` path and MSVC uses `/openmp`.

Windows OpenMP generated libraries retain the existing process-pinned lifetime. Parallelizing `copy_into` does not create a new DLL ownership model, and `clear_native_cache()` still cannot unload process-pinned generated code that the OpenMP runtime may retain.

## Verified fallbacks

The following remain serial even when the executable was compiled with `parallel=True`:

- scalar copy targets;
- zero-extent copy targets;
- target layouts whose logical indices may overlap in backing storage;
- input materialization and terminal output materialization;
- full-root `binary_inplace` effects.

This is a correctness policy, not a profitability model.

## Evidence boundary

Regression coverage proves generated-source scheduling, broadcast source indexing, deterministic serial fallback for scalar/empty/overlapping layouts, borrowed-source immutability, signed-stride target execution, and native GCC/MSVC execution.

No wall-clock speedup, optimal thread count, grain size, or scheduling profitability claim is made. CI timing is not benchmark evidence.

## Phase promotion

This closes the first verified parallel-copy effect slice. Further OpenMP pragma variants, chunk-size knobs, thread-count knobs, or additional copy-shape examples are not a new architectural milestone. The next storage/mutation frontier should add genuinely new semantics, such as an explicit destination-casting policy or broader dependence-aware scheduling across multiple ordered effects, or the project should promote to another independent compiler subsystem.
