# Architecture

`tiny-tensor-compiler` is deliberately small, but the `v0.1.0` milestone is a complete compiler vertical slice rather than a collection of disconnected demos.

```mermaid
flowchart LR
    A[Python tensor expressions\n+ static typed inputs]
    B[Typed tensor IR]
    C[Verifier]
    D[Optimization passes\nfold / simplify / DCE / canonicalize / CSE]
    E[Virtual-buffer CPU IR]
    F[Liveness memory planner]
    G[Loop / kernel IR\nexplicit broadcast maps]
    H[Verifier-backed fusion]
    I[Deterministic C11]
    J[Native compiler\nGCC / Clang / MSVC]
    K[Reusable native executable\nprocess + persistent cache]
    R[NumPy reference executor]

    A --> B --> C --> D --> E --> F --> G --> H --> I --> J --> K
    B --> R
```

## Correctness boundaries

The project treats verification and explicit semantics as compiler features, not test-only conveniences.

- Tensor IR verifies operation arity, ordering/dominance, inferred result types, return structure, legal opcodes, constants, input indices, and use-def consistency.
- Buffer IR keeps virtual values single-write and separates virtual liveness from physical slot reuse.
- Memory reuse is allowed only after the previous value's last use and only for an exact `TensorType` match.
- Loop IR makes broadcasting explicit through deterministic index maps; kernels may not overwrite a physical slot still read by the same kernel.
- Integer lowering preserves fixed-width wrapping semantics across the scalar and generated-C paths.
- Floating-point ReLU preserves NaN behavior and canonicalizes negative zero to match the reference semantics.
- Runtime inputs are exact: count, static shape, and dtype must match the compiled graph; there is no silent cast.

## Execution paths

There are intentionally separate execution paths so optimized lowering can be checked against a simpler semantic baseline.

1. **Reference** — executes verified tensor IR with NumPy and defines the semantic baseline.
2. **Loop interpreter** — executes explicit planned loop IR one output index at a time.
3. **Native** — emits deterministic C11, compiles a shared library, and invokes a stable output-first ABI through `ctypes`.

Native code may be reused in-process or persisted by content-addressed cache identity. Persistent library bytes are staged before loading so the cache remains immutable and Windows DLL locking does not poison reusable artifacts.

## Optimization philosophy

`v0.1.0` is correctness-first and conservative by design.

- Algebraic simplification avoids floating-point identities whose IEEE edge cases would change behavior.
- CSE is exact rather than algebraic.
- Fusion is verifier-backed and limited to bounded elementwise shapes with explicit alias/liveness checks.
- Contiguous-loop linearization happens only when identity indexing proves a row-major flat loop equivalent.
- Compiler vectorization hints do not select a vector width or change fallback semantics.
- SSE2 paths are guarded specializations for selected exact contiguous `int32` kernels; unsupported forms fall back to the general generated-C path.

## v0.1.0 scope boundary

Included: static shapes, one returned tensor, explicit external inputs, CPU lowering, scalar reference/interpreter execution, C11 code generation, native compilation, cache reuse, conservative fusion, and selected SIMD specialization.

Deferred: dynamic shapes, multiple outputs, zero-copy input aliasing, generalized SIMD abstraction, general expression-DAG matching, parallel scheduling, and accelerator backends.

This boundary is intentional. After `v0.1.0`, maintenance should prefer correctness and portability fixes over enumerating more opcode × fusion-shape × dtype SIMD micro-variants. A broader capability should start only as an explicitly chosen later milestone.
