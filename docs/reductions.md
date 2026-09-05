# Deterministic reductions

This document defines the correctness boundary for the compiler's current reduction subsystem. The supported operators are `sum` and `prod`; they share one `ReductionOperator` / `ReductionPlan` model from typed tensor IR through reference execution, Buffer/Loop IR, generated C, native execution, OpenMP scheduling, serialization, and repro replay.

## Reduction domains

A reduction domain is selected at graph construction time with `axis=`:

- `axis=None` reduces the complete logical tensor in C-order and produces a scalar.
- `axis=<int>` reduces one logical axis and preserves the historical single-axis IR form.
- `axis=<non-empty iterable of ints>` reduces several logical axes at once.

Axis collections are canonicalized before they enter tensor IR. Negative axes are normalized against the input rank, duplicates are rejected after normalization, and the remaining axes are sorted into strictly increasing order. A singleton collection collapses to the historical integer form, so `axis=1`, `axis=(1,)`, and equivalent normalized singleton spellings have one IR/CSE identity. Empty collections, booleans, strings, duplicates, and out-of-range axes are rejected instead of being silently reinterpreted.

For an input shape `(d0, ..., dn)`, a non-full reduction removes every canonical reduced axis from the result shape while preserving the relative order of unreduced axes. The dtype is unchanged.

## Deterministic fold order

Every reduction is an explicit same-dtype left fold. `sum` starts from typed zero and `prod` from typed one. Fixed-width integer arithmetic therefore wraps at every combine boundary; floating-point execution preserves one fixed operation order rather than delegating to a backend-specific tree reduction.

Full-tensor reductions traverse the complete logical tensor in C-order. Single- and multi-axis reductions iterate output coordinates in C-order, then iterate the canonical reduction-domain shape lexicographically. For canonical axes `(a0, a1, ...)`, the coordinate for `a0` is the outermost reduction loop and the last reduced axis varies fastest. Reference execution and the explicit Loop CPU backend use the same `ReductionPlan` index mapping.

Empty reduction extents execute zero combines and return the operator identity for each output coordinate. This is deliberate and shared by reference, Loop, generated-C, and native execution.

## Logical views and physical layout

Reduction axes refer to logical tensor axes, not backing-storage strides. Whole-storage views, positive-stride slices, reversals, and transposes retain their verified `StorageLayout`; the reduction domain is composed with that layout when reading source elements. No view is materialized merely because it participates in a reduction.

Generated C emits explicit output-domain loops followed by reducer loops. The legacy single-axis source shape remains unchanged, using one `r` induction variable. Multi-axis domains use deterministic `r0`, `r1`, ... loops in canonical-axis order. Signed or permuted storage layouts are translated through the same verified stride offsets used by other general kernels.

## Parallel execution

OpenMP never parallelizes one reduction accumulator in this phase. For a non-scalar reduction result, `parallel=True` may schedule only the independent output domain; every output coordinate still performs its complete deterministic serial left fold. Scalar reductions remain serial. Per-kernel implicit barriers retain the existing producer/consumer ordering.

This is a correctness and scheduling contract, not a performance claim. CI timing is not benchmark evidence, and this phase makes no claim about profitable thread count, grain size, or reduction speedup.

## Dynamic specialization, optimization, and reproducibility

Symbolic dimensions are fully specialized before Buffer/Loop lowering, exactly as for other operations. A reduced symbolic axis therefore becomes concrete before physical code generation, while an unreduced symbolic axis remains visible in the concrete result shape for that specialization.

Reductions remain known-pure operations for DCE and exact CSE. Canonical axis metadata is part of expression identity, so equivalent axis collections may merge while different domains or different operators remain distinct.

Canonical tuple attributes round-trip through the existing versioned tensor-IR serialization. Repro capture/replay therefore preserves the exact reduction domain and can compare reference and native execution without introducing a second reduction format.

## Deliberate boundaries

This phase expands the reduction *domain*, not the operator count. Adding `min`, `max`, arg-reductions, arbitrary reducer callbacks, dtype-changing accumulation, unstable tree reassociation, parallel accumulator trees, runtime-selected axes, or `keepdims` semantics would require separate executable milestones and their own verifier/backend contracts. More `sum`/`prod` axis spelling variants are not a roadmap goal.
