# Deterministic reductions

This document defines the correctness boundary for the compiler's current reduction subsystem. `sum` and `prod` are deterministic same-dtype monoid reductions; `argmax` is the first index-producing selection reduction. All three share the same typed tensor IR, `ReductionPlan` domain model, logical-layout mapping, Buffer/Loop lowering, reference/CPU execution, generated C, native execution, OpenMP output scheduling, optimization, serialization, and repro pipeline.

## Reduction domains

A reduction domain is selected at graph construction time with `axis=`:

- `axis=None` reduces the complete logical tensor in C-order.
- `sum`/`prod` accept `axis=<int>` or a non-empty iterable of integer axes.
- `argmax` deliberately accepts only `axis=<int>` or `None` in this phase.

Axes are canonicalized before they enter tensor IR. Negative axes are normalized against input rank. For multi-axis `sum`/`prod`, duplicates are rejected after normalization and axes are sorted into strictly increasing order; a singleton collection collapses to the historical integer form. Booleans, strings, duplicates, out-of-range axes, and unsupported argmax axis collections are rejected instead of being silently reinterpreted.

For an input shape `(d0, ..., dn)`, a non-full reduction removes every reduced axis while preserving the relative order of unreduced axes. `sum` and `prod` preserve the input dtype. `argmax` always produces `i64` indices: `axis=None` returns the flattened logical C-order index, while `axis=k` returns the coordinate along logical axis `k` for each output position.

## Dimension retention (`keepdims`)

`Tensor.sum`, `Tensor.prod`, and `Tensor.argmax` support `keepdims=True`. Every reduced logical axis is reinserted as extent `1`; a full reduction of a rank-`N` tensor therefore exposes `(1,) * N`, while a rank-zero full reduction remains scalar. `keepdims` must be an actual Python `bool`.

Dimension retention is deliberately implemented as a canonical composition rather than a second reduction backend:

1. build the ordinary deterministic reduction with its drop-axis result type and canonical `axis` metadata;
2. expose that result through the existing verified whole-storage zero-copy `view`, reinserting extent-1 axes at the reduced logical positions.

This keeps accumulator/selection order, generated reduction code, OpenMP scheduling, serialization, and alias lifetime rules unchanged. `keepdims=False` is the default and preserves the ordinary reduction IR exactly.

## Deterministic monoid fold order

`sum` and `prod` are explicit same-dtype left folds. `sum` starts from typed zero and `prod` from typed one. Fixed-width integer arithmetic therefore wraps at every combine boundary; floating-point execution preserves one fixed operation order rather than delegating to a backend-specific tree reduction.

Full-tensor reductions traverse the complete logical tensor in C-order. Single- and multi-axis reductions iterate output coordinates in C-order, then iterate the canonical reduction-domain shape lexicographically. For canonical axes `(a0, a1, ...)`, the coordinate for `a0` is the outermost reduction loop and the last reduced axis varies fastest. Reference execution and the explicit Loop CPU backend use the same `ReductionPlan` index mapping.

Empty `sum`/`prod` reduction extents execute zero combines and return the typed operator identity for each output coordinate.

## Deterministic argmax selection

`argmax` is not modeled as a fake monoid. It has no identity and no binary `+/*` combiner. Selection scans the logical reduction domain in the same deterministic order used by the layout-aware reduction machinery and keeps an `i64` logical index.

The selection policy is explicit and shared by reference execution, the Loop CPU backend, and generated C:

- replacement uses a strict greater-than comparison, so equal maxima keep the first logical index;
- for floating-point inputs, the first NaN wins, matching NumPy `argmax` scan behavior;
- once a NaN has been selected, later values and later NaNs do not replace it;
- `axis=None` reports the flattened logical C-order position, independent of backing-storage strides;
- `axis=k` reports the coordinate within that single logical axis.

Because argmax has no identity, the selected reduction extent must be non-empty. A statically known empty domain is rejected during type inference. If a symbolic extent specializes to zero, concrete specialization is reverified and rejected before Buffer/Loop/native lowering. A zero extent on an unreduced axis remains legal; it simply yields an empty output domain with no accumulator invocation.

## Logical views and physical layout

Reduction axes refer to logical tensor axes, not backing-storage strides. Whole-storage views, positive-stride slices, reversals, and transposes retain their verified `StorageLayout`; the reduction domain is composed with that layout when reading source elements. No source view is materialized merely because it participates in a reduction.

Generated C emits explicit output-domain loops followed by reducer/selector loops. The single-axis source shape uses one `r` induction variable. Multi-axis `sum`/`prod` domains use deterministic `r0`, `r1`, ... loops in canonical-axis order. Full argmax uses one logical linear scan through the existing layout-aware linear source mapping; axis argmax uses the same logical-axis source mapping as single-axis reductions. Signed or permuted storage layouts therefore retain the same semantics across reference, Loop, and native execution.

## Parallel execution

OpenMP never parallelizes one reduction accumulator or one argmax scan in this phase. For a non-scalar reduction result, `parallel=True` may schedule only the independent output domain; each output coordinate performs its complete deterministic serial fold/selection. Scalar/full reductions remain serial. Per-kernel implicit barriers retain producer/consumer ordering.

This is a correctness and scheduling contract, not a performance claim. CI timing is not benchmark evidence, and this phase makes no claim about profitable thread count, grain size, reduction speedup, or parallel selection speedup.

## Dynamic specialization, optimization, and reproducibility

Symbolic dimensions are fully specialized before Buffer/Loop lowering. A reduced symbolic axis therefore becomes concrete before physical code generation; argmax's non-empty-domain rule is rechecked on that concrete specialization.

Reductions remain known-pure operations for DCE and exact CSE. Canonical axis metadata and the operator are part of expression identity, so equivalent normalized argmax axes can merge while different domains or operators remain distinct. Keepdims uses an existing pure alias view and requires no reduction-specific optimizer path.

Canonical attributes and reduction/view operations round-trip through the existing versioned tensor-IR serialization. Repro capture/replay therefore preserves argmax tie/NaN/index semantics and retained-rank composition across reference and native backends.

## Deliberate boundaries

The current subsystem supports deterministic same-dtype `sum`/`prod` over full, single-axis, or canonical multi-axis domains plus deterministic `argmax` over a full tensor or one logical axis, all with optional retained dimensions through verified view composition.

This argmax milestone intentionally does not add multi-axis argmax, argmin, min/max value reductions, arbitrary reducer callbacks, dtype-changing accumulation rules, unstable tree reassociation, parallel accumulator/selection trees, or runtime-selected axes. Mechanically adding the mirror operator or more spelling variants is not the next architectural goal. Higher-value reduction frontiers are a genuinely dtype-changing accumulation policy or a reproducible parallel reduction algorithm with an explicit deterministic partition/merge contract and evidence.
