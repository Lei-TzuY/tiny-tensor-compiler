# Deterministic reductions

This document defines the correctness boundary for the compiler's current reduction subsystem. The supported operators are `sum` and `prod`; they share one `ReductionOperator` / `ReductionPlan` model from typed tensor IR through reference execution, Buffer/Loop IR, generated C, native execution, OpenMP scheduling, serialization, and repro replay.

## Reduction domains

A reduction domain is selected at graph construction time with `axis=`:

- `axis=None` reduces the complete logical tensor in C-order and normally produces a scalar.
- `axis=<int>` reduces one logical axis and preserves the historical single-axis IR form.
- `axis=<non-empty iterable of ints>` reduces several logical axes at once.

Axis collections are canonicalized before they enter tensor IR. Negative axes are normalized against the input rank, duplicates are rejected after normalization, and the remaining axes are sorted into strictly increasing order. A singleton collection collapses to the historical integer form, so `axis=1`, `axis=(1,)`, and equivalent normalized singleton spellings have one IR/CSE identity. Empty collections, booleans, strings, duplicates, and out-of-range axes are rejected instead of being silently reinterpreted.

For an input shape `(d0, ..., dn)`, a non-full reduction removes every canonical reduced axis from the reduction result while preserving the relative order of unreduced axes.

## Accumulator and result dtype

`Tensor.sum()` and `Tensor.prod()` preserve their historical same-dtype behavior when `dtype` is omitted. An explicit same-dtype request is canonicalized back to that historical IR and does not emit redundant `dtype` metadata.

This phase also permits two deliberate same-kind widenings:

- `i32 -> i64`
- `f32 -> f64`

The selected dtype is both the accumulator dtype and the result dtype. Every logical source element is converted to that dtype before it participates in the deterministic fold, and the operator identity is created in the same selected dtype. This means `i32 -> i64` can avoid intermediate 32-bit wrap that would occur in the historical same-dtype reduction, while `f32 -> f64` performs every ordered combine in `f64` rather than widening only the final stored result.

Narrowing (`i64 -> i32`, `f64 -> f32`) and integer/float kind changes are rejected. They require a separate conversion/cast contract and are not inferred implicitly. The current dtype surface is therefore a bounded safe-widening policy, not a general cast subsystem.

A non-default reduction dtype is represented canonically by a `dtype` tensor-IR attribute such as `"i64"` or `"f64"`. That attribute is verifier checked, participates in exact CSE identity, round-trips through serialization/repro, and disappears into the concrete output `TensorType` at Buffer/Loop lowering rather than creating a second backend path.

## Dimension retention (`keepdims`)

`Tensor.sum(..., keepdims=True)` and `Tensor.prod(..., keepdims=True)` retain every reduced logical axis as extent `1`. A full reduction of a rank-`N` tensor therefore exposes `(1,) * N`; a rank-zero full reduction remains scalar. `keepdims` must be an actual Python `bool`.

Dimension retention is deliberately implemented as a canonical composition rather than a second reduction backend:

1. build the ordinary deterministic reduction with its drop-axis result shape, canonical `axis` metadata, and selected accumulator/result dtype;
2. expose that result through the existing verified whole-storage zero-copy `view`, reinserting extent-1 axes at the reduced logical positions.

This keeps accumulation order, reduction code generation, OpenMP scheduling, and reduction serialization unchanged. It also means the retained-rank result participates naturally in the existing alias-lifetime verifier and broadcasting system. For example, reducing `(B, 3, 4)` over axes `(1, 2)` with `dtype="i64"` produces the ordinary `(B,)` `i64` reduction result and a verified `(B, 1, 1)` logical view over the same storage.

`keepdims=False` is the default and preserves the historical rank behavior. Equivalent keepdims expressions may share both their canonical reduction and their identical view through existing exact CSE, while unused retained-rank results are removable by ordinary DCE.

## Deterministic fold order

Every reduction is an explicit left fold in its selected accumulator dtype. `sum` starts from typed zero and `prod` from typed one. Same-dtype fixed-width integer arithmetic therefore preserves historical wrapping at every combine boundary; a widened integer reduction wraps only at the wider selected type. Floating-point execution preserves one fixed operation order rather than delegating to a backend-specific tree reduction.

Full-tensor reductions traverse the complete logical tensor in C-order. Single- and multi-axis reductions iterate output coordinates in C-order, then iterate the canonical reduction-domain shape lexicographically. For canonical axes `(a0, a1, ...)`, the coordinate for `a0` is the outermost reduction loop and the last reduced axis varies fastest. Reference execution and the explicit Loop CPU backend use the same `ReductionPlan` index mapping.

Empty reduction extents execute zero combines and return the operator identity in the selected result dtype for each output coordinate. This is deliberate and shared by reference, Loop, generated-C, and native execution. A retained dimension does not change this rule; it only exposes the already-computed output through extent-1 logical axes.

## Logical views and physical layout

Reduction axes refer to logical tensor axes, not backing-storage strides. Whole-storage views, positive-stride slices, reversals, and transposes retain their verified `StorageLayout`; the reduction domain is composed with that layout when reading source elements. No source view is materialized merely because it participates in a reduction.

Generated C emits explicit output-domain loops followed by reducer loops. The accumulator variable uses the selected result C type, and each loaded source element is explicitly converted to that type before combination. The legacy single-axis source shape remains unchanged, using one `r` induction variable. Multi-axis domains use deterministic `r0`, `r1`, ... loops in canonical-axis order. Signed or permuted storage layouts are translated through the same verified stride offsets used by other general kernels. A keepdims result does not add another reducer loop form; the existing result storage is exposed with a verified logical view and gathered through the established contiguous output ABI when returned.

## Parallel execution

OpenMP never parallelizes one reduction accumulator in this phase. For a non-scalar reduction result, `parallel=True` may schedule only the independent output domain; every output coordinate still performs its complete deterministic serial left fold. Scalar reductions remain serial. Per-kernel implicit barriers retain the existing producer/consumer ordering.

Dimension retention and accumulator widening do not change scheduling profitability or fold order. They change only the logical result shape policy and/or the verified accumulator/result type.

This is a correctness and scheduling contract, not a performance claim. CI timing is not benchmark evidence, and this phase makes no claim about profitable thread count, grain size, or reduction speedup.

## Dynamic specialization, optimization, and reproducibility

Symbolic dimensions are fully specialized before Buffer/Loop lowering, exactly as for other operations. A reduced symbolic axis therefore becomes concrete before physical code generation. Under keepdims it is represented by extent `1`; an unreduced symbolic axis remains visible in the concrete retained-rank result shape for that specialization. The selected reduction dtype is independent of the concrete shape binding and is preserved across every specialization.

Reductions remain known-pure operations for DCE and exact CSE. Canonical axis and non-default dtype metadata are both part of expression identity: equivalent widened reductions may merge, while reductions that differ by axis, operator, or result dtype remain distinct. The keepdims view is itself an existing pure alias operation, so no reduction-specific optimizer path is required.

Canonical tuple attributes, the optional canonical dtype attribute, and the explicit view operation round-trip through the existing versioned tensor-IR serialization. Repro capture/replay therefore preserves both retained-rank and widened-accumulator semantics without introducing a second reduction encoding.

## Deliberate boundaries

The current phase supports deterministic `sum`/`prod` over full, single-axis, or canonical multi-axis domains, optional retained dimensions through verified view composition, and the bounded safe widenings `i32 -> i64` and `f32 -> f64`.

Adding `min`, `max`, arg-reductions, arbitrary reducer callbacks, narrowing, integer/float kind conversion, a generic cast operation, unstable tree reassociation, parallel accumulator trees, or runtime-selected axes requires separate executable milestones and verifier/backend contracts. More `sum`/`prod` axis spelling variants, additional keepdims syntax, or additional same-pattern widening pairs are not roadmap goals.