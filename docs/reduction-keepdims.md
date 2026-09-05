# Reduction dimension retention milestone

This milestone adds `keepdims=True` to deterministic `Tensor.sum()` and `Tensor.prod()` without creating a second reduction backend or changing accumulator order.

## Canonical lowering

Dimension retention is represented compositionally:

```text
ordinary deterministic reduction
-> verified whole-storage view that reinserts extent-1 reduced axes
```

The reduction itself keeps the established canonical axis metadata and drop-axis result type. The following view exposes the retained-rank logical type over the same storage. `keepdims=False` emits exactly the historical reduction IR with no extra attribute or alias operation.

For example, reducing a `(B, 3, 4)` tensor over `(1, 2)` produces an ordinary `(B,)` reduction result. With `keepdims=True`, a verified `(B, 1, 1)` view is added over that result. A full rank-three reduction similarly exposes `(1, 1, 1)`; rank-zero input remains scalar.

## Correctness boundaries

- `keepdims` must be an actual Python `bool`.
- Reduced axes are still normalized and canonicalized before the reduction op is created.
- The reducer retains the exact deterministic same-dtype left-fold order established by the reduction subsystem.
- Empty domains still produce the operator identity for each ordinary reduction output coordinate.
- The retained-rank view is read-only and carries no new storage allocation or writable alias path.
- Downstream broadcasting consumes the retained logical shape through the existing view/layout model.
- OpenMP may schedule only the existing independent reduction output domain; no accumulator tree or reassociation is introduced.
- No performance claim is made.

## Integration proof

Regression coverage crosses canonical IR construction, default-IR compatibility, logical transpose/reverse inputs, reference and Loop execution, GCC/MSVC native execution, OpenMP, verified borrowed inputs, multi-output execution, dynamic symbolic specialization/cache reuse, DCE/CSE, tensor-IR serialization, and repro replay.

The implementation deliberately reuses the existing verified view subsystem so every lower layer receives already-supported operations rather than a new keepdims-specific execution path.

## Promotion

This closes the reduction dimension-retention phase. Further `sum`/`prod` axis spelling or keepdims syntax variants would be low-value repetition. A later reduction milestone should introduce qualitatively new semantics such as index-producing reductions, dtype-changing accumulation, or a reproducible parallel reduction algorithm with evidence; otherwise the project should promote to another compiler subsystem.
