# Reduction dimension retention milestone

This milestone defines `keepdims=True` as a composition shared by deterministic `Tensor.sum()`, `Tensor.prod()`, and `Tensor.argmax()` rather than a second reduction backend.

## Canonical lowering

Dimension retention is represented compositionally:

```text
ordinary deterministic reduction or selection
-> verified whole-storage view that reinserts extent-1 reduced axes
```

The reduction itself keeps the established canonical axis metadata and drop-axis result type. The following view exposes the retained-rank logical type over the same storage. `keepdims=False` emits exactly the ordinary reduction IR with no extra attribute or alias operation.

For example, reducing a `(B, 3, 4)` tensor over `(1, 2)` with `sum`/`prod` produces an ordinary `(B,)` result; `keepdims=True` exposes `(B, 1, 1)`. `argmax(axis=1)` on the same tensor produces an ordinary `(B, 4)` `i64` index tensor and `keepdims=True` exposes `(B, 1, 4)`. A full rank-three reduction exposes `(1, 1, 1)`; rank-zero input remains scalar.

## Correctness boundaries

- `keepdims` must be an actual Python `bool`.
- Reduced axes are normalized before the reduction op is created; `sum`/`prod` may use canonical multi-axis tuples, while argmax accepts only one integer axis or `None` in its current phase.
- `sum`/`prod` retain the exact deterministic same-dtype left-fold order established by the reduction subsystem.
- `argmax` retains its deterministic first-tie / first-NaN selection order and its `i64` index result.
- Empty `sum`/`prod` domains still produce the operator identity for each ordinary reduction output coordinate. Argmax has no identity, so an empty selected reduction domain is rejected before physical lowering.
- The retained-rank view is read-only and carries no new storage allocation or writable alias path.
- Downstream broadcasting consumes the retained logical shape through the existing view/layout model.
- OpenMP may schedule only the existing independent reduction output domain; no accumulator tree, selection tree, or reassociation is introduced.
- No performance claim is made.

## Integration proof

Regression coverage crosses canonical IR construction, default-IR compatibility, logical transpose/reverse inputs, reference and Loop execution, GCC/MSVC native execution, OpenMP, verified borrowed inputs, multi-output execution, dynamic symbolic specialization/cache reuse, DCE/CSE, tensor-IR serialization, and repro replay.

The implementation deliberately reuses the existing verified view subsystem so every lower layer receives already-supported operations rather than a keepdims-specific execution path. Argmax therefore did not require a second retained-dimension code generator when it became the first index-producing reduction.

## Promotion

Dimension retention is now a shared composition rather than an active feature frontier. Further axis spelling or keepdims syntax variants would be low-value repetition. After the deterministic argmax milestone, higher-value reduction work should introduce a genuinely new semantic layer such as dtype-changing accumulation or a reproducible parallel reduction algorithm with an explicit deterministic partition/merge contract; mechanically adding mirror reducers is not a phase promotion.
