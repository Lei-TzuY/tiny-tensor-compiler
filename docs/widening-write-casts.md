# Widening write casts

`copy_into` and partial `binary_into` keep exact-dtype writable-effect IR, but the public frontend now has an explicit destination-casting policy:

```python
next_root = root.copy_into(target, source, casting="widen")
next_root = root.add_into(target, source, casting="widen")
next_root = root.mul_into(target, source, casting="widen")
```

The default remains `casting="exact"`, preserving the historical contract.

## Supported widening set

The first bounded policy accepts only conversions that preserve every source value exactly across the complete dtype domain:

- `i32 -> i64`
- `i32 -> f64`
- `f32 -> f64`

A same-dtype source is already exact and does not need a materialization.

The policy deliberately rejects narrowing, floating-to-integer conversion, and `i64 -> f64`. Although many individual `i64` values are representable as `f64`, the complete `i64` domain is not, so it is not classified as an exact widening here.

## Lowering boundary

Widening is resolved before writable-effect IR. The frontend materializes the RHS with the compiler's existing typed binary semantics by multiplying by a scalar one in the destination dtype. For an allowed pair this produces an owning tensor with:

- the same logical shape as the source;
- the destination dtype;
- ordinary verified `mul` semantics through reference, Buffer IR, Loop IR, CPU, generated C, native, and OpenMP execution.

Multiplication by positive one is intentional rather than an addition-by-zero identity: for floating-point widening it preserves the sign of `-0.0`, which is part of the compiler's existing IEEE edge-semantics contract. Regression coverage checks signed-zero preservation in both reference and native execution.

The subsequent `copy_into` / `binary_into` operation therefore still receives an exact-dtype source. No casting field is added to `BufferCopyInto`, `BufferBinaryInto`, `LoopCopyInto`, or `LoopBinaryInto`, and no second alias/layout/storage-generation proof is introduced.

This also keeps serialization canonical: serialized IR records the explicit pure materialization followed by the existing exact writable effect rather than a backend-only implicit cast.

## Interaction with aliasing and scheduling

All existing mutation rules remain unchanged after materialization:

- mutation roots are compiler-owned internal storage;
- target handles must be fresh aliases of the supplied current root generation;
- same-root RHS values keep the existing snapshot-before-write rule;
- broadcast source maps are verified against the already-widened source;
- borrowed runtime inputs remain read-only;
- signed-stride and transposed target layouts use the existing write emitter;
- eligible `parallel=True` effects reuse the existing barriered OpenMP scheduling proof.

The widening materialization is an ordinary pure kernel and may itself use the established kernel scheduling/backend paths. The writable effect never sees mixed dtypes. It also acts as an explicit boundary for the first dependence-aware multi-effect scheduler: that scheduler groups only already-consecutive write effects and never moves an effect across the widening kernel.

## Deliberate non-goals

This phase does not add:

- arbitrary `astype` or a general cast opcode;
- narrowing or lossy conversion;
- float-to-int conversion or its rounding/overflow policy;
- `i64 -> f64` under value-dependent guards;
- implicit promotion when `casting` is omitted;
- widening for full-root `binary_inplace`;
- zero-copy cast views;
- a performance claim for the extra materialization.

Dependence-aware scheduling across consecutive ordered effects is now a separate implemented phase. Further storage/mutation promotion should add genuinely new dependence information—such as region-aware disjoint-write analysis on one storage root or safe scheduling across intervening pure operations—rather than adding more ad-hoc conversion pairs or more scheduler spellings.