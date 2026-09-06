# Verified concatenate

`GraphBuilder.concatenate(tensors, axis=...)` is a first-class pure tensor operation for joining two or more exact-typed tensors along one compile-time axis.

## Typed contract

- At least two tensor operands are required.
- All operands must have the same rank and exact dtype.
- The axis accepts Python-style negative indexing at the builder boundary and is stored canonically as a non-negative integer in tensor IR.
- Every non-concatenated dimension must match structurally exactly.
- The result extent on the concatenated axis is the exact sum of the operand extents. Existing `SymbolicDim`, `AffineDim`, and `LinearDim` arithmetic is reused, so expressions such as `B + (2*B + 1)` canonicalize to `3*B + 1` without introducing another symbolic-shape representation.
- Rank-zero tensors are rejected.

These rules are rechecked by tensor-IR, Buffer-IR, and Loop-IR verification rather than trusted only at the frontend.

## Storage and layout semantics

Concatenate is an owning operation. Its output receives distinct storage and never aliases any input storage root.

Inputs may themselves be verified read-only aliases such as `view`, `slice`, `reverse`, or `transpose`. CPU execution consumes those logical NumPy layouts directly. Generated C walks each operand in logical index order using its verified root-relative signed strides, then writes the values into the appropriate contiguous output segment. This keeps concatenate correct for positive-stride, negative-stride, and permuted inputs without materializing those input aliases first.

Because concatenate changes the relationship between logical coordinates and multiple source storage roots, it is an explicit elementwise-fusion boundary in this phase.

## Lowering

The operation lowers as one variadic `BufferKernel` / `LoopKernel` carrying a canonical `concat_axis` field. It deliberately carries no broadcast `IndexMap`; concatenation is segmented data movement, not elementwise broadcasting.

Memory planning treats the result like any other owning kernel result. Every input remains live through the concatenate operation, including the root lifetimes of transitive views. Existing output/input storage-root alias verification therefore remains the safety boundary; no concatenate-specific alias bypass is introduced.

## Native execution

Generated C emits one deterministic nested logical-index copy nest per operand segment. The source offset uses the operand's verified `StorageLayout`; the destination offset uses the output's contiguous C-order layout plus the compile-time segment prefix along the concatenated axis.

`parallel=True` remains accepted by the high-level/native compilation path, but concatenate's own segment-copy loops intentionally stay serial in this first phase. The existing OpenMP rewriter schedules one verified kernel loop; a variadic concatenate owns multiple segment loops, so partially parallelizing only the first segment would create an unclear scheduling contract. Other kernels in the same program may still use their existing verified OpenMP paths.

This is an executable correctness capability, not a wall-clock speedup claim.

## Integration boundaries

- Borrowed runtime inputs remain read-only and may feed concatenate without hidden normalization copies.
- Multi-output native execution may return a concatenate result alongside later derived tensors.
- Dynamic specialization resolves all symbolic extents before Buffer/Loop/native lowering; concatenate introduces no runtime-sized physical IR.
- DCE may remove an unused concatenate because it is pure.
- CSE may merge only an exact duplicate with the same ordered operands, canonical axis, and result type. Operand order is observable and is never canonicalized as commutative.

## Deliberate non-goals

This phase does not add dtype promotion/casting inside concatenate, destination writes, mutation broadcasting, stack/split operations, asynchronous segment scheduling, or a view-based concatenate representation.

Further work should not farm axis or arity variants. The repository's active mutation/reduction-verification work owns those compiler-core surfaces; future concatenate expansion should happen only when it forms a distinct cross-layer milestone, such as an explicitly verified all-segment parallel schedule or a broader data-movement subsystem.
