# Proven disjoint same-root copy canonicalization

`Tensor.copy_into(target, source)` now has one bounded high-level exception to the historical different-root source rule: when `target` and `source` resolve to the same internally owned storage root and their concrete root-relative bounding spans are statically proven disjoint, the frontend materializes an explicit source snapshot before emitting the writable effect.

This is a canonicalization rule, not a relaxation of the low-level write invariant.

## Canonical form

For a proven same-root source, the builder rewrites the conceptual operation

```text
copy_into(root, target_same_root, source_same_root)
```

into the equivalent explicit tensor IR

```text
snapshot = reshape(source_same_root, source_same_root.shape)  # owning C-order copy
updated = copy_into(root, target_same_root, snapshot)         # different-root source
```

The existing `reshape` operation already has verified copy semantics across reference execution, Buffer/Loop lowering, generated C, native GCC/MSVC execution, borrowed inputs, and OpenMP scheduling. The snapshot therefore becomes a normal owning tensor with a storage root different from the destination root.

Tensor verification, Buffer verification, and Loop verification continue to require every actual `copy_into` source operand to use a different storage root. Malformed low-level IR cannot bypass the snapshot requirement.

## Conservative region proof

The frontend derives each concrete alias chain as a root-relative `StorageLayout(offset, strides)` and computes its minimum and maximum reachable element offsets. A same-root pair is proven disjoint only when these closed bounding intervals do not overlap.

This proof is deliberately sufficient but not necessary. For example, two adjacent row slices of a contiguous `(2, 4)` root occupy spans `[0, 3]` and `[4, 7]` and are accepted. Reversing the first row changes its logical stride to `-1` but keeps the same reachable span, so it is also accepted and the explicit snapshot preserves the reversed logical C-order sequence before the write.

Interleaved regions such as even and odd columns may touch no identical storage element while still having overlapping bounding intervals. They remain rejected in this phase rather than requiring an exact strided-set intersection solver.

Any region with unresolved symbolic extents also remains unproven before runtime specialization and is rejected by this high-level rule. The compiler does not speculate about runtime disjointness.

## Generation and execution semantics

The snapshot is evaluated before the write and therefore reads the old fresh storage generation. `copy_into` then advances the destination root generation exactly as before. All pre-write root/view handles become stale; only the returned full-root handle represents the new generation.

No backend receives a same-root source/destination copy. CPU still executes verified low-level writes with `numpy.copyto(target, source)`, generated C still emits the existing deterministic logical element copy, and OpenMP ordering remains unchanged because the snapshot kernel completes before the serial write effect.

This phase does not claim reduced memory use or faster execution. A temporary snapshot allocation is intentional correctness evidence.

## Deliberate boundary

This phase does not add:

- overlapping same-root or `memmove`-style copy semantics;
- direct same-root `copy_into` in tensor, Buffer, or Loop IR;
- exact intersection solving for interleaved signed/strided regions;
- runtime/symbolic region-disjointness proofs;
- in-place elementwise kernels or general destination-bearing operators;
- optimizer motion across writable effects;
- a performance or peak-memory improvement claim.

The next storage milestone should only broaden this surface if it introduces a stronger verifier-backed dependence capability, such as an exact bounded affine-stride region solver or explicit overlap-safe snapshot semantics. Adding more examples that already fit the same bounding-span proof is not a new phase.
