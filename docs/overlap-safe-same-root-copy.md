# Overlap-safe same-root copy canonicalization

`Tensor.copy_into(target, source)` defines high-level same-root source semantics through an explicit snapshot-before-write rule. When `source` and `target` resolve to the same internally owned storage root, the frontend first materializes the complete logical source value into distinct owning storage and only then emits the writable effect.

This is a canonicalization rule, not a relaxation of the low-level write invariant.

## Canonical form

A conceptual same-root request

```text
copy_into(root, target_same_root, source_same_root)
```

is rewritten to explicit tensor IR equivalent to

```text
snapshot = reshape(source_same_root, source_same_root.shape)  # owning C-order copy
updated = copy_into(root, target_same_root, snapshot)         # different-root source
```

The existing `reshape` operation has verified copy semantics across reference execution, Buffer/Loop lowering, generated C, native GCC/MSVC execution, borrowed inputs, dynamic specialization, and OpenMP scheduling. The snapshot is therefore a normal owning tensor whose storage root differs from the destination root.

Tensor verification, Buffer verification, and Loop verification continue to require every actual `copy_into` source operand to use a different storage root. Hand-built malformed low-level same-root writes remain rejected.

## Snapshot semantics

The snapshot is evaluated before the write and captures the source's logical C-order element sequence from the current fresh storage generation. This makes the high-level result independent of physical source/target overlap and copy traversal order.

The rule covers:

- disjoint same-root regions;
- shifted partially overlapping regions;
- interleaved strided regions;
- negative-stride reversed sources;
- transposed/permuted sources;
- runtime-symbolic unsliced axes that are concretized by the existing specialization boundary.

For example, copying `root[:, 0:3]` into `root[:, 1:4]` first snapshots columns `0:3`. The later destination write therefore cannot observe values already modified earlier in the same write. Likewise, copying odd columns into even columns snapshots the odd-column logical sequence before any even-column location changes.

This is deliberately stronger and simpler than the previous bounding-span disjointness gate. No exact strided-set intersection solver or runtime disjointness proof is required because high-level same-root writes never reach the backend as same-root copies.

## Generation and execution semantics

The snapshot reads the current fresh root generation. `copy_into` then advances that destination root generation exactly as before. All pre-write root/view handles become stale; only the returned full-root handle represents the new generation.

The reference runtime and Loop CPU backend still execute only verified different-root `copy_into` effects. Generated C reuses the same deterministic logical copy emitter in both serial and parallel modes. With `parallel=True`, a non-scalar, non-empty copy whose target layout is conservatively proven injective schedules only its outer target loop with `#pragma omp parallel for schedule(static)`. The implicit barrier publishes the completed destination generation before later operations. Scalar, zero-extent, or potentially self-overlapping target layouts retain the historical serial write loop.

For a high-level same-root source, the ordinary snapshot kernel completes before the write effect begins. If the later copy is eligible for OpenMP, the snapshot's own completion barrier/order still precedes that effect, so snapshot-before-write semantics are unchanged.

Borrowed runtime inputs remain read-only. Same-root mutation is still limited to compiler-owned internal storage roots; the snapshot rule does not make caller-owned input roots writable.

## Correctness boundary

This phase intentionally does not add:

- direct same-root `copy_into` in canonical tensor IR, Buffer IR, or Loop IR;
- backend `memmove` or traversal-direction-dependent semantics;
- zero-copy overlap handling or a peak-memory reduction claim;
- writable caller-owned input roots;
- destination casts or promotion during `copy_into`;
- asynchronous effects, effect reordering, or graph-level task scheduling;
- general in-place elementwise kernels or arbitrary destination-bearing operators;
- effect-aware optimizer motion across writes;
- caller-visible mutable native views.

Parallel copy scheduling does not make a self-overlapping target safe: if injectivity cannot be proven, codegen deliberately falls back to the existing serial copy. The explicit same-root snapshot allocation remains a correctness mechanism. No wall-clock speedup or memory-usage improvement is claimed.

## Phase promotion

The same-root copy dependence phase remains closed at the public API boundary, and the verified copy effect now also has a bounded parallel execution policy. Further copy-region examples, OpenMP chunk-size variants, or thread-count knobs are not a new milestone. The next storage/mutation frontier should add new semantics such as an explicit destination-casting policy or broader dependence-aware scheduling across ordered effects, or the project should promote to another independent subsystem.
