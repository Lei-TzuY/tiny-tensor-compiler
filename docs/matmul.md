# Matrix multiplication

`Tensor.__matmul__` (`lhs @ rhs`) and `Tensor.matmul(rhs)` expose a bounded, verifier-backed rank-2 matrix-multiplication surface.

For `lhs: (M, K)` and `rhs: (K, N)`, both operands must be rank 2 and the contraction dimension must be structurally identical. The result has shape `(M, N)`.

## Verified compositional lowering

The first phase deliberately defines matrix multiplication using existing verified tensor operations rather than introducing a second contraction backend:

1. copy-reshape `lhs` to `(M, K, 1)`;
2. copy-reshape `rhs` to `(1, K, N)`;
3. broadcast multiply to `(M, K, N)`;
4. run the existing deterministic `sum(axis=1)` reduction to `(M, N)`.

The expansion is ordinary tensor IR. Existing verification, Buffer/Loop lowering, storage-generation checks, borrowed-input handling, native compilation, serialization, and OpenMP execution therefore remain the sole implementation paths.

Because the two reshape operations are explicit C-order copies, non-contiguous logical inputs such as transposed or reversed views are accepted without inventing a hidden layout rule: their logical element sequence is materialized before broadcast multiplication.

## Numeric semantics

Dtype promotion is inherited from the existing binary `mul` operation and follows `numpy.result_type`.

Integer multiplication and accumulation inherit the compiler's fixed-width semantics. Each product is computed at the promoted fixed width and the deterministic reduction applies its existing fixed-width accumulation boundaries in increasing contraction-index order.

Floating-point traversal is likewise the existing deterministic reduction order; this API does not delegate semantics to a platform BLAS implementation.

When `K == 0`, the broadcast product has an empty contraction axis and the existing sum identity produces zeros of the result dtype.

## Dynamic shapes and execution

Named symbolic dimensions work through the normal specialization boundary. A module such as `(B, K) @ (K, N)` binds `B`, `K`, and `N` from runtime inputs, builds one fully concrete specialization, reverifies it, and then enters the unchanged physical compiler.

`borrow_inputs=True` remains valid because the matmul expansion performs ordinary verified reads and explicit reshape copies. `parallel=True` uses the existing barriered OpenMP implementation for eligible expansion kernels; no separate matmul scheduler is introduced.

## Cost and evidence boundary

This phase materializes an `(M, K, N)` product tensor. It establishes executable matrix-multiplication semantics and architecture reuse, not GEMM efficiency or a wall-clock speedup claim. CI duration is not benchmark evidence.

No claim is made that this lowering is memory- or cache-efficient. The explicit intermediate is intentional in the first milestone because every physical step is already verifier-backed and cross-platform executable.

## Deliberate non-goals

This phase does not add:

- batched matrix multiplication;
- implicit vector dot products;
- transpose flags;
- a first-class contraction Buffer/Loop IR node;
- tiled or blocked lowering;
- BLAS dispatch;
- new SIMD/ISA-specific matmul code;
- a matmul-specific parallel scheduler.

The next matrix-multiplication milestone should only proceed if it removes the `M * K * N` materialization through direct verifier-backed contraction lowering, or otherwise adds a genuinely new executable backend capability. Adding syntax aliases or more rank-shape variants alone is not a phase promotion.

## Direct physical lowering milestone

After the compositional rank-2 semantic surface was established, the physical lowering
pipeline learned one conservative contraction optimization. The exact private
`reshape -> reshape -> mul -> sum(axis=1)` shape emitted by `Tensor.matmul()` is recognized
only when both reshape results and the product are single-use intermediates. Buffer/Loop IR
then contains one `matmul` kernel over the original `(M,K)` and `(K,N)` logical values, so
compiler-owned `(M,K,N)` product storage is not materialized.

The tensor IR deliberately remains compositional and serializable; reference execution is
therefore an independent oracle for the direct physical kernel. The direct kernel preserves
left-to-right `k=0..K-1` accumulation, casts each product and accumulator update through the
promoted output dtype, and returns additive identity for `K=0`. Generated C uses the same
ordered contraction. OpenMP may schedule independent output rows, but the `K` reduction is
never parallelized or reassociated. Logical transpose/reverse/slice layouts are indexed
through their verified strides without forcing a copy.

This is a storage-elimination and executable lowering claim, not a GEMM performance claim.
BLAS dispatch, tiling, vector-dot SIMD, batched matmul, transpose flags, and parallel K
reductions remain separate future work.
