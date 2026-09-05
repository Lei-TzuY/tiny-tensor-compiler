# Verified reshape copy semantics

This phase adds `Tensor.reshape(shape)` as a first-class typed tensor operation and carries it through verification, dynamic specialization, Buffer IR, Loop IR, reference execution, the explicit CPU interpreter, generated C, native execution, and opt-in OpenMP scheduling.

## Semantic contract

`reshape` preserves dtype and the row-major element sequence. It changes only the logical tensor shape.

`reshape` intentionally remains a **copy operation**. Lowering allocates a distinct output buffer and copies elements in C-order. Callers that want the separately verified zero-copy whole-storage alias contract use `Tensor.view(shape)` instead; the two operations remain distinct in tensor IR and physical lowering.

This boundary keeps the existing zero-copy input contract honest: borrowed runtime inputs may be read directly, but reshape writes a distinct internal/output buffer. No hidden input normalization copy is introduced by borrowing, and `reshape` itself never claims view semantics.

## Exact element-count proof

Static and symbolic reshapes must prove element-count equality before the operation is admitted to typed IR.

Each supported shape dimension is converted to an exact canonical polynomial:

- integer extents are constants;
- `SymbolicDim` contributes one symbol;
- `AffineDim` contributes its positive scale and non-negative offset;
- `LinearDim` contributes its canonical positive-coefficient multi-symbol relation.

The compiler multiplies the dimension polynomials for the source and target shapes and requires exact polynomial equality. This admits identities such as:

```text
(B, 4) -> (2, 2*B)
(B+1, W+1) -> (W+1, B+1)
```

while rejecting forms whose equality would only hold for some runtime bindings, such as `(B, 4) -> (B, 5)`.

A reshape target also may not introduce a symbolic dimension that is absent from the source tensor. Runtime inputs remain the source of dynamic bindings; reshape is not a mechanism for inventing new unconstrained symbols.

After dynamic bindings are solved, normal module specialization concretizes both the operand and reshape result types and reruns the verifier before physical lowering. Buffer IR, Loop IR, generated C, and the native ABI therefore remain concrete-sized exactly as before.

The same shape proof is reused by `view`; the distinction between the two operations is physical copy versus alias semantics, not a weaker shape rule.

## Physical lowering

Buffer IR represents reshape as a normal one-input pure kernel with a distinct virtual output buffer.

Loop IR gives reshape the output iteration shape but deliberately uses no broadcast `IndexMap`. Rank-changing reshape is not broadcasting. Its semantics are one flat row-major copy from the source physical buffer to the destination physical buffer.

Generated C emits the same flat `n` copy loop. The ordinary serial path keeps the existing vectorization hint. With `parallel=True`, the existing verified OpenMP scheduler may place `#pragma omp parallel for schedule(static)` on that non-scalar, non-empty copy loop; the normal per-kernel implicit barrier remains in force. Scalar and zero-extent cases retain the existing serial scheduling boundary.

The reference executor explicitly copies the NumPy C-order reshape result, and the loop CPU executor copies between flattened C-contiguous physical buffers. This preserves one observable semantic contract across all execution layers.

By contrast, `view` receives no copy kernel or new physical allocation; its separate contract is documented in `docs/views.md`.

## Fusion and optimization

Reshape is a fusion boundary. Elementwise fusion may not absorb a reshape or reinterpret its rank change as identity indexing.

Reshape is nevertheless a known pure operation for existing optimizer infrastructure:

- dead-code elimination may erase an unused reshape;
- exact common-subexpression elimination may merge two reshapes only when opcode, source value, and result type are identical.

No new algebraic reshape rewrites are introduced.

## Verification evidence

Regression coverage includes:

- static typed reshape and C-order reference behavior;
- exact static and symbolic element-count rejection;
- rejection of target-only symbolic dimensions;
- Buffer/Loop lowering with distinct storage and no broadcast index map;
- explicit fusion-boundary behavior;
- scalar and zero-extent reshapes;
- generated serial and OpenMP C copy loops;
- native reshape combined with verified borrowed inputs, ordered multi-output execution, and OpenMP;
- dynamic symbolic reshape across multiple runtime bindings with specialization-cache reuse;
- reshape DCE and exact CSE behavior;
- Ubuntu and Windows CI with Python 3.11 and 3.13.

The tests establish correctness and executable interoperability. They do not establish a performance advantage for copying reshape or for parallelizing it.

## Deliberately out of scope

`reshape` itself does not add:

- inferred `-1` dimensions;
- transpose or arbitrary stride transforms;
- implicit zero-copy behavior;
- reshape-chain folding or constant-reshape folding;
- runtime-sized Buffer/Loop IR;
- a performance/profitability policy for parallel reshape copies.

A separate `view` operation now provides verified whole-storage C-contiguous zero-copy shape aliases with alias-aware lifetimes and storage-generation checks. Non-zero offsets, arbitrary strides, transpose/permutation, slicing, partial overlapping views, and writable/in-place view kernels remain a later layout phase rather than being hidden inside reshape.
