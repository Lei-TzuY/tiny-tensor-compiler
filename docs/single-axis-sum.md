# Deterministic single-axis sum

This phase extends the existing same-dtype `Tensor.sum()` reduction with one compile-time axis while preserving the original full-tensor scalar reduction contract.

## Public and typed-IR contract

`Tensor.sum()` continues to reduce the entire logical tensor to shape `()`. `Tensor.sum(axis=k)` instead removes exactly one axis from the result shape and preserves every unreduced dimension and the input dtype.

The Python frontend accepts ordinary negative-axis spelling and normalizes it before constructing IR. Tensor IR therefore stores only a canonical non-negative `axis` attribute. Full-tensor sum keeps its historical attribute-free spelling, so existing modules and exact CSE keys remain compatible.

Boolean axes and axes outside `[-rank, rank)` are rejected. This phase does not add axis tuples or `keepdims`.

## Deterministic reduction order

For each output coordinate, execution fixes all unreduced logical coordinates and consumes the selected axis strictly in increasing logical index order. The accumulator starts at zero in the exact result dtype, and every addition is committed in that same dtype before the next element is consumed.

Consequences:

- `i32` and `i64` retain their fixed-width wrap boundaries instead of widening;
- floating-point evaluation order is explicit and stable;
- reducing a zero-extent axis produces zeros with the unreduced output shape;
- the operation is not reassociated into a tree reduction.

The reference runtime and Loop CPU backend implement this explicit left fold rather than delegating the semantic contract to `numpy.sum`.

## Views and logical indexing

Axis selection refers to the logical tensor after view composition, not the backing storage root. A transposed, sliced, or reversed tensor is therefore reduced along the axis the caller sees.

Buffer IR and Loop IR carry the canonical reduction axis as first-class kernel metadata. Axis-sum Loop kernels use the unreduced result shape as `iteration_shape` and have no broadcast `IndexMap`; the reduction domain is represented separately by the axis metadata.

Generated C maps each logical input coordinate through the verified root-relative `StorageLayout`. Signed strides and permutations therefore preserve the same logical-axis order as reference/CPU execution without materializing the view.

## Native and parallel execution

For a non-scalar axis-sum result, generated C emits loops over independent output coordinates and a serial inner reduction loop over `r = 0 .. reduced_extent-1`.

`parallel=True` may schedule the outer independent output loop with the existing OpenMP `parallel for schedule(static)` transform, but it does not parallelize, vector-tree, or reassociate the reduction itself. A rank-one input reduced on axis zero has a scalar output and remains serial.

The same semantics are covered through GCC-style and MSVC native execution and compose with verified borrowed inputs, ordered multiple outputs, signed-stride/permuted views, and runtime-symbolic specialization/cache reuse.

CI establishes executable correctness and portability only. It is not evidence of a reduction speedup.

## Optimization boundary

`sum` remains a known-pure operation for DCE. Exact CSE includes its canonical attribute payload, so two sums of the same operand on the same axis may merge while sums on different axes remain distinct. Full-tensor attribute-free sums retain their previous CSE behavior.

Reduction/elementwise fusion, constant folding of reductions, and reduction reassociation are not introduced here.

## Promotion after this phase

The sum-only implementation has now been promoted into a shared reduction model used by both `sum` and `prod`. `ReductionOperator` owns each operator's identity/combine semantics, while `ReductionPlan` represents the full-tensor or one-axis domain. Existing `reduction_axis` metadata remains as a compatibility view, but verifier, lowering, reference/CPU execution, generated C, DCE, and CSE no longer require a sum-only semantic path.

See `docs/product-reduction.md` for the second-operator executable milestone and its validation evidence.

The next reduction promotion is therefore **multi-axis reduction domains**, not more one-axis spelling or another operator micro-case. A multi-axis phase must canonicalize and verify the reduced axis set, preserve deterministic logical traversal across views, keep same-dtype fixed-width/float fold semantics, and execute through native/OpenMP/serialization/repro/dynamic-specialization paths without introducing an operator-specific lowering engine.

Still out of scope here:

- multiple-axis or axis-tuple reductions;
- `keepdims`;
- parallel/tree/SIMD reduction of one output value;
- runtime-selected axes;
- reduction-elementwise fusion or reassociation;
- performance claims.
