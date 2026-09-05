# Deterministic product reduction and shared reduction plan

This phase promotes the reduction subsystem from one hard-coded `sum` operator into a reusable deterministic reduction model exercised by two real operators: `sum` and `prod`.

The implementation does not add a parallel product-only lowering path. `ReductionOperator` owns the operator-specific identity, scalar combine rule, and C operator; `ReductionPlan` pairs one operator with either the full logical tensor or one canonical compile-time axis. Tensor verification, Buffer IR, Loop IR, reference/CPU execution, generated C, DCE, and CSE all consume the same reduction model.

## Public contract

`Tensor.prod()` reduces the complete logical tensor to shape `()` and preserves the input dtype. `Tensor.prod(axis=k)` removes exactly one logical axis, preserving every unreduced dimension and the input dtype.

Negative Python axes are normalized before tensor IR is created. Tensor IR therefore stores only canonical non-negative `axis` attributes. Full-tensor product is attribute-free, matching the existing full-tensor sum representation.

Boolean axes and axes outside `[-rank, rank)` are rejected. Axis tuples, `keepdims`, runtime-selected axes, and multiple-axis domains remain out of scope in this phase.

## Deterministic semantics

For every output value, product is an explicit left fold in logical index order:

1. initialize the accumulator to one in the exact result dtype;
2. consume elements in deterministic C-order for a full-tensor product, or increasing logical index order along the selected axis;
3. after every multiply, commit the intermediate back to the exact result dtype before the next element is consumed.

Consequences:

- `i32` and `i64` preserve fixed-width multiply wrap boundaries instead of widening;
- floating-point products have explicit stable evaluation order rather than an implementation-selected reduction tree;
- a zero-element full reduction returns scalar one in the input dtype;
- reducing a zero-extent selected axis produces ones with the unreduced output shape;
- the compiler does not reassociate one product output into a SIMD/OpenMP/tree reduction.

The reference executor and Loop CPU backend implement this contract directly rather than delegating semantics to `numpy.prod`.

## Shared reduction representation

`ReductionOperator` currently contains `SUM` and `PRODUCT`. It defines the exact identity and scalar combiner for each operator. `ReductionPlan(operator, axis)` defines the deterministic reduction domain.

Buffer and Loop kernels retain the historical `reduction_axis` field as a compatibility surface for existing sum tests and callers, but expose a computed `reduction` plan whenever their opcode is a supported reduction. This keeps old full/axis-sum IR stable while removing sum-only semantic branches from verification and backend execution.

The same inference path validates both operators. The same Buffer/Loop verifier path checks one input, no broadcast map at Loop IR, output shape/dtype, canonical axis, and alias/storage safety. Product remains a fusion boundary just like sum.

## Views and physical layout

A reduction consumes the logical tensor after zero-copy view composition. Transpose, positive-stride slice, signed-stride reverse, and whole-storage view layouts therefore affect address calculation but not reduction order.

Full reductions enumerate the complete logical tensor in C-order. Single-axis reductions fix every unreduced logical coordinate and enumerate the selected logical axis from zero to extent minus one. Generated C maps those logical coordinates through the verified root-relative `StorageLayout`, including signed strides.

No view is materialized solely because it feeds a product reduction.

## Native and OpenMP execution

Generated C uses one shared reduction emitter for sum and product. Operator-specific behavior is limited to the identity literal (`0` or `1`) and scalar C operator (`+` or `*`). The established sum source spelling remains stable (`sum_value`); product uses `prod_value`.

For non-scalar single-axis outputs, `parallel=True` may schedule only the independent outer output loop with the existing `#pragma omp parallel for schedule(static)` transform. The inner fold remains serial and ordered. Full-tensor product and rank-one axis product produce scalar outputs and remain serial.

The product path is executed by both GCC-style and MSVC native CI and composes with verified borrowed inputs, ordered multiple outputs, dynamic specialization/cache reuse, and signed-stride/permuted views.

CI demonstrates executable correctness and portability only. It is not a speedup claim.

## Optimization, serialization, and repro

Product is a known-pure operation for DCE. Exact CSE includes the canonical reduction attribute payload, so duplicate products on the same operand and axis may merge while different axes remain distinct. The opcode remains part of the key, so a sum and product can never merge.

Canonical serialization and repro capture/replay require no product-specific container format: the existing opcode/attribute representation is generic, and deserialization re-enters the normal verifier. Native repro replay therefore exercises the same shared reduction lowering.

Constant folding of reductions, reduction-elementwise fusion, and reduction reassociation remain deliberately out of scope.

## Validation

The production candidate `3da81d3cf49059506594fde546854f439a34179a` passed CI #954 / run `33996456340` on Ubuntu and Windows with Python 3.11 and 3.13. Every matrix cell passed Ruff and the full pytest suite; Ubuntu Python 3.11 reported 521 passed tests.

An earlier candidate stopped at Ruff because `lowering.py` retained one unused import after the reduction refactor. The import was removed without suppressing or weakening the lint rule; no pytest result from that failed run was treated as evidence.

## Phase boundary and next promotion

This closes the second-operator reduction phase. Adding `min`, `max`, or another one-axis operator next would mostly widen the operator list and, for extrema, introduce separate empty-domain/NaN policy questions without deepening the reduction domain.

The next reduction milestone should instead generalize the domain itself: verifier-backed **multi-axis reductions** that canonicalize an axis set/tuple, preserve deterministic logical traversal, define empty-domain behavior through the existing operator identity, and execute end-to-end through views, native code, OpenMP outer-output scheduling, serialization, repro, and dynamic specialization. That phase should reuse `ReductionPlan` rather than creating another operator-specific lowering path.
