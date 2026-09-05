# Deterministic full-tensor sum

This phase adds the compiler's first reduction operation: `Tensor.sum()` reduces one complete logical tensor to a rank-zero tensor with the same dtype.

## Semantic contract

The reduction is deliberately narrow and deterministic:

- the result shape is `()`;
- the result dtype is exactly the input dtype, including `i32` and `i64` rather than NumPy's default integer widening rules;
- the identity for an empty tensor is zero in that exact dtype;
- values are consumed in logical C-order;
- each addition is committed in the result dtype before the next element is consumed, preserving fixed-width integer wrap boundaries and one explicit floating-point evaluation order;
- `parallel=True` does not parallelize or reassociate the reduction.

The semantic oracle therefore uses an explicit left fold instead of delegating to `numpy.sum`.

## IR and lowering

Tensor IR represents attribute-free `sum` as one pure operand/result operation. The verifier requires the result to equal `infer_sum(input)`, which is a scalar tensor of the same numeric dtype.

Buffer IR lowers the operation to one ordinary owning output buffer. The output does not alias the input, including when the input is itself a read-only view.

Loop IR represents full-tensor `sum` as a scalar-output reduction kernel:

- `iteration_shape == ()` describes the output domain;
- there is exactly one input;
- there are no broadcast `IndexMap` values;
- the input's logical `TensorType` and `StorageLayout` remain independently available to execution/code generation.

This is intentionally different from elementwise kernels: a scalar output index cannot describe the input iteration domain.

## Logical views

Reduction follows the logical tensor order, not backing-root physical order. Whole-storage views, positive-stride slices, reversed axes, and transposes therefore reduce the same sequence that a caller observes through the logical tensor.

The Python Loop executor walks the logical NumPy view with `np.ndindex`. Generated C linearizes the logical input index and maps it through the verified root-relative layout strides before each load. No view materialization is required solely for reduction.

## Native and parallel execution

Generated C emits one sequential accumulator loop. It does not emit the ordinary independent-loop vectorization hint, SSE2 reduction code, or OpenMP `parallel for` for the scalar full reduction. This keeps integer wrap points and floating-point operation order identical across serial and `parallel=True` compilation.

The same reduction executes through GCC-style and MSVC native paths and composes with:

- verified borrowed external inputs;
- read-only signed-stride/permuted views;
- ordered multiple outputs;
- runtime-symbolic specialization and specialization caching.

## Optimization boundary

`sum` is known pure, so an unused reduction may be removed by DCE and exact duplicate attribute-free sums may be merged by CSE. Constant folding and reduction/elementwise fusion are not added here.

## Phase boundary

The original full-tensor phase did not add axis selection, axis tuples, `keepdims`, `mean`, `max`, parallel/tree reductions, vector reduction intrinsics, or performance claims. CI proves executable correctness and portability, not reduction speed.

The immediately promoted milestone, one compile-time axis whose output preserves the unreduced axes, is now documented in [`single-axis-sum.md`](single-axis-sum.md). The attribute-free full-tensor spelling and deterministic logical-C-order semantics described here remain unchanged.