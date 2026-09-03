# tiny-tensor-compiler

A compact, correctness-first tensor compiler. The current milestone implements a real end-to-end vertical slice rather than a framework scaffold:

```text
Python tensor expressions
-> explicit typed tensor IR
-> verifier
-> constant folding / algebraic simplification / dead-code elimination / canonicalization / CSE
-> explicit virtual-buffer CPU IR
-> liveness-based physical memory planning
-> explicit loop/kernel IR with broadcast index maps
-> NumPy-backed scalar CPU execution
```

The NumPy CPU executor is intentionally the first backend, not the final destination. It provides a stable semantic baseline while lowering is made progressively more compiler-like through explicit buffers, memory planning, loop IR, generated C, and eventually native code.

## Working example

```python
from tiny_tensor_compiler import (
    GraphBuilder,
    execute_cpu,
    lower_to_cpu,
    lower_to_loops,
    plan_memory,
    verify,
)

builder = GraphBuilder()
x = builder.tensor([1, 2, 3])
z = (x * 2 + 1).relu()
module = builder.finish(z)

verify(module)
print(module.dump())
program = lower_to_cpu(module)
print(program.dump())
print(plan_memory(program).dump())
print(lower_to_loops(program).dump())
print(execute_cpu(program))
```

The tensor IR is explicit and deterministic:

```text
func @main() {
  %0 = const [1, 2, 3] : tensor<3xi64>
  %1 = const 2 : tensor<i64>
  %2 = mul %0, %1 : tensor<3xi64>
  %3 = const 1 : tensor<i64>
  %4 = add %2, %3 : tensor<3xi64>
  %5 = relu %4 : tensor<3xi64>
  return %5
}
```

Lowering produces an explicit virtual-buffer IR rather than implicitly allocating arrays while kernels execute:

```text
alloc b0 : tensor<3xi64>
b0 = const [1, 2, 3]
alloc b1 : tensor<i64>
b1 = const 2
alloc b2 : tensor<3xi64>
b2 = mul b0, b1
alloc b3 : tensor<i64>
b3 = const 1
alloc b4 : tensor<3xi64>
b4 = add b2, b3
alloc b5 : tensor<3xi64>
b5 = relu b4
return b5
```

A separate deterministic memory planner maps those virtual buffers onto physical slots. Reuse is allowed only after the previous virtual buffer's last use and only when the complete `TensorType` matches. A buffer still read by the current kernel therefore cannot alias that kernel's output.

For a same-typed linear ReLU chain, virtual buffers can alternate safely between two physical slots:

```text
b0 -> p0 : tensor<3xi32>
b1 -> p1 : tensor<3xi32>
b2 -> p0 : tensor<3xi32>
b3 -> p1 : tensor<3xi32>
```

The next lowering layer makes elementwise iteration and broadcasting explicit. For example, adding tensors with shapes `(2, 1)` and `(1, 3)` produces a `(2, 3)` loop whose reads are indexed as `lhs[i0, 0]` and `rhs[0, i1]`. Scalar broadcasts use an empty input index map, and ReLU uses identity indexing. CPU execution now interprets these loop kernels one output index at a time rather than delegating broadcasting to vectorized NumPy operations.

## Implemented now

- SSA-like `Value` objects with producer metadata and explicit use-def edges
- strongly typed `TensorType(shape, dtype)` values (`i32`, `i64`, `f32`, `f64`)
- `const`, `add`, `mul`, `relu`, and `return` operations
- NumPy-compatible shape broadcasting and explicit tensor dtype promotion
- verifier checks for operation arity, dominance/order, inferred result types, return structure, opcode legality, constant types, and use-def consistency
- constant-folding optimization with post-pass verification
- conservative algebraic simplification for exact integer `x + 0`, `0 + x`, `x * 1`, and `1 * x` identities when replacement type/shape exactly matches the result
- dead-code elimination for unused known-pure operations, including cascading producer cleanup and simplification residue
- deterministic integer commutative canonicalization for `add` and `mul`, ordered by current SSA definition order
- conservative common-subexpression elimination for repeated exact `add`, `mul`, and `relu` expressions
- deterministic lowering to explicit `BufferAlloc`, `BufferKernel`, and `BufferReturn` operations
- buffer-IR structural/type verification for allocation, read-before-write, kernel arity, inferred output types, and return validity
- liveness-based virtual-to-physical memory planning with exact-type reuse and deterministic lowest-slot selection
- explicit `LoopAlloc`, `LoopKernel`, and `LoopReturn` IR over planned physical buffers
- explicit broadcast index maps for elementwise `add`, `mul`, and `relu` loops
- loop-IR verification for allocation order, read-before-write, non-in-place outputs, iteration shape, index maps, kernel types, and return validity
- CPU execution through explicit scalar loop iteration over planned physical NumPy buffers
- direct tensor-IR reference execution and separately lowered CPU execution
- malformed-IR tests, broadcasting tests, deterministic dump tests, randomized NumPy differential tests, linting, and CI

Python scalar literals are coerced to the peer tensor's dtype (`float32_tensor * 2` remains `f32`). Tensor-vs-tensor operations use explicit `numpy.result_type` promotion.

Algebraic simplification is intentionally conservative: floating-point neutral-element rewrites are not enabled yet because preserving strict IEEE behavior, including signed zero and NaN edge cases, takes priority over reducing operation count.

Dead-code elimination is side-effect conservative. It only removes currently known pure operations (`const`, `add`, `mul`, `relu`) whose results have no uses; terminators such as `return` are never candidates.

Canonicalization currently reorders only integer `add` and `mul` operands. Earlier SSA definitions sort first, which makes commutatively equivalent integer expressions structurally identical for later CSE without relying on object identity, hash iteration, or floating-point algebra assumptions. Floating-point operands are deliberately left untouched.

Common-subexpression elimination is deliberately exact rather than algebraic. It only merges attribute-free `add`, `mul`, and `relu` operations with the same opcode, operand identities in the same order, and identical result types. It does not deduplicate constants or independently apply commutativity.

Virtual buffers remain single-write. Physical reuse is computed separately from virtual-buffer liveness, which keeps aliasing decisions explicit and testable. The planner does not perform in-place kernels: a physical slot is reusable only when its previous virtual buffer's last use is strictly before the new virtual allocation.

Loop IR is also deliberately conservative. Physical buffers are allocated before loop kernels, kernel outputs may overwrite only slots whose earlier virtual value is already dead, and a kernel output may not alias any input read by that same kernel. Broadcasting is represented by deterministic index maps rather than delegated implicitly to NumPy.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python examples/basic.py
```

## Near-term compiler roadmap

The next improvements should stay independently testable: generated C for the explicit loop IR, then native CPU compilation/execution. Operator fusion should come only after those invariants are stable. SIMD, parallel loop scheduling, and CUDA are deliberately out of scope until the scalar CPU path is compiler-like and well tested.
