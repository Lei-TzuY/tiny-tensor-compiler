# tiny-tensor-compiler

A compact, correctness-first tensor compiler. The current milestone implements a real end-to-end vertical slice rather than a framework scaffold:

```text
Python tensor expressions
-> explicit typed tensor IR
-> verifier
-> constant folding / algebraic simplification / dead-code elimination / canonicalization / CSE
-> explicit virtual-buffer CPU IR
-> NumPy-backed CPU kernels
```

The NumPy CPU executor is intentionally the first backend, not the final destination. It provides a stable semantic baseline while lowering is made progressively more compiler-like through explicit buffers, memory planning, loop IR, generated C, and eventually native code.

## Working example

```python
from tiny_tensor_compiler import GraphBuilder, execute_cpu, lower_to_cpu, verify

builder = GraphBuilder()
x = builder.tensor([1, 2, 3])
z = (x * 2 + 1).relu()
module = builder.finish(z)

verify(module)
print(module.dump())
program = lower_to_cpu(module)
print(program.dump())
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

Lowering now produces an explicit virtual-buffer IR rather than implicitly allocating arrays while kernels execute:

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
- CPU execution that allocates typed output buffers first and then writes NumPy kernel results into them
- direct tensor-IR reference execution and separately lowered CPU execution
- malformed-IR tests, broadcasting tests, deterministic dump tests, randomized NumPy differential tests, linting, and CI

Python scalar literals are coerced to the peer tensor's dtype (`float32_tensor * 2` remains `f32`). Tensor-vs-tensor operations use explicit `numpy.result_type` promotion.

Algebraic simplification is intentionally conservative: floating-point neutral-element rewrites are not enabled yet because preserving strict IEEE behavior, including signed zero and NaN edge cases, takes priority over reducing operation count.

Dead-code elimination is side-effect conservative. It only removes currently known pure operations (`const`, `add`, `mul`, `relu`) whose results have no uses; terminators such as `return` are never candidates.

Canonicalization currently reorders only integer `add` and `mul` operands. Earlier SSA definitions sort first, which makes commutatively equivalent integer expressions structurally identical for later CSE without relying on object identity, hash iteration, or floating-point algebra assumptions. Floating-point operands are deliberately left untouched.

Common-subexpression elimination is deliberately exact rather than algebraic. It only merges attribute-free `add`, `mul`, and `relu` operations with the same opcode, operand identities in the same order, and identical result types. It does not deduplicate constants or independently apply commutativity.

The current buffer IR uses one virtual buffer per tensor result on purpose. Physical buffer reuse is a separate memory-planning step so liveness and aliasing decisions can be tested independently rather than hidden inside lowering.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python examples/basic.py
```

## Near-term compiler roadmap

The next improvements should stay independently testable: basic liveness-based memory planning over virtual buffers, then explicit loop/kernel IR. Operator fusion should come only after those invariants are stable. CUDA is deliberately out of scope until the CPU path is compiler-like and well tested.
