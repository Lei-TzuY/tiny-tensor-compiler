# tiny-tensor-compiler

A compact, correctness-first tensor compiler. The current milestone implements a real end-to-end vertical slice rather than a framework scaffold:

```text
Python tensor expressions
-> explicit typed tensor IR
-> verifier
-> constant folding / algebraic simplification / dead-code elimination
-> deterministic CPU lowering
-> NumPy-backed CPU execution
```

The NumPy CPU executor is intentionally the first backend, not the final destination. It provides a stable semantic baseline for later loop IR, explicit buffers, generated C, and native code while the compiler architecture is still small enough to verify aggressively.

## Working example

```python
from tiny_tensor_compiler import GraphBuilder, execute_cpu, lower_to_cpu, verify

builder = GraphBuilder()
x = builder.tensor([1, 2, 3])
z = (x * 2 + 1).relu()
module = builder.finish(z)

verify(module)
print(module.dump())
print(execute_cpu(lower_to_cpu(module)))
```

The IR is explicit and deterministic:

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

## Implemented now

- SSA-like `Value` objects with producer metadata and explicit use-def edges
- strongly typed `TensorType(shape, dtype)` values (`i32`, `i64`, `f32`, `f64`)
- `const`, `add`, `mul`, `relu`, and `return` operations
- NumPy-compatible shape broadcasting and explicit tensor dtype promotion
- verifier checks for operation arity, dominance/order, inferred result types, return structure, opcode legality, constant types, and use-def consistency
- constant-folding optimization with post-pass verification
- conservative algebraic simplification for exact integer `x + 0`, `0 + x`, `x * 1`, and `1 * x` identities when replacement type/shape exactly matches the result
- dead-code elimination for unused known-pure operations, including cascading producer cleanup and simplification residue
- deterministic lowering to buffer-numbered CPU instructions
- direct IR reference execution and separately lowered CPU execution
- malformed-IR tests, broadcasting tests, deterministic dump tests, randomized NumPy differential tests, linting, and CI

Python scalar literals are coerced to the peer tensor's dtype (`float32_tensor * 2` remains `f32`). Tensor-vs-tensor operations use explicit `numpy.result_type` promotion.

Algebraic simplification is intentionally conservative: floating-point neutral-element rewrites are not enabled yet because preserving strict IEEE behavior, including signed zero and NaN edge cases, takes priority over reducing operation count.

Dead-code elimination is side-effect conservative. It only removes currently known pure operations (`const`, `add`, `mul`, `relu`) whose results have no uses; terminators such as `return` are never candidates.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python examples/basic.py
```

## Near-term compiler roadmap

The next improvements should stay independently testable: common-subexpression elimination, canonicalization, then a lower-level loop/buffer IR with basic memory planning. Operator fusion should come only after those invariants are stable. CUDA is deliberately out of scope until the CPU path is compiler-like and well tested.
