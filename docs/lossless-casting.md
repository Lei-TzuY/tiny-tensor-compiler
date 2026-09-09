# Verified lossless casting

`tiny_tensor_compiler.casting.lossless_cast(tensor, dtype)` is an explicit conversion API for dtype changes that are provably exact for every value representable by the source dtype.

It is intentionally narrower than NumPy `astype` and does not add implicit destination promotion to writable effects.

## Supported conversions

The first bounded policy accepts:

- `i32 -> i64`
- `i32 -> f64`
- `f32 -> f64`
- same-dtype conversion, which returns the original tensor without creating an operation

These conversions are selected by full source-domain representability, not by whether one particular runtime value happens to fit.

The following remain rejected, including:

- `i64 -> f64`, because not every 64-bit integer is exactly representable by binary64
- `i32 -> f32`
- floating-point to integer conversions
- integer or floating-point narrowing

Unsupported conversions fail with `TypeInferenceError`; the compiler never rounds, saturates, truncates, or silently changes dtype through this API.

## Lowering model

The API deliberately does not introduce a second backend-specific cast opcode. A supported dtype-changing cast is canonicalized to the compiler's existing verified elementwise multiplication semantics with scalar `1` in the target dtype.

That gives one shared execution path:

```text
lossless_cast
-> typed target-dtype multiply by scalar one
-> existing tensor verifier
-> Buffer IR / memory planning
-> Loop IR / layout verification
-> CPU reference and lowered execution
-> generated C / native GCC-MSVC execution
-> existing OpenMP scheduling when the surrounding compiled program selects it
```

The cast result keeps the source logical shape and changes only dtype. Existing optimizer guards remain authoritative: a dtype-changing integer multiply-by-one cannot be removed by algebraic simplification because the source and result `TensorType` values differ, while constant casts may use the existing constant-folding semantics.

## Writable-effect composition

`copy_into` and `binary_into` continue to require exact-dtype sources at their effect boundary. There is no hidden destination conversion or promotion.

Cross-dtype writes are expressed explicitly:

```python
from tiny_tensor_compiler.casting import lossless_cast

converted = lossless_cast(source, target.type.dtype)
updated = target.copy_into(target, converted)
```

This preserves one exact-dtype mutation verifier and one conversion policy instead of duplicating casting rules across writable primitives. Source broadcasting remains the existing target-preserving broadcast contract after conversion.

## Correctness evidence

Regression coverage locks:

- all currently supported and rejected dtype pairs;
- exact conversion of `i32` extrema to `i64` and `f64`;
- `f32 -> f64` NaN/infinity behavior and positive/negative zero sign preservation;
- same-dtype identity behavior;
- resistance to incorrect integer multiply-by-one simplification when dtype changes;
- existing constant-folding behavior for constant casts;
- explicit cast followed by broadcast `copy_into` through borrowed-input and parallel native execution.

This phase is a correctness and executable-capability claim, not a performance claim.

## Deliberate boundary

General casting, lossy conversions, value-dependent narrowing, saturation, rounding-mode APIs, implicit writable-effect destination promotion, and a new dedicated backend cast opcode remain out of scope. They require separate semantics and evidence rather than widening this policy table opportunistically.
