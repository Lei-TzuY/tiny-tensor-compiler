# High-level module compilation

`compile_module()` is the end-to-end native compilation entrypoint for a verified tensor `Module`.

```python
import numpy as np

from tiny_tensor_compiler import GraphBuilder, compile_module

builder = GraphBuilder()
x = builder.input((3,), dtype="float32")
module = builder.finish((x * 2 + 1).relu())

executable = compile_module(module)
result = executable(
    inputs=[np.array([-2.0, 0.0, 3.0], dtype=np.float32)],
)
```

The entrypoint runs the existing correctness-preserving lowering path:

```text
tensor Module
-> tensor IR verification
-> virtual-buffer CPU IR
-> liveness-based physical memory planning
-> explicit loop IR
-> conservative elementwise fusion
-> eager native compilation/loading
-> reusable NativeExecutable
```

It deliberately does **not** run tensor optimization passes such as constant folding, algebraic simplification, dead-code elimination, canonicalization, or common-subexpression elimination. Call those passes explicitly before `compile_module()` when they are desired. Compilation does not mutate the input `Module`.

`compiler=` and `cache_dir=` are forwarded to the existing native compilation layer. Runtime inputs therefore retain the same exact count, static shape, and dtype requirements as `compile_native()`, and the returned `NativeExecutable` retains the same cache and resource-lifecycle guarantees.

Native execution can optionally write directly into a caller-provided NumPy output array:

```python
out = np.empty((3,), dtype=np.float32)
result = executable(
    inputs=[np.array([-2.0, 0.0, 3.0], dtype=np.float32)],
    out=out,
)
assert result is out
```

A provided `out` must be a writable, aligned, C-contiguous `numpy.ndarray` with the exact compiled result shape and dtype. It must not overlap any runtime input. Scalar and zero-extent outputs are supported under the same exact contract. Omitting `out` keeps the existing behavior and allocates a fresh result array. Invalid outputs are rejected before `execute_native()` performs compiler lookup or compilation; `compile_native()` itself remains an eager compilation API.

Floating-point ReLU source explicitly canonicalizes both `+0.0` and `-0.0` to positive zero while preserving NaNs. The generated C uses `fabsf`/`fabs` on the zero branch rather than relying on compiler-dependent signed-zero behavior, so the same semantics hold on GCC/Clang and MSVC native paths.

Dynamic shapes, multiple outputs, zero-copy external input aliasing, new ABI behavior, and additional fusion forms are outside this API's scope.
