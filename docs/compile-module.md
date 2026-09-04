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
-> optional verified external-input borrowing
-> eager native compilation/loading
-> reusable NativeExecutable
```

It deliberately does **not** run tensor optimization passes such as constant folding, algebraic simplification, dead-code elimination, canonicalization, or common-subexpression elimination. Call those passes explicitly before `compile_module()` when they are desired. Compilation does not mutate the input `Module`.

`compiler=` and `cache_dir=` are forwarded to the existing native compilation layer. Runtime inputs therefore retain the same exact count, static shape, and dtype requirements as `compile_native()`, and the returned `NativeExecutable` retains the same cache and resource-lifecycle guarantees.

By default, external inputs retain the historical compatibility path: runtime arrays may be normalized to contiguous storage and are copied into planned physical buffers before kernels execute. Passing `borrow_inputs=True` selects the verified zero-copy path:

```python
executable = compile_module(module, borrow_inputs=True)
values = np.array([-2.0, 0.0, 3.0], dtype=np.float32)
result = executable(inputs=[values])
```

Borrowed inputs must already be `numpy.ndarray` objects with the exact compiled shape and dtype, C-contiguous layout, and aligned storage. Python sequences, non-contiguous views, or misaligned arrays are rejected rather than silently materialized. The borrowing transform separates each external input's read-only lifetime from later scratch-buffer reuse and constructs a new verified loop program. Generated C then aliases the borrowed physical slot directly to the ABI input pointer and omits that input's materialization loop; the loop interpreter binds the caller array to the same slot directly.

Native execution can optionally write directly into caller-provided NumPy output arrays. A single-output executable accepts one array; a multi-output executable accepts an ordered sequence matching its returned tensors:

```python
out = np.empty((3,), dtype=np.float32)
result = executable(
    inputs=[np.array([-2.0, 0.0, 3.0], dtype=np.float32)],
    out=out,
)
assert result is out
```

A provided output must be writable, aligned, C-contiguous, and have the exact compiled result shape and dtype. Every output must remain disjoint from all runtime inputs and from every other output. Scalar and zero-extent outputs are supported under the same exact contract. Omitting `out` allocates fresh result arrays. Invalid outputs are rejected before native execution.

Floating-point ReLU source explicitly canonicalizes both `+0.0` and `-0.0` to positive zero while preserving NaNs. The generated C uses `fabsf`/`fabs` on the zero branch rather than relying on compiler-dependent signed-zero behavior, so the same semantics hold on GCC/Clang and MSVC native paths.

Dynamic shapes remain outside this API's current scope. Multiple outputs and verified zero-copy external inputs are supported without changing the default copied-input compatibility behavior.
