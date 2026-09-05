# High-level module compilation

`compile_module()` is the end-to-end eager native compilation entrypoint for a verified concrete-shape tensor `Module`.

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
concrete tensor Module
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

`compiler=` and `cache_dir=` are forwarded to the existing native compilation layer. Runtime inputs therefore retain the same exact count, concrete shape, and dtype requirements as `compile_native()`, and the returned `NativeExecutable` retains the same cache and resource-lifecycle guarantees.

By default, external inputs retain the historical compatibility path: runtime arrays may be normalized to contiguous storage and are copied into planned physical buffers before kernels execute. Passing `borrow_inputs=True` selects the verified zero-copy path:

```python
executable = compile_module(module, borrow_inputs=True)
values = np.array([-2.0, 0.0, 3.0], dtype=np.float32)
result = executable(inputs=[values])
```

Borrowed inputs must already be `numpy.ndarray` objects with the exact compiled shape and dtype, C-contiguous layout, and aligned storage. Python sequences, non-contiguous views, or misaligned arrays are rejected rather than silently materialized. The borrowing transform separates each external input's read-only lifetime from later scratch-buffer reuse and constructs a new verified loop program. Generated C then aliases the borrowed physical slot directly to the ABI input pointer and omits that input's materialization loop; the loop interpreter binds the caller array to the same slot directly.

## Runtime symbolic-shape specialization

`compile_dynamic_module()` is the separate entrypoint for runtime-specialized tensor shapes. Tensor IR may contain one or more named `SymbolicDim` values on arbitrary axes, and may use bounded one-variable affine terms with a positive integer scale and non-negative integer offset:

```python
import numpy as np

from tiny_tensor_compiler import GraphBuilder, SymbolicDim, compile_dynamic_module

B = SymbolicDim("B")
W = SymbolicDim("W")
builder = GraphBuilder()
lhs = builder.input((2 * B + 1, 1), dtype="float32")
rhs = builder.input((1, 3 * W + 2), dtype="float32")
module = builder.finish((lhs + rhs).relu())

executable = compile_dynamic_module(module)
result = executable(
    inputs=[
        np.zeros((5, 1), dtype=np.float32),
        np.ones((1, 8), dtype=np.float32),
    ]
)
```

The runtime contract solves `2*B+1 = 5` and `3*W+2 = 8`, producing `B=2,W=2`. An affine axis is accepted only when `(runtime_extent - offset)` is non-negative and exactly divisible by the positive scale. A repeated direct or affine occurrence of the same symbol must resolve to the identical integer binding.

The symbolic path does not introduce runtime-sized Buffer IR, Loop IR, C arrays, or variable-length native ABI types. Instead it preserves a strict specialization boundary:

```text
symbolic / affine tensor Module
-> tensor IR verification
-> exact runtime input rank/static-axis/dtype validation
-> solve every named symbol from direct or affine runtime input axes
-> clone tensor IR and evaluate all symbolic/affine terms to concrete integers
-> reverify the concrete tensor Module
-> ordinary compile_module() pipeline
-> NativeExecutable cached by the complete binding tuple
```

Symbolic broadcasting remains conservative. The same `SymbolicDim` may broadcast with itself or dimension `1`, and structurally identical affine terms may align with one another. Distinct symbols are not implicitly unified, and a direct `B` aligned against `2*B` is not treated as conditionally equal for selected runtime values. This keeps type inference deterministic before specialization.

`bind_dynamic_shapes(module, inputs)` exposes the same runtime-binding validation and returns the complete `{SymbolicDim: int}` mapping. `DynamicExecutable.symbolic_dims` reports the deterministic symbol order, `cached_bindings` reports the concrete binding tuples already compiled, and `specialize({...})` accepts either `SymbolicDim` or string keys. For a one-symbol executable, the existing convenience surface remains compatible: `symbolic_dim`, integer `specialize(2)`, and `cached_batch_sizes` still work. Those single-symbol convenience properties deliberately reject multi-symbol executables rather than returning ambiguous data.

`DynamicExecutable` deep-clones the symbolic module when it is created, including constant payloads. Later caller mutation of the original `Module` therefore cannot make existing and future specializations represent different programs. Each distinct complete binding lazily creates one ordinary `NativeExecutable`; repeated calls with the same binding reuse that specialization. `cache_dir=` and `borrow_inputs=True` are forwarded into every specialization, so persistent native caching, multi-output execution, and verified zero-copy inputs remain available. Zero-valued symbolic dimensions are valid when the affine expression evaluates to a legal zero extent, such as `2*B` with `B=0`.

The current affine contract is intentionally bounded: each expression contains exactly one named symbol, a strictly positive integer scale, and a non-negative integer offset. Multi-variable expressions, subtraction, division, implicit equality solving between distinct symbols, reshape-style symbolic transforms, and runtime-sized physical buffers are not claimed.

## Outputs

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
