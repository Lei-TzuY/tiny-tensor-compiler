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

`compile_dynamic_module()` is the separate entrypoint for runtime-specialized tensor shapes. Tensor IR supports plain `SymbolicDim`, bounded one-variable `AffineDim` terms, and canonical positive multi-symbol `LinearDim` relations:

```python
import numpy as np

from tiny_tensor_compiler import GraphBuilder, SymbolicDim, compile_dynamic_module

B = SymbolicDim("B")
W = SymbolicDim("W")
builder = GraphBuilder()
lhs = builder.input((B + W, 1), dtype="float32")
rhs = builder.input((1, 2 * B + W), dtype="float32")
module = builder.finish((lhs + rhs).relu())

executable = compile_dynamic_module(module)
result = executable(
    inputs=[
        np.zeros((5, 1), dtype=np.float32),
        np.ones((1, 7), dtype=np.float32),
    ]
)
```

The runtime shape equations are `B+W=5` and `2*B+W=7`, so exact elimination produces `B=2,W=3`. No floating-point approximation is used: the solver operates with exact rational arithmetic and accepts a completed system only when every symbolic dimension has one unique, non-negative integer solution.

Direct and one-variable affine behavior remains compatible. A direct `B` axis binds its runtime extent immediately. An affine axis such as `2*B+1` still requires `(runtime_extent - 1)` to be non-negative and exactly divisible by `2`. Those already-known bindings are substituted into any relational equations before exact elimination. Redundant relational equations are checked again against the final binding, so contradictory constraints cannot be ignored merely because enough earlier axes already determined the symbols.

The symbolic path still does not introduce runtime-sized Buffer IR, Loop IR, C arrays, or variable-length native ABI types. It preserves one strict specialization boundary:

```text
symbolic / affine / linear tensor Module
-> tensor IR verification
-> exact runtime input rank/static-axis/dtype validation
-> collect direct bindings and multi-symbol linear equations
-> exact rational elimination for the remaining symbols
-> require a unique non-negative integer binding for every symbol
-> clone tensor IR and evaluate every symbolic expression to concrete integers
-> reverify the concrete tensor Module
-> ordinary compile_module() pipeline
-> NativeExecutable cached by the complete binding tuple
```

An inconsistent system is rejected. A rank-deficient/underdetermined system is rejected rather than choosing an arbitrary solution. A unique fractional or negative solution is also rejected because tensor extents are non-negative integers. Coefficients in `LinearDim` are positive integers and the constant offset is non-negative; subtraction, division, negative coefficients, and nonlinear symbolic products are deliberately outside this bounded solver.

Symbolic broadcasting remains conservative and structural. The same `SymbolicDim`, identical `AffineDim`, or identical `LinearDim` expressions may align with one another, and any symbolic expression may broadcast with concrete dimension `1`. Different expressions are not conditionally unified by the runtime solver during type inference. For example, `B+W` and `2*B+W` remain different dimensions even if a particular runtime binding could make their concrete values equal.

`bind_dynamic_shapes(module, inputs)` exposes the same runtime solving and returns the complete `{SymbolicDim: int}` mapping. `DynamicExecutable.symbolic_dims` reports deterministic symbol order, `cached_bindings` reports concrete binding tuples already compiled, and `specialize({...})` accepts either `SymbolicDim` or string keys. For a one-symbol executable, the existing convenience surface remains compatible: `symbolic_dim`, integer `specialize(2)`, and `cached_batch_sizes` still work. Those single-symbol convenience properties deliberately reject multi-symbol executables rather than returning ambiguous data.

`DynamicExecutable` deep-clones the symbolic module when it is created, including constant payloads. Later caller mutation of the original `Module` therefore cannot make existing and future specializations represent different programs. Each distinct complete binding lazily creates one ordinary `NativeExecutable`; repeated calls with the same binding reuse that specialization. `cache_dir=` and `borrow_inputs=True` are forwarded into every specialization, so persistent native caching, multi-output execution, and verified zero-copy inputs remain available. Zero-valued relational solutions are valid when the full-rank system uniquely determines them, such as `B+W=0` and `B+2*W=0`, which resolve to `B=0,W=0`.

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
