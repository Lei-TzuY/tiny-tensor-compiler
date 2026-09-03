# tiny-tensor-compiler

A compact, correctness-first tensor compiler. The current milestone implements a real end-to-end vertical slice rather than a framework scaffold:

```text
Python tensor expressions + static typed external inputs
-> explicit typed tensor IR
-> verifier
-> constant folding / algebraic simplification / dead-code elimination / canonicalization / CSE
-> explicit virtual-buffer CPU IR
-> liveness-based physical memory planning
-> explicit loop/kernel IR with broadcast index maps
-> conservative verifier-backed elementwise fusion
-> deterministic generated C11 source
-> process-local / optional persistent native shared-library cache
-> reusable compiled native executable handles
-> NumPy-backed scalar CPU interpretation/reference
```

The NumPy CPU executor remains the semantic baseline while lowering is made progressively more compiler-like through explicit buffers, memory planning, loop IR, generated C, and native execution. The native path compiles verified generated C into a shared library, reuses exact generated-source/compiler matches within the current process, optionally persists content-addressed artifacts across processes through `cache_dir=`, and invokes the stable output-first ABI through `ctypes` on POSIX-like GCC/Clang toolchains and Windows MSVC.

## Working example

```python
import numpy as np

from tiny_tensor_compiler import (
    GraphBuilder,
    compile_native,
    fuse_elementwise,
    lower_to_cpu,
    lower_to_loops,
    verify,
)

builder = GraphBuilder()
x = builder.input((3,), dtype="float32")
z = (x * 2 + 1).relu()
module = builder.finish(z)

verify(module)
loops = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))
executable = compile_native(loops)
result = executable(
    inputs=[np.array([-2.0, 0.0, 3.0], dtype=np.float32)],
)
print(result)
```

`compile_native()` eagerly compiles or loads the native artifact and returns a reusable `NativeExecutable`. Repeated calls with new runtime input values reuse the same native code. The executable freezes the compiler command and cache identity chosen at compile time; if `clear_native_cache()` later releases process-owned shared libraries, the handle remains valid and safely reacquires or recompiles the same artifact on its next invocation.

External inputs are explicit in tensor IR and numbered densely by declaration order:

```text
func @main() {
  %0 = input 0 : tensor<3xf32>
  %1 = const 2.0 : tensor<f32>
  %2 = mul %0, %1 : tensor<3xf32>
  %3 = const 1.0 : tensor<f32>
  %4 = add %2, %3 : tensor<3xf32>
  %5 = relu %4 : tensor<3xf32>
  return %5
}
```

Lowering preserves inputs as explicit write sources rather than turning caller data into constants:

```text
alloc b0 : tensor<3xf32>
b0 = input 0
alloc b1 : tensor<f32>
b1 = const 2.0
alloc b2 : tensor<3xf32>
b2 = mul b0, b1
alloc b3 : tensor<f32>
b3 = const 1.0
alloc b4 : tensor<3xf32>
b4 = add b2, b3
alloc b5 : tensor<3xf32>
b5 = relu b4
return b5
```

A separate deterministic memory planner maps virtual buffers onto physical slots. Reuse is allowed only after the previous virtual buffer's last use and only when the complete `TensorType` matches. External inputs therefore participate in the same explicit lifetime model as constants and computed values; the planner does not gain a hidden zero-copy alias path.

The next lowering layer makes elementwise iteration and broadcasting explicit. For example, adding tensors with shapes `(2, 1)` and `(1, 3)` produces a `(2, 3)` loop whose reads are indexed as `lhs[i0, 0]` and `rhs[0, i1]`. Scalar broadcasts use an empty input index map, and ReLU uses identity indexing. CPU execution interprets these loop kernels one output index at a time rather than delegating broadcasting to vectorized NumPy operations.

The same verified loop IR can be emitted as deterministic C11 source. Physical buffers become fixed-width typed local arrays, loop bounds become nested `int64_t` loops, broadcast index maps become explicit row-major offset expressions, constants are embedded as typed literals, and external inputs become typed `const` pointers appended after the existing output pointer. For example, a graph returning `f32` with `f32` and `i32` inputs exposes an ABI shaped like `tiny_tensor_run(float *out, const float *input0, const int32_t *input1)`. Generated source exposes `tiny_tensor_run` through a portable export macro that expands to `__declspec(dllexport)` on Windows. `execute_native()` and `compile_native()` use `cc`-style GCC/Clang flags on POSIX-like systems and MSVC `cl /std:c11 /O2 /LD` on Windows, then invoke the resulting shared library through `ctypes`. GCC-style native compilation uses `-fwrapv` so signed integer add/multiply behavior matches NumPy's fixed-width wrapping semantics. Floating-point ReLU source explicitly preserves NaN propagation and NumPy's `-0.0 -> +0.0` behavior.

## Implemented now

- SSA-like `Value` objects with producer metadata and explicit use-def edges
- strongly typed `TensorType(shape, dtype)` values (`i32`, `i64`, `f32`, `f64`)
- static typed `input`, `const`, `add`, `mul`, `relu`, and `return` operations
- dense declaration-order external input indices with verifier enforcement
- NumPy-compatible shape broadcasting and explicit tensor dtype promotion
- verifier checks for operation arity, dominance/order, inferred result types, return structure, opcode legality, constant types, input indices, and use-def consistency
- constant-folding optimization with post-pass verification
- conservative algebraic simplification for exact integer `x + 0`, `0 + x`, `x * 1`, and `1 * x` identities when replacement type/shape exactly matches the result
- dead-code elimination for unused known-pure operations, including cascading producer cleanup and simplification residue
- deterministic integer commutative canonicalization for `add` and `mul`, ordered by current SSA definition order
- conservative common-subexpression elimination for repeated exact `add`, `mul`, and `relu` expressions
- deterministic lowering to explicit `BufferAlloc`, `BufferInput`, `BufferKernel`, and `BufferReturn` operations
- buffer-IR structural/type verification for allocation, input writes, read-before-write, kernel arity, inferred output types, and return validity
- liveness-based virtual-to-physical memory planning with exact-type reuse and deterministic lowest-slot selection
- explicit `LoopAlloc`, `LoopInput`, `LoopKernel`, and `LoopReturn` IR over planned physical buffers
- explicit broadcast index maps for elementwise `add`, `mul`, and `relu` loops
- loop-IR verification for allocation order, input writes, read-before-write, non-in-place outputs, iteration shape, index maps, kernel types, and return validity
- explicit `fuse_elementwise()` pass for safe adjacent `add`/`mul` plus ReLU, greedy safe ReLU tails, and one safe same-dtype integer `add`/`mul` -> `add`/`mul` chain with verified liveness, aliasing, shape, and broadcast-map constraints
- deterministic C11 source generation with output-first ABI plus typed `const` external-input pointers
- exact runtime-input validation for input count, static shape, and dtype with no silent casting
- contiguous row-major normalization before scalar/native execution while leaving caller arrays unmodified
- native CPU compilation/execution through `.so`, `.dylib`, or `.dll` libraries
- process-local native artifact reuse keyed by exact generated C source and compiler command, independent of runtime input values
- opt-in persistent native artifact reuse through `execute_native(..., cache_dir=...)` with compiler/target fingerprinting and atomic publication
- process-owned staging copies for persistent libraries so cached `.dll` / `.so` / `.dylib` files are not directly loaded or locked
- stale or corrupt persistent artifacts are discarded and rebuilt rather than poisoning later executions
- reusable `NativeExecutable` handles from eager `compile_native()` with frozen compiler/cache configuration and safe reuse after process-cache clears
- explicit `clear_native_cache()` resource release plus automatic process-exit cleanup without deleting user-owned persistent cache files
- GCC/Clang-compatible native compilation on POSIX-like systems and MSVC `cl` compilation/loading on Windows
- CPU execution through explicit scalar loop iteration over planned physical NumPy buffers
- direct tensor-IR reference execution and separately lowered CPU execution
- malformed-IR tests, broadcasting tests, deterministic dump tests, randomized NumPy differential tests, generated-C syntax checks, cross-platform native differential tests, fusion/overflow regressions, external-input ABI/cache regressions, persistent-cache regressions, reusable-native-executable regressions, linting, and CI

Python scalar literals are coerced to the peer tensor's dtype (`float32_tensor * 2` remains `f32`). Tensor-vs-tensor operations use explicit `numpy.result_type` promotion.

External inputs are intentionally static and strict. `GraphBuilder.input(shape, dtype)` fixes each input's shape and dtype in the compiled graph. Runtime calls must provide exactly the declared number of inputs with exactly matching shapes and dtypes; the runtime does not silently cast. NumPy views may be normalized to contiguous row-major copies before execution, and input data is copied into planned internal physical buffers, so generated kernels never mutate caller-owned input arrays. Dynamic dimensions, named inputs, zero-copy input aliasing, and multiple outputs are not part of this milestone.

Algebraic simplification is intentionally conservative: floating-point neutral-element rewrites are not enabled yet because preserving strict IEEE behavior, including signed zero and NaN edge cases, takes priority over reducing operation count.

Dead-code elimination is side-effect conservative. It only removes currently known pure operations (`const`, `add`, `mul`, `relu`) whose results have no uses; `input` remains explicit so the declared runtime signature stays stable, and terminators such as `return` are never candidates.

Canonicalization currently reorders only integer `add` and `mul` operands. Earlier SSA definitions sort first, which makes commutatively equivalent integer expressions structurally identical for later CSE without relying on object identity, hash iteration, or floating-point algebra assumptions. Floating-point operands are deliberately left untouched.

Common-subexpression elimination is deliberately exact rather than algebraic. It only merges attribute-free `add`, `mul`, and `relu` operations with the same opcode, operand identities in the same order, and identical result types. It does not deduplicate constants, inputs, or independently apply commutativity.

Virtual buffers remain single-write. Physical reuse is computed separately from virtual-buffer liveness, which keeps aliasing decisions explicit and testable. The planner does not perform in-place kernels: a physical slot is reusable only when its previous virtual buffer's last use is strictly before the new virtual allocation.

Loop IR is also deliberately conservative. Physical buffers are allocated before loop execution, external inputs explicitly write their planned slots, kernel outputs may overwrite only slots whose earlier virtual value is already dead, and a kernel output may not alias any input read by that same kernel. Broadcasting is represented by deterministic index maps rather than delegated implicitly to NumPy.

Elementwise fusion is explicit rather than automatic in `lower_to_loops()`. A binary `add` or `mul` may fuse with an immediately following ReLU, producing the existing verified `relu_add` / `relu_mul` form. That fused kernel, or a standalone ReLU, can then greedily absorb additional adjacent ReLU consumers. Every absorption still requires equal iteration shapes, identity ReLU indexing, no later live use of the producer's physical value, and a final output slot that does not alias any original producer input. Binary broadcast maps are retained exactly, so a repeated ReLU tail does not weaken broadcasting or memory-planning invariants. ReLU idempotence is relied on only under the existing runtime semantics, including NaN propagation and `-0.0 -> +0.0`. The same pass can also fuse one immediately adjacent integer `add`/`mul` producer into one `add`/`mul` consumer when both kernels iterate the same shape, the producer is consumed through identity indexing, its physical value has no later use, the final output does not alias any fused input, and the intermediate/output plus all fused inputs use one exact `i32` or `i64` dtype. The generated C preserves the original fixed-width intermediate explicitly before evaluating the outer operation. Floating-point binary chains, deeper binary chains, kernel reordering, non-adjacent fusion, and general expression DAGs remain deliberately out of scope.

Native execution separates deterministic code generation, process-local loaded-artifact reuse, optional persistent storage, and reusable executable handles. Without `cache_dir`, behavior remains process-local: an exact `(compiler command, generated C source)` match reuses the already loaded shared library, and `clear_native_cache()` or process exit releases it and removes its temporary build directory. Passing `cache_dir` enables a versioned content-addressed disk cache. Its digest includes the generated source, full compiler command, resolved compiler executable path/size/mtime, operating system, platform, machine architecture, pointer width, and library format. A library is published into that cache only after successful compilation using an atomic same-filesystem `os.replace()`, so failed compilations do not create cache hits. Persistent library bytes are copied into a process-owned temporary staging directory before `ctypes` loads them; this keeps the persistent file itself immutable and avoids Windows DLL locking. If a persisted library cannot be loaded, it is treated as stale or corrupt, removed, and rebuilt. `clear_native_cache()` unloads and removes process staging directories but deliberately leaves the user-owned persistent cache intact for later processes. `compile_native()` additionally freezes its selected compiler command and persistent artifact identity into a `NativeExecutable`; the handle never owns a raw DLL/SO lifetime independently of the cache, so clearing process-owned native resources cannot leave it with a dangling loaded-library pointer. Different graphs, compiler commands, compiler executable fingerprints, or targets therefore compile independently, while runtime input values do not affect the artifact key.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python examples/basic.py
```

A native C compiler is required to exercise `execute_native()` or `compile_native()`: a `cc`-compatible GCC/Clang toolchain on POSIX-like systems, or an MSVC developer environment exposing `cl` on Windows. CI executes the full suite on Ubuntu and Windows for Python 3.11 and 3.13.

## Near-term compiler roadmap

The runtime-input boundary, safe ReLU-tail chain fusion, first constrained two-stage integer binary fusion, optional persistent native artifact cache, and reusable compiled-executable handle are now explicit. Follow-up milestones can independently explore broader multi-operation fusion DAGs, SIMD/parallel loop scheduling, higher-level execution ergonomics, or eventually CUDA. Dynamic shapes, zero-copy external-buffer aliasing, and multiple-output ABI design remain separate correctness problems rather than implicit extensions of this execution milestone.
