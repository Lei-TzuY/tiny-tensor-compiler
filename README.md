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
-> conservative verifier-backed elementwise fusion
-> deterministic generated C11 source
-> cached native CPU shared-library execution
-> NumPy-backed scalar CPU interpretation/reference
```

The NumPy CPU executor remains the semantic baseline while lowering is made progressively more compiler-like through explicit buffers, memory planning, loop IR, generated C, and native execution. The native path compiles verified generated C into a shared library, caches exact generated-source/compiler-command matches for reuse within the current process, and invokes the stable output-pointer ABI through `ctypes` on POSIX-like GCC/Clang toolchains and Windows MSVC.

## Working example

```python
from tiny_tensor_compiler import (
    GraphBuilder,
    execute_cpu,
    execute_native,
    fuse_elementwise,
    generate_c,
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
loops = lower_to_loops(program)
print(loops.dump())
fused = fuse_elementwise(loops)
print(fused.dump())
print(generate_c(fused))
print(execute_cpu(program))
print(execute_native(fused))
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

The next lowering layer makes elementwise iteration and broadcasting explicit. For example, adding tensors with shapes `(2, 1)` and `(1, 3)` produces a `(2, 3)` loop whose reads are indexed as `lhs[i0, 0]` and `rhs[0, i1]`. Scalar broadcasts use an empty input index map, and ReLU uses identity indexing. CPU execution interprets these loop kernels one output index at a time rather than delegating broadcasting to vectorized NumPy operations.

The same verified loop IR can be emitted as deterministic C11 source. Physical buffers become fixed-width typed local arrays, loop bounds become nested `int64_t` loops, broadcast index maps become explicit row-major offset expressions, constants are embedded as typed literals, and the returned physical buffer is copied into the generated function's output pointer. Generated source exposes `tiny_tensor_run` through a portable export macro that expands to `__declspec(dllexport)` on Windows. `execute_native()` uses `cc`-style GCC/Clang flags on POSIX-like systems and MSVC `cl /std:c11 /O2 /LD` on Windows, then invokes the resulting shared library through `ctypes`. GCC-style native compilation uses `-fwrapv` so signed integer add/multiply behavior matches NumPy's fixed-width wrapping semantics. Floating-point ReLU source explicitly preserves NaN propagation and NumPy's `-0.0 -> +0.0` behavior.

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
- explicit `fuse_elementwise()` pass for safe adjacent `add`/`mul` followed by ReLU, represented as verified `relu_add` / `relu_mul` loop kernels
- deterministic C11 source generation from explicit loop IR with fixed-width dtypes, nested loops, row-major indexing, scalar broadcasting, typed constants, fused kernels, and portable DLL export
- native CPU compilation/execution through `.so`, `.dylib`, or `.dll` libraries and a stable `tiny_tensor_run` output-pointer ABI
- process-local native artifact reuse keyed by exact generated C source and compiler command, with failed compilations excluded from the cache
- explicit `clear_native_cache()` resource release plus automatic process-exit cleanup
- GCC/Clang-compatible native compilation on POSIX-like systems and MSVC `cl` compilation/loading on Windows
- CPU execution through explicit scalar loop iteration over planned physical NumPy buffers
- direct tensor-IR reference execution and separately lowered CPU execution
- malformed-IR tests, broadcasting tests, deterministic dump tests, randomized NumPy differential tests, generated-C syntax checks, cross-platform native differential tests, fusion/overflow regressions, cache regressions, linting, and CI

Python scalar literals are coerced to the peer tensor's dtype (`float32_tensor * 2` remains `f32`). Tensor-vs-tensor operations use explicit `numpy.result_type` promotion.

Algebraic simplification is intentionally conservative: floating-point neutral-element rewrites are not enabled yet because preserving strict IEEE behavior, including signed zero and NaN edge cases, takes priority over reducing operation count.

Dead-code elimination is side-effect conservative. It only removes currently known pure operations (`const`, `add`, `mul`, `relu`) whose results have no uses; terminators such as `return` are never candidates.

Canonicalization currently reorders only integer `add` and `mul` operands. Earlier SSA definitions sort first, which makes commutatively equivalent integer expressions structurally identical for later CSE without relying on object identity, hash iteration, or floating-point algebra assumptions. Floating-point operands are deliberately left untouched.

Common-subexpression elimination is deliberately exact rather than algebraic. It only merges attribute-free `add`, `mul`, and `relu` operations with the same opcode, operand identities in the same order, and identical result types. It does not deduplicate constants or independently apply commutativity.

Virtual buffers remain single-write. Physical reuse is computed separately from virtual-buffer liveness, which keeps aliasing decisions explicit and testable. The planner does not perform in-place kernels: a physical slot is reusable only when its previous virtual buffer's last use is strictly before the new virtual allocation.

Loop IR is also deliberately conservative. Physical buffers are allocated before loop kernels, kernel outputs may overwrite only slots whose earlier virtual value is already dead, and a kernel output may not alias any input read by that same kernel. Broadcasting is represented by deterministic index maps rather than delegated implicitly to NumPy.

Elementwise fusion is explicit rather than automatic in `lower_to_loops()`. It currently combines only an adjacent `add` or `mul` producer with its immediately following ReLU when both loops have the same iteration shape, the ReLU consumes the producer through an identity map, the producer value has no other use before its physical slot is overwritten, and the fused output does not alias either producer input. The producer's broadcast maps are retained exactly. Both the interpreter and generated C force the binary intermediate through the fused output dtype before applying ReLU, preserving fixed-width integer overflow behavior and floating-point rounding/NaN/signed-zero semantics.

Native execution separates deterministic code generation from process-local artifact reuse. An exact `(compiler command, generated C source)` match reuses the already loaded shared library; a different source or compiler command compiles independently, and compilation failures are never inserted into the cache. Each cached artifact owns its build directory until `clear_native_cache()` or process exit. Cache clearing releases Windows DLLs before deleting their build directories, preserving the platform lifecycle invariant that loaded DLLs cannot be unlinked like POSIX shared objects. The cache is intentionally process-local: no persistent on-disk cache is introduced by this stage.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python examples/basic.py
```

A native C compiler is required to exercise `execute_native()`: a `cc`-compatible GCC/Clang toolchain on POSIX-like systems, or an MSVC developer environment exposing `cl` on Windows. CI executes the full suite on Ubuntu and Windows for Python 3.11 and 3.13.

## Near-term compiler roadmap

The next independently testable milestone is a typed external tensor-input ABI so one compiled graph can execute caller-provided tensor data instead of embedding every input as a constant. That work should add explicit input values without weakening verifier guarantees, keep buffer/loop lowering deterministic, and preserve the current native output-pointer contract while extending it deliberately. Longer fusion chains, SIMD, parallel loop scheduling, a persistent on-disk artifact cache, and CUDA remain out of scope until that runtime-input boundary is explicit and well tested.
