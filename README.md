# tiny-tensor-compiler

A compact, correctness-first tensor compiler. The current milestone implements real end-to-end vertical slices rather than framework scaffolding:

```text
Python tensor expressions + typed external inputs
-> explicit typed tensor IR (concrete or bounded runtime symbolic/affine/linear dimensions)
-> verifier
-> optional runtime symbolic solving and concrete specialization
-> constant folding / algebraic simplification / dead-code elimination / canonicalization / CSE
-> explicit virtual-buffer CPU IR
-> liveness-based physical memory planning with read-only storage aliases
-> explicit loop/kernel IR with broadcast index maps, reshape copies, and root-relative view layouts
-> cost-ranked verifier-backed elementwise fusion
-> deterministic generated C11 source with conservative contiguous-loop linearization and vectorization hints
-> optional verified OpenMP native loop scheduling
-> process-local / optional persistent native shared-library cache
-> reusable compiled native executable handles
-> NumPy-backed scalar CPU interpretation/reference
```

The NumPy CPU executor remains the semantic baseline while lowering is made progressively more compiler-like through explicit buffers, memory planning, loop IR, generated C, and native execution. The native path compiles verified generated C into a shared library, reuses exact generated-source/compiler matches within the current process, optionally persists content-addressed artifacts across processes through `cache_dir=`, and invokes one stable output-first ABI through `ctypes` on POSIX-like GCC/Clang toolchains and Windows MSVC. Multiple returned tensors use ordered typed output pointers in that same entrypoint; single-output programs preserve the original one-`out` ABI. Native loop parallelism is an explicit opt-in (`parallel=True`), not an implicit change to the historical serial backend.

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

`compile_native()` eagerly compiles or loads the native artifact and returns a reusable `NativeExecutable`. Repeated calls with new runtime input values reuse the same native code. The executable freezes the compiler command and cache identity chosen at compile time; if `clear_native_cache()` later releases process-owned serial shared libraries, the handle remains valid and safely reacquires or recompiles the same artifact on its next invocation. Single-output calls return one `numpy.ndarray`; multi-output calls return an ordered tuple and may receive an ordered sequence of preallocated output arrays. On Windows, OpenMP-enabled generated DLLs use a separate process-pinned lifetime because the worker runtime may retain generated code references beyond a parallel region; those pinned artifacts are intentionally not unloaded by `clear_native_cache()` and their marked staging directories are eligible for best-effort stale cleanup by a later process.

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

A separate deterministic memory planner maps owning virtual buffers onto physical slots while recording read-only aliases independently. Owning-slot reuse is allowed only after the previous root lifetime ends and only when the complete `TensorType` matches. Whole-storage views, positive-stride slices, and axis permutations receive logical handles with `StorageLayout(offset, strides)` instead of new storage; every direct or transitive alias use extends the storage root lifetime. Returned values remain live through the terminal return block. The opt-in `borrow_inputs()` transform is applied only after planning and re-verifies its rewritten Loop IR, including alias handles.

The next lowering layer makes elementwise iteration and broadcasting explicit. For example, adding tensors with shapes `(2, 1)` and `(1, 3)` produces a `(2, 3)` loop whose reads are indexed as `lhs[i0, 0]` and `rhs[0, i1]`. Scalar broadcasts use an empty input index map, and ReLU uses identity indexing. CPU execution interprets these loop kernels one output index at a time rather than delegating broadcasting to vectorized NumPy operations. First-class `reshape` is a verified row-major copy into distinct storage. In contrast, `view`, positive-stride `slice`, and compile-time `transpose` are explicit read-only aliases over one storage root; their logical shape and root-relative layout are carried separately and remain fusion boundaries.

The same verified loop IR can be emitted as deterministic C11 source. Owning physical buffers become fixed-width typed local arrays; logical views become typed pointer aliases plus verified root-relative layouts. Broadcast index maps and view strides become explicit row-major offset expressions, constants are embedded as typed literals, and external inputs become typed `const` pointers appended after the ordered output pointers. Non-scalar, non-empty kernels whose every input has the full output shape, identity indexing, and a contiguous logical layout are emitted as one contiguous `int64_t n` loop. Strided/permuted layouts use the general nested indexing path instead. Reshape copy kernels use one flat row-major `n` loop. In serial mode proven-independent flat loops receive the existing compiler-specific vectorization hint; the hint does not select a vector width or introduce parallel execution. Opting into `parallel=True` schedules verified non-scalar, non-empty general-C kernel iteration with `#pragma omp parallel for schedule(static)`: flat kernels parallelize `n`, broadcast/strided kernels parallelize only the outer `i0`, and the implicit barrier at the end of every kernel preserves producer/consumer order. Explicit SSE2 kernels remain on their proven serial vector-loop + scalar-tail path and require exact contiguous `int32` layouts. Native returns from internal views gather logical elements into the existing contiguous caller-owned output ABI rather than exposing internal alias lifetimes.

## Implemented now

- SSA-like `Value` objects with producer metadata and explicit use-def edges
- strongly typed `TensorType(shape, dtype)` values (`i32`, `i64`, `f32`, `f64`)
- concrete typed `input`, `const`, `add`, `mul`, `relu`, `reshape`, `view`, `slice`, `transpose`, and `return` operations plus bounded `SymbolicDim`, one-variable `AffineDim`, and canonical multi-symbol `LinearDim` support in tensor shapes
- one-or-more named runtime symbolic dimensions on arbitrary axes with exact direct/affine/relational runtime solving, concrete tensor-IR cloning, and post-specialization verification
- exact rational elimination for multi-symbol runtime shape equations, requiring a unique non-negative integer solution and rejecting inconsistent, underdetermined, fractional, or negative systems
- exact reshape element-count proofs by canonical polynomial identity over integer, symbolic, affine, and linear shape dimensions, with no target-only symbols
- verified C-order reshape copy semantics through reference execution, Buffer/Loop IR, explicit CPU execution, generated C, native execution, dynamic specialization, borrowed inputs, multi-output execution, and OpenMP scheduling
- read-only zero-copy whole-storage views, positive-stride single-axis slices, and complete compile-time axis permutations over one verified storage-root/layout model
- `StorageLayout(offset, strides)` root-relative descriptors with bounds verification, transitive alias composition, and layout-aware CPU/C/native indexing
- storage-root lifetime extension and generation tracking that reject stale aliases after the root is rewritten
- reusable `DynamicExecutable` handles that freeze the symbolic template and cache one ordinary `NativeExecutable` per complete deterministic symbolic binding
- ordered one-or-more tensor returns across frontend, verifier, reference execution, lowering, memory planning, loop execution, generated C, and native execution
- dense declaration-order external input indices with verifier enforcement
- NumPy-compatible concrete broadcasting plus conservative symbolic broadcasting (an expression with itself or dimension `1`; different symbolic/affine/linear expressions are not implicitly unified)
- verifier checks for operation arity, dominance/order, inferred result types, return structure, opcode legality, constant types, input indices, use-def consistency, storage-root aliasing, view-layout bounds, and view-generation freshness
- constant-folding optimization with post-pass verification
- conservative algebraic simplification for exact integer `x + 0`, `0 + x`, `x * 1`, and `1 * x` identities when replacement type/shape exactly matches the result
- dead-code elimination for unused known-pure operations, including reshape/view/slice/transpose aliases, cascading producer cleanup, and simplification residue
- deterministic integer commutative canonicalization for `add` and `mul`, ordered by current SSA definition order
- conservative common-subexpression elimination for repeated exact attribute-free `add`, `mul`, `relu`, `reshape`, and whole-storage `view` expressions
- deterministic lowering to explicit owning buffers plus `BufferView` aliases, kernels, inputs, and returns
- liveness-based virtual-to-physical memory planning with exact-type root reuse and transitive alias lifetime preservation
- explicit `LoopAlloc`, `LoopInput`, `LoopView`, `LoopKernel`, and `LoopReturn` IR with storage-root identity and generation checks
- explicit broadcast index maps for elementwise kernels plus root-relative logical layouts for read-only aliases
- first-class `FusedExpression` metadata consumed by Loop IR verification, loop execution, generated C, and SIMD planning
- topology-driven `fuse_elementwise()` with legacy two- through four-node chain/tree/chain-tree compatibility forms plus bounded five-/six-node structured `fused_dag` kernels selected by a deterministic static materialization/input-footprint heuristic
- deterministic C11 source generation with ordered typed output pointers followed by typed `const` external-input pointers
- conservative contiguous-loop linearization only when logical storage layouts are actually contiguous; non-contiguous slice/transpose inputs retain explicit general indexing
- compiler-specific vectorization dependency hints only on proven-independent contiguous loops, without selecting a fixed SIMD width or changing fallback kernel semantics
- expression-driven SSE2 selection for exact contiguous `int32` add/ReLU semantics, including eligible all-add generic DAGs, while expressions containing multiplication or non-contiguous layouts retain the general C fallback
- opt-in verified native OpenMP scheduling for general non-scalar/non-empty Loop kernels, with static scheduling, per-kernel barriers, broadcast/strided outer-loop scheduling, GCC-style/MSVC compiler integration, and dynamic-specialization forwarding
- Windows process-pinned OpenMP generated-DLL lifetime so native cache clearing cannot unload code still potentially retained by the OpenMP worker runtime
- exact runtime-input validation for input count, resolved shape, and dtype with no silent casting
- verified opt-in zero-copy input binding that rejects hidden normalization copies and preserves planner scratch reuse through input-lifetime splitting
- exact preallocated native-output validation for output count, shape, dtype, C-contiguity, writability, alignment, input overlap, and cross-output overlap
- contiguous row-major normalization before copied scalar/native execution while leaving caller arrays unmodified
- native CPU compilation/execution through `.so`, `.dylib`, or `.dll` libraries
- process-local native artifact reuse keyed by exact generated C source and compiler command, independent of runtime input values
- opt-in persistent native artifact reuse through `execute_native(..., cache_dir=...)` with compiler/target fingerprinting and atomic publication
- process-owned staging copies for persistent libraries so cached `.dll` / `.so` / `.dylib` files are not directly loaded or locked
- stale or corrupt persistent artifacts are discarded and rebuilt rather than poisoning later executions
- reusable `NativeExecutable` handles from eager `compile_native()` with frozen compiler/cache configuration and safe reuse after process-cache clears; Windows OpenMP executables instead retain their generated DLL for process lifetime
- explicit `clear_native_cache()` resource release plus automatic process-exit cleanup without deleting user-owned persistent cache files; Windows OpenMP process-pinned artifacts are intentionally excluded from unsafe early unload
- GCC/Clang-compatible native compilation on POSIX-like systems and MSVC `cl` compilation/loading on Windows
- CPU execution through explicit scalar loop iteration over planned physical NumPy storage and logical alias views
- direct tensor-IR reference execution and separately lowered CPU execution
- malformed-IR tests, broadcasting tests, deterministic dump tests, randomized NumPy differential tests, generated-C syntax checks, cross-platform native differential tests, fusion/overflow/generic-DAG regressions, OpenMP scheduling/lifetime regressions, reshape/view/slice/transpose alias regressions, external-input ABI/cache regressions, multi-output ABI/alias regressions, zero-copy regressions, direct/affine/relational runtime-symbolic specialization regressions, persistent-cache regressions, reusable-native-executable regressions, linting, and CI

Python scalar literals are coerced to the peer tensor's dtype (`float32_tensor * 2` remains `f32`). Tensor-vs-tensor operations use explicit `numpy.result_type` promotion.

External inputs remain exact and explicit. The ordinary `compile_module()` path requires fully concrete input shapes; runtime calls must match the declared input count, shape, and dtype with no silent cast. `compile_dynamic_module()` adds the bounded specialization boundary: named `SymbolicDim` values may appear directly, inside positive affine terms such as `2*B+1`, or inside positive multi-symbol linear expressions such as `2*B+W+3`. Multi-symbol axes contribute exact linear equations and the system must uniquely determine every unresolved symbol as a non-negative integer before Buffer/Loop/native lowering. Compile-time transpose simply moves already-typed dimensions between axes and adds no runtime shape equation.

Copied input materialization remains the default compatibility behavior. Opting into `borrow_inputs()` or `compile_module(..., borrow_inputs=True)` / `compile_dynamic_module(..., borrow_inputs=True)` instead requires exact NumPy, C-contiguous, aligned arrays and binds verified external root slots directly; any input that would require a hidden normalization copy is rejected. Read-only view/slice/transpose handles may then alias that borrowed root without materialization. Subtraction/division, negative symbolic coefficients, nonlinear symbolic products, inequalities, named inputs, runtime-sized physical buffers, inferred `-1` reshape dimensions, negative/reverse strides, writable alias kernels, and runtime permutation axes are not part of the current bounded system.

Algebraic simplification is intentionally conservative: floating-point neutral-element rewrites are not enabled yet because preserving strict IEEE behavior, including signed zero and NaN edge cases, takes priority over reducing operation count.

Dead-code elimination is side-effect conservative. It removes only known-pure operations whose results have no uses; this includes read-only reshape/view/slice/transpose transforms, while `input` remains explicit so the declared runtime signature stays stable and `return` is never a candidate.

Canonicalization currently reorders only integer `add` and `mul` operands. Earlier SSA definitions sort first, which makes commutatively equivalent integer expressions structurally identical for later CSE without relying on object identity, hash iteration, or floating-point algebra assumptions. Floating-point operands are deliberately left untouched.

Common-subexpression elimination is deliberately exact rather than algebraic. It merges only currently supported exact expression keys; attribute-bearing slice/transpose operations are not given attribute-aware CSE in this phase.

Owning virtual buffers remain single-write. Physical root reuse is computed separately from logical alias handles. Every direct or transitive view use extends the owning root's lifetime; the planner therefore cannot recycle a root while an alias may still observe it. Alias layouts preserve dtype and are verified against the backing root bounds. Whole-storage views, positive-stride slices, and axis permutations receive no independent physical allocation.

Loop IR remains conservative about writes. `LoopAlloc` values are storage roots; `LoopView` values are read-only logical handles with their own `TensorType` and root-relative `StorageLayout`. Kernels may read roots or views but may write only owning roots, and a kernel output may not share a storage root with any input. Every root write advances a generation; stale view handles created against an older generation are rejected on later read or return. Direct, affine, and relational symbolic dimensions are fully resolved before this layer.

Elementwise fusion remains explicit rather than automatic in `lower_to_loops()`. The planner builds dependency edges from adjacent binary kernels and refuses any candidate requiring reassociation or kernel reordering. Returned intermediates, storage aliases, and view creation remain observable boundaries. Reshape, view, slice, and transpose creation are not absorbed into or crossed by elementwise fusion in this phase.

Two- through four-node legal fused shapes retain historical checked chain/tree/chain-tree spellings. Five- and six-node legal binary windows use one structured `fused_dag` Loop kernel whose `FusedExpression` metadata contains the scalar semantics. Candidate ranking is a static structural heuristic, not benchmark evidence.

Generated-C scheduling remains deliberately conservative. Flat/SSE2 eligibility requires an actually contiguous logical layout in addition to the existing shape/index-map constraints. Non-contiguous positive-stride slices and ordinary axis permutations therefore use explicit general indexing from their verified `StorageLayout`; OpenMP may still schedule the outer loop of that general kernel. Internal aliases are zero-copy, while terminal native outputs remain contiguous caller-owned arrays populated in logical order.

Parallel mode is an opt-in scheduling policy over the same verified Loop IR. Each OpenMP kernel retains its implicit barrier; no kernel reordering, `nowait`, asynchronous return, or graph-level task scheduling is introduced. Scalar/zero-extent kernels remain serial and explicit SSE2 remains on its separate serial vector path. This is an executable scheduling capability, not a speedup claim.

Native execution separates deterministic code generation, process-local loaded-artifact reuse, optional persistent storage, and reusable executable handles. Persistent artifact identity includes the generated source, compiler command/fingerprint, operating-system target, ABI shape, and concrete dynamic specialization. `clear_native_cache()` releases ordinary process-owned serial libraries and leaves user-owned persistent cache files intact.

Windows OpenMP execution keeps its stricter process-pinned lifetime boundary: generated OpenMP DLLs are not unloaded by `clear_native_cache()` because the runtime may retain outlined code references. PID-marked staging directories are only eligible for best-effort cleanup by a later process once the owner PID is no longer live.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python examples/basic.py
```

A native C compiler is required to exercise `execute_native()`, `compile_native()`, or a newly observed `compile_dynamic_module()` specialization: a `cc`-compatible GCC/Clang toolchain on POSIX-like systems, or an MSVC developer environment exposing `cl` on Windows. Parallel native execution additionally requires that selected toolchain's OpenMP compiler/runtime support (`-fopenmp` or `/openmp`). CI executes the full suite, including OpenMP and alias-layout native regressions, on Ubuntu and Windows for Python 3.11 and 3.13.

## Near-term compiler roadmap

The project now has explicit typed shape transforms, exact runtime shape solving, bounded cost-ranked fusion, expression-driven SSE2, verified native OpenMP scheduling, and a read-only root-relative storage-layout subsystem covering contiguous whole-storage views, positive-stride single-axis slices, and arbitrary compile-time axis permutations. Raising fusion-node limits, adding more positive slice-step/permutation spellings, or tuning OpenMP knobs without controlled evidence would be low-value farming. The next CPU-verifiable storage frontier must change the alias model itself—most plausibly negative-stride/reverse semantics or verifier-backed writable alias regions with overlap/lifetime rules—while a second executable ISA/backend or accelerator backend remains a parallel architectural option when its toolchain/hardware can be validated.
