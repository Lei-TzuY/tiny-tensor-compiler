# tiny-tensor-compiler

A compact, correctness-first tensor compiler. The current milestone implements real end-to-end vertical slices rather than framework scaffolding:

```text
Python tensor expressions + typed external inputs
-> explicit typed tensor IR (concrete or bounded runtime symbolic/affine/linear dimensions)
-> verifier
-> optional runtime symbolic solving and concrete specialization
-> constant folding / algebraic simplification / dead-code elimination / canonicalization / CSE
-> explicit virtual-buffer CPU IR
-> liveness-based physical memory planning
-> explicit loop/kernel IR with broadcast index maps, reshape copies, and verified read-only contiguous alias views
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

A separate deterministic memory planner maps virtual buffers onto physical slots. Reuse is allowed only after the previous virtual buffer's last use and only when the complete `TensorType` matches. External inputs therefore participate in the same explicit lifetime model as constants and computed values. Returned values remain live through the terminal return block, so co-returned values cannot be accidentally collapsed onto one physical slot while they are simultaneously observable. The opt-in `borrow_inputs()` transform is applied only after this planning step and re-verifies its rewritten Loop IR before execution. A later contiguous-alias rewrite may introduce read-only logical `LoopView` ids over those already-planned storage roots; it does not mutate the original memory plan or silently permit writes through an alias.

The next lowering layer makes elementwise iteration and broadcasting explicit. For example, adding tensors with shapes `(2, 1)` and `(1, 3)` produces a `(2, 3)` loop whose reads are indexed as `lhs[i0, 0]` and `rhs[0, i1]`. Scalar broadcasts use an empty input index map, and ReLU uses identity indexing. CPU execution interprets these loop kernels one output index at a time rather than delegating broadcasting to vectorized NumPy operations. First-class `reshape` remains value-oriented and initially lowers as a verified row-major copy with no broadcast `IndexMap`. After fusion and optional borrowed-input lifetime splitting, `alias_contiguous_reshapes()` may replace that copy with a read-only `LoopView` only when the source storage root is proven stable through the reshape result's final use. If the root is rewritten, the original copy remains. Reshape and the resulting view remain explicit fusion boundaries.

The same verified loop IR can be emitted as deterministic C11 source. Physical buffers become fixed-width typed local arrays, broadcast index maps become explicit row-major offset expressions, constants are embedded as typed literals, and external inputs become typed `const` pointers appended after the ordered output pointers. Non-scalar, non-empty kernels whose every input has the full output shape and an exact identity index map are emitted as one contiguous `int64_t n` loop; this preserves row-major semantics while exposing a simpler auto-vectorization target to the native C compiler. Reshape copy kernels use one explicit flat row-major `n` loop when the alias proof is not available. A verified contiguous `LoopView` instead emits only a typed read-only pointer alias such as `const int32_t *p_view = p_source;`; it performs no kernel iteration. In the default serial mode, proven-independent flat loops receive a compiler-specific vectorization dependency hint: MSVC uses `loop(ivdep)`, Clang uses `clang loop vectorize(enable)`, and GCC uses `GCC ivdep`. Scalar kernels, zero-extent kernels, and any kernel that still performs broadcasting retain the explicit nested `i0`, `i1`, ... loop form without that hint. The hint does not select a vector width and does not introduce parallel execution. Opting into `parallel=True` instead schedules verified non-scalar, non-empty general-C kernel iteration with `#pragma omp parallel for schedule(static)`: flat kernels parallelize `n`, broadcast kernels parallelize only the outer `i0`, and the implicit OpenMP barrier at the end of every kernel preserves the existing producer/consumer order. A `LoopView` receives no OpenMP pragma because it is only metadata/pointer aliasing. Input copies and terminal output copies remain serial, and explicit SSE2 kernels keep their existing serial vector-loop implementation rather than stacking OpenMP onto the SIMD path. For example, a single-output graph returning `f32` with `f32` and `i32` inputs exposes `tiny_tensor_run(float *out, const float *input0, const int32_t *input1)`, while two returned tensors expose `out0`, `out1`, then the same ordered inputs. Generated source exposes `tiny_tensor_run` through a portable export macro that expands to `__declspec(dllexport)` on Windows. `execute_native()` and `compile_native()` use `cc`-style GCC/Clang flags on POSIX-like systems and MSVC `cl /std:c11 /O2 /LD` on Windows, then invoke the resulting shared library through `ctypes`; parallel mode additionally selects `-fopenmp` for GCC-style commands or `/openmp` for MSVC. GCC-style native compilation uses `-fwrapv` so signed integer add/multiply behavior matches NumPy's fixed-width wrapping semantics. Floating-point ReLU source explicitly preserves NaN propagation and NumPy's `-0.0 -> +0.0` behavior.

## Implemented now

- SSA-like `Value` objects with producer metadata and explicit use-def edges
- strongly typed `TensorType(shape, dtype)` values (`i32`, `i64`, `f32`, `f64`)
- concrete typed `input`, `const`, `add`, `mul`, `relu`, `reshape`, and `return` operations plus bounded `SymbolicDim`, one-variable `AffineDim`, and canonical multi-symbol `LinearDim` support in tensor shapes
- one-or-more named runtime symbolic dimensions on arbitrary axes with exact direct/affine/relational runtime solving, concrete tensor-IR cloning, and post-specialization verification
- exact rational elimination for multi-symbol runtime shape equations, requiring a unique non-negative integer solution and rejecting inconsistent, underdetermined, fractional, or negative systems
- exact reshape element-count proofs by canonical polynomial identity over integer, symbolic, affine, and linear shape dimensions, with no target-only symbols
- verified C-order reshape value semantics with a copy fallback plus conservative internal copy elision through read-only contiguous alias views when storage lifetime safety is proven
- first-class `LoopView` logical buffers with independent `TensorType`, one resolved storage root, and verifier-enforced read-only alias lifetimes
- storage-root-aware verification that rejects writes while dependent views are live and still permits physical-slot reuse after each view's final use
- reusable `DynamicExecutable` handles that freeze the symbolic template and cache one ordinary `NativeExecutable` per complete deterministic symbolic binding
- ordered one-or-more tensor returns across frontend, verifier, reference execution, lowering, memory planning, loop execution, generated C, and native execution
- dense declaration-order external input indices with verifier enforcement
- NumPy-compatible concrete broadcasting plus conservative symbolic broadcasting (an expression with itself or dimension `1`; different symbolic/affine/linear expressions are not implicitly unified)
- verifier checks for operation arity, dominance/order, inferred result types, return structure, opcode legality, constant types, input indices, and use-def consistency
- constant-folding optimization with post-pass verification
- conservative algebraic simplification for exact integer `x + 0`, `0 + x`, `x * 1`, and `1 * x` identities when replacement type/shape exactly matches the result
- dead-code elimination for unused known-pure operations (`const`, `add`, `mul`, `relu`, `reshape`), including cascading producer cleanup and simplification residue
- deterministic integer commutative canonicalization for `add` and `mul`, ordered by current SSA definition order
- conservative common-subexpression elimination for repeated exact `add`, `mul`, `relu`, and `reshape` expressions
- deterministic lowering to explicit `BufferAlloc`, `BufferInput`, `BufferKernel`, and `BufferReturn` operations
- buffer-IR structural/type verification for allocation, input writes, read-before-write, kernel arity, inferred output types, and return validity
- liveness-based virtual-to-physical memory planning with exact-type reuse and deterministic lowest-slot selection
- explicit `LoopAlloc`, `LoopInput`, read-only `LoopView`, `LoopKernel`, and `LoopReturn` IR over planned physical storage plus logical aliases
- explicit broadcast index maps for elementwise `add`, `mul`, and `relu` loops; reshape copies deliberately use no broadcast map
- loop-IR verification for allocation/declaration order, input writes, read-before-write, storage-root alias lifetime, non-in-place outputs, iteration shape, index maps, kernel types, and return validity
- first-class `FusedExpression` metadata consumed by Loop IR verification, loop execution, generated C, and SIMD planning
- topology-driven `fuse_elementwise()` with legacy two- through four-node chain/tree/chain-tree compatibility forms plus bounded five-/six-node structured `fused_dag` kernels selected by a deterministic static materialization/input-footprint heuristic
- deterministic C11 source generation with ordered typed output pointers followed by typed `const` external-input pointers, plus pointer-only lowering for verified contiguous views
- conservative contiguous-loop linearization for non-scalar, non-empty generated-C kernels with exact full-shape identity indexing, while scalar, zero-extent, and broadcast kernels keep explicit nested loops
- compiler-specific vectorization dependency hints only on those proven-independent contiguous loops, without selecting a fixed SIMD width or changing fallback kernel semantics
- expression-driven SSE2 selection for exact contiguous `int32` add/ReLU semantics, including eligible all-add generic DAGs, while expressions containing multiplication retain the general C fallback
- opt-in verified native OpenMP scheduling for general non-scalar/non-empty Loop kernels, with static scheduling, per-kernel barriers, broadcast outer-loop scheduling, GCC-style/MSVC compiler integration, and dynamic-specialization forwarding
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
- CPU execution through explicit scalar loop iteration over planned physical NumPy buffers plus zero-copy NumPy reshape views for verified `LoopView` aliases
- direct tensor-IR reference execution and separately lowered CPU execution
- malformed-IR tests, broadcasting tests, deterministic dump tests, randomized NumPy differential tests, generated-C syntax checks, cross-platform native differential tests, fusion/overflow/generic-DAG regressions, OpenMP scheduling/lifetime regressions, reshape copy/symbolic/purity regressions, alias-view lifetime/copy-elision regressions, external-input ABI/cache regressions, multi-output ABI/alias regressions, zero-copy regressions, direct/affine/relational runtime-symbolic specialization regressions, persistent-cache regressions, reusable-native-executable regressions, linting, and CI

Python scalar literals are coerced to the peer tensor's dtype (`float32_tensor * 2` remains `f32`). Tensor-vs-tensor operations use explicit `numpy.result_type` promotion.

External inputs remain exact and explicit. The ordinary `compile_module()` path requires fully concrete `GraphBuilder.input(shape, dtype)` shapes; runtime calls must match the declared input count, shape, and dtype with no silent cast. `compile_dynamic_module()` adds a bounded specialization boundary: named `SymbolicDim` values may appear directly, inside one-variable positive affine terms such as `2*B+1`, or inside positive multi-symbol linear expressions such as `2*B+W+3`. Direct and affine occurrences retain their exact binding behavior. Multi-symbol axes contribute linear equations; known bindings are substituted and the remaining system is solved with exact rational elimination. The system must uniquely determine every unresolved symbol as a non-negative integer, after which the tensor module is cloned, fully concretized, reverified, and only then enters the existing Buffer/Loop/C11/native pipeline. Complete bindings such as `B=2,W=3` and `B=2,W=7` cache separate native specializations; a later identical binding reuses the first without making the physical backend itself dynamic. Selecting `parallel=True` is frozen into that dynamic executable and forwarded independently to each concrete native specialization. Reshape target types participate in the same specialization: source and target element counts must already be proven identical as symbolic polynomials before the op is admitted, and both types are concretized before physical lowering.

Copied input materialization remains the default compatibility behavior. Opting into `borrow_inputs()` or `compile_module(..., borrow_inputs=True)` / `compile_dynamic_module(..., borrow_inputs=True)` instead requires exact NumPy, C-contiguous, aligned arrays and binds verified external slots directly; any input that would require a hidden normalization copy is rejected. The high-level compiler runs borrowed-input lifetime splitting before alias-view rewriting, so a safe reshape may then point directly at that read-only input storage without mutating it. Public reshape results remain ordinary values: terminal native results are copied into caller-owned output arrays even when their internal producer is a zero-copy view. Subtraction/division, negative symbolic coefficients, nonlinear symbolic products, inequalities, named inputs, runtime-sized physical buffers, inferred `-1` reshape dimensions, non-zero view offsets, arbitrary strides, transpose/slicing, and write-through views are not part of the current bounded shape system.

Algebraic simplification is intentionally conservative: floating-point neutral-element rewrites are not enabled yet because preserving strict IEEE behavior, including signed zero and NaN edge cases, takes priority over reducing operation count.

Dead-code elimination is side-effect conservative. It only removes currently known pure operations (`const`, `add`, `mul`, `relu`, `reshape`) whose results have no uses; `input` remains explicit so the declared runtime signature stays stable, and terminators such as `return` are never candidates.

Canonicalization currently reorders only integer `add` and `mul` operands. Earlier SSA definitions sort first, which makes commutatively equivalent integer expressions structurally identical for later CSE without relying on object identity, hash iteration, or floating-point algebra assumptions. Floating-point operands are deliberately left untouched.

Common-subexpression elimination is deliberately exact rather than algebraic. It only merges attribute-free `add`, `mul`, `relu`, and `reshape` operations with the same opcode, operand identities in the same order, and identical result types. It does not deduplicate constants, inputs, or independently apply commutativity.

Virtual buffers remain single-write. Physical reuse is computed separately from virtual-buffer liveness, which keeps base allocation decisions deterministic. The planner itself still requires exact `TensorType` reuse and does not perform in-place kernels. Read-only alias views are introduced only after planning: each logical view resolves to one storage root, and Loop IR verification forbids writes to that root until the view's final use. Root reuse after the alias dies remains legal. The alias rewrite may leave a now-unused planned reshape destination allocation in place, so this phase does not claim allocation-count reduction.

Loop IR is deliberately conservative. Physical buffers are allocated before execution, external inputs explicitly establish storage contents, and kernel writes target allocated storage roots only. Broadcasting is represented by deterministic index maps rather than delegated implicitly to NumPy. A contiguous `LoopView` is a separate read-only logical id with its own shape and dtype metadata but the same resolved storage root as its source. View dtype and element count must match the source, kernel input/output safety is checked by storage root, and any root write while a view is live is rejected. Direct, affine, and relational symbolic dimensions are fully resolved before this layer, so Loop IR continues to use concrete integer shapes.

Elementwise fusion remains explicit rather than automatic in `lower_to_loops()`. The planner builds dependency edges from the original adjacent binary kernel order and refuses any candidate that would require reassociation or kernel reordering. Every fused internal logical value must have exactly one internal consumer and no later external use, internal edges must use identity indexing, all intermediate/output values must use one exact `i32` or `i64` type, and the final output may not alias a fused leaf input. Returned intermediates therefore block fusion instead of becoming unobservable. Fusion runs before the alias-view rewrite, so reshape/view boundaries are never reinterpreted as elementwise identity indexing.

Two- through four-node legal shapes retain the historical checked chain/tree/chain-tree opcode spellings for compatibility. Five- and six-node legal binary windows instead become one structured `fused_dag` Loop kernel. Its `FusedExpression` metadata contains the ordered leaf names and scalar semantic steps; the opcode itself intentionally does not encode a new combinatorial family name. CPU interpretation, generated C, and SIMD planning consume those same steps. One following identity-indexed ReLU may be absorbed into the generic expression without creating another opcode family.

When more than one legal window begins at the same operation, the planner ranks candidates by the number of eliminated intermediate materializations, then the smaller external-input footprint, then the larger coherent binary window. This is a static compiler heuristic only. It is not a wall-clock performance claim, and CI duration is not treated as benchmark evidence. Shared internal subexpressions, non-adjacent fusion, floating-point generic DAGs, windows larger than six binary nodes, and any form requiring reassociation remain out of scope.

Generated-C loop scheduling is deliberately conservative. In serial mode, a kernel is flattened only when it is non-scalar, has at least one element, and every input has the full output shape with an exact identity index map. In that case a single row-major `n` loop reads and writes matching `[n]` offsets, preserving the same per-element fixed-width arithmetic and ReLU semantics. Because verified Loop IR forbids a kernel output from aliasing any input storage root, those flat loops have no output/input loop-carried dependency; generated C therefore emits `TINY_TENSOR_VECTORIZE_LOOP` immediately before them, expanding to the platform compiler's dependency/vectorization hint. A reshape that cannot be safely aliased retains its flat copy loop. A verified contiguous view emits a pointer alias and no loop. Any scalar, zero-extent, broadcast, or otherwise non-identity-indexed elementwise kernel falls back to the existing explicit nested-loop code without the hint.

Parallel mode is a separate opt-in scheduling policy over the same verified Loop IR. A non-scalar, non-empty general-C kernel receives `#pragma omp parallel for schedule(static)`: the row-major `n` loop is scheduled directly when linearized, a retained reshape copy schedules its flat `n` loop, while a broadcast kernel schedules only its outer `i0` loop and keeps inner loops nested within each iteration. Each pragma retains the default implicit barrier, so later kernels cannot observe a partially produced physical buffer; no kernel reordering, `nowait`, asynchronous return, or graph-level task scheduling is introduced. `LoopView` performs no iteration and receives no pragma. Scalar/zero-extent kernels remain serial. The explicit SSE2 code path also remains serial in this phase because it owns a vector loop plus scalar tail whose control flow is already separately verified. This is an executable scheduling capability, not a speedup claim: the compiler does not yet choose thread count, grain threshold, or a profitability policy, and CI timing is not benchmark evidence.

Native execution separates deterministic code generation, process-local loaded-artifact reuse, optional persistent storage, and reusable executable handles. Without `cache_dir`, serial behavior remains process-local: an exact `(compiler command, generated C source)` match reuses the already loaded shared library, and `clear_native_cache()` or process exit releases it and removes its temporary build directory. Passing `cache_dir` enables a versioned content-addressed disk cache. Its digest includes the generated source, full compiler command, resolved compiler executable path/size/mtime, operating system, platform, machine architecture, pointer width, and library format. A library is published into that cache only after successful compilation using an atomic same-filesystem `os.replace()`, so failed compilations do not create cache hits. Persistent library bytes are copied into a process-owned temporary staging directory before `ctypes` loads them; this keeps the persistent file itself immutable and avoids Windows DLL locking. If a persisted library cannot be loaded, it is treated as stale or corrupt, removed, and rebuilt. `clear_native_cache()` unloads and removes ordinary process staging directories but deliberately leaves the user-owned persistent cache intact for later processes.

Windows OpenMP execution adds one stricter lifetime boundary. MSVC OpenMP worker threads can outlive an individual parallel region long enough that unloading the generated DLL can invalidate outlined loop code still retained by the runtime. OpenMP-generated Windows artifacts are therefore removed from the ordinary unloadable cache and retained in a process-pinned registry until operating-system process teardown. `clear_native_cache()` does not `FreeLibrary` those artifacts. Their staging directories receive a PID marker; a later process performs best-effort cleanup only when that marked owner PID is no longer live. Serial Windows artifacts retain the historical unload/reacquire behavior. Different graphs, compiler commands (including the OpenMP flag), compiler executable fingerprints, targets, generated output signatures, concrete dynamic-shape bindings, or serial/parallel source forms therefore compile independently, while runtime input values with an already-cached specialization do not affect artifact identity. For preallocated multi-output execution, every destination is validated before the native call and must be disjoint from all runtime inputs and every other destination.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python examples/basic.py
```

A native C compiler is required to exercise `execute_native()`, `compile_native()`, or a newly observed `compile_dynamic_module()` specialization: a `cc`-compatible GCC/Clang toolchain on POSIX-like systems, or an MSVC developer environment exposing `cl` on Windows. Parallel native execution additionally requires that selected toolchain's OpenMP compiler/runtime support (`-fopenmp` or `/openmp`). CI executes the full suite, including OpenMP, reshape, and alias-view native regressions, on Ubuntu and Windows for Python 3.11 and 3.13.

## Near-term compiler roadmap

The runtime-input boundary, persistent native artifact cache, reusable compiled-executable handle, conservative contiguous-loop scheduling, cross-compiler vectorization hints, ordered multi-output ABI, verified zero-copy input binding, exact relational symbolic-shape specialization, first-class fused expressions, bounded cost-ranked generic DAG fusion, expression-driven SSE2 selection, verified opt-in native OpenMP loop scheduling, verified reshape value semantics, and read-only contiguous storage-root alias views are now explicit. Raising the fusion-node limit, adding reshape-chain micro-rewrites, or polishing alias syntax would be low-value farming. The next CPU-verifiable architectural frontier is an explicit logical layout descriptor carrying storage root, offset, shape, and strides, followed by one genuinely non-contiguous read-only transform such as transpose or bounded slicing with verifier-backed bounds and lifetime semantics. A second executable ISA/backend or accelerator backend remains a parallel option when its toolchain or hardware can be validated end to end.
