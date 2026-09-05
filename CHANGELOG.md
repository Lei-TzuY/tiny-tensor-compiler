# Changelog

All notable project milestones are recorded here.

## [Unreleased]

### Added

- Ordered multiple tensor returns in the Python frontend and typed tensor IR.
- Multi-output reference and explicit loop-CPU execution while preserving the existing single-output `numpy.ndarray` result contract.
- Return-aware virtual-buffer liveness and physical memory planning so simultaneously returned values remain distinct when required.
- Fusion safety for intermediates that are both returned and consumed by later kernels.
- Generated C/native execution with one typed output pointer per returned tensor and one shared native entrypoint.
- Ordered multi-output native results plus optional preallocated output sequences with exact shape, dtype, layout, alignment, mutability, input-alias, and cross-output alias validation.
- Opt-in verified zero-copy external-input binding for loop CPU and native execution, including input-lifetime splitting when the original planned slot is reused later as scratch storage.
- Strict borrowed-input contracts requiring exact NumPy arrays with compiled shape/dtype plus C-contiguous aligned storage, so the zero-copy path never hides a normalization copy.
- `SymbolicDim` in typed tensor IR plus conservative symbolic broadcasting for named runtime dimensions.
- `compile_dynamic_module()` and reusable `DynamicExecutable` handles that bind one-or-more symbols on arbitrary axes, clone/reverify a fully concrete specialization, and cache one `NativeExecutable` per complete deterministic binding tuple.
- `bind_dynamic_shapes()` plus generalized `symbolic_dims` / `cached_bindings` execution metadata while retaining the existing one-symbol `symbolic_dim`, integer `specialize()`, and `cached_batch_sizes` convenience surface.
- Bounded one-variable `AffineDim` shape terms such as `2*B+1`, with exact runtime inversion, non-negative/integral constraint checks, concrete specialization, and unchanged physical lowering boundaries.
- Dynamic reference execution plus native multi-output, verified borrowed-input, multi-symbol broadcast, affine-shape, and zero-extent coverage across distinct runtime bindings.
- First-class structured fused expressions carried by Loop IR and consumed directly by verification, CPU execution, generated C, and SIMD planning while retaining legacy opcode spelling as a checked compatibility encoding.
- A bounded topology-driven elementwise fusion planner for two- through four-node integer `add`/`mul` DAGs, with logical-value lifetime checks that remain correct across physical-slot reuse.
- Fusion of safe mirrored producer order and reversed-root chain-tree layouts without reassociation or expansion of the existing fused opcode families.
- Expression-driven SSE2 selection for exact contiguous `int32` fused kernels whose canonical semantic steps use only addition and ReLU, including ReLU add-trees and all-add chain-trees without fused-opcode whitelists.

### Compatibility

- Single-output generated C and the existing `out=np.ndarray` native call contract remain unchanged.
- External inputs still use the historical copied-buffer path by default; zero-copy binding is explicitly selected through `borrow_inputs()` or `compile_module(..., borrow_inputs=True)`.
- `compile_module()` remains the eager concrete-shape entrypoint. Symbolic tensor IR must use the explicit `compile_dynamic_module()` specialization boundary before physical lowering.
- Existing one-symbol dynamic callers retain `bind_dynamic_batch()`, `DynamicExecutable.symbolic_dim`, integer `specialize(size)`, and `cached_batch_sizes`; multi-symbol executables use complete binding mappings instead of ambiguous batch-only values.
- Plain `SymbolicDim` shapes keep their existing direct runtime-binding behavior. Affine terms only add positive-scale/non-negative-offset relations over one named symbol and are fully concretized before Buffer/Loop IR.
- Distinct symbolic names are not implicitly unified; multi-variable expressions, subtraction, division, general equality solving, and runtime-sized physical IR remain out of scope.
- `tiny_tensor_compiler.loop_ir.fuse_elementwise()` remains as a compatibility entry point but delegates to the sole topology-driven fusion planner; the former family-specific fusion engine has been retired.
- SSE2 eligibility still requires exact contiguous `int32` storage. Expressions containing multiplication, broadcast indexing, scalar shapes, and other unsupported forms retain the general generated-C fallback.

## [0.1.0] - 2026-09-04

First frozen compiler milestone: a compact, correctness-first tensor compiler with a complete CPU vertical slice from typed tensor expressions to verified generated C and reusable native execution.

### Added

- Typed SSA-like tensor IR with static `i32`, `i64`, `f32`, and `f64` tensors.
- Explicit external inputs, NumPy-compatible broadcasting, and deterministic dtype promotion.
- Verifier coverage for tensor, buffer, and loop IR invariants.
- Constant folding, conservative algebraic simplification, DCE, canonicalization, and exact CSE.
- Explicit virtual-buffer lowering and liveness-based physical memory planning.
- Explicit loop IR with broadcast index maps and verifier-backed elementwise fusion.
- Deterministic C11 generation with portable GCC/Clang and Windows MSVC native compilation.
- Contiguous-loop linearization, compiler vectorization hints, and guarded SSE2 fast paths for selected exact contiguous `int32` kernels.
- Exact runtime input validation, preallocated native outputs, reusable native executable handles, and process-local / optional persistent native artifact caching.
- Differential, malformed-IR, overflow, broadcasting, cache, ABI, fusion, native, and cross-platform regression coverage.
- CI on Ubuntu and Windows with Python 3.11 and 3.13.

### Frozen scope

`v0.1.0` intentionally keeps shapes static, returns a single output, copies inputs into planned internal buffers, and uses a CPU/C11 backend. Dynamic shapes, multiple outputs, zero-copy input aliasing, generalized SIMD abstraction, general DAG fusion, parallel scheduling, and accelerator backends are deferred to later milestones.

### Release policy

The `v0.1.x` line is maintenance-oriented. Changes should address correctness, regressions, diagnostics, portability, tests, or bounded release-quality improvements. New language/backend scope belongs in a separately selected later milestone rather than an endless stream of opcode/SIMD micro-specializations.
