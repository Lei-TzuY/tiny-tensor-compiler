# Reverse-mode autodiff

`differentiate_module()` is a bounded source-to-source reverse-mode automatic-differentiation transform for verified tensor IR.

The transform does not introduce an autodiff runtime or a gradient-specific backend opcode. It rebuilds the selected forward ancestor slice and expresses every adjoint with ordinary verified tensor operations, then verifies the resulting module again. The transformed module therefore enters the existing reference, Buffer/Loop IR, generated-C, native, cache, and verification paths like any hand-written module.

## Supported bounded contract

- The selected return value must be a static scalar `f32` or `f64` tensor.
- `wrt` identifies one or more runtime-input indices; gradients are returned in exactly that order.
- Every value on the backward-reachable slice must use the same exact floating dtype as the selected output.
- Supported backward operations are `add`, `mul`, `sum`, `reshape`, whole-storage `view`, `transpose`, `reverse`, and positive-stride `slice`, with `input` and `const` as leaves.
- Runtime inputs not on the selected loss slice are retained in the transformed module's input ABI. A requested but unused input receives an exact zero tensor gradient.
- Multiple reverse paths accumulate through ordinary tensor `add` operations.

## Broadcasting and reductions

Elementwise broadcasting is differentiated structurally. A contribution whose forward operand was broadcast is reduced across every inserted leading axis and every expanded size-one axis, then reshaped back to the exact operand type. This uses ordinary tensor `sum` and `reshape` operations rather than host-side NumPy precomputation.

The adjoint of `sum` reconstructs reduced singleton axes as needed and broadcasts the upstream gradient back to the source shape by multiplying with a typed all-ones constant. The adjoints of `reshape` and whole-storage `view` reshape the upstream gradient back to the source shape.

## Alias VJPs

`transpose` applies the exact inverse permutation to the upstream gradient, while `reverse` applies the same axis reversal because the transform is self-inverse. A `slice` VJP is a true scatter: autodiff materializes one compiler-owned zero root, recreates the verified slice alias on that root, and emits ordinary `copy_into` to write the upstream gradient into the selected region. The differentiated program therefore exercises the existing storage-generation, alias-layout, Buffer/Loop effect, generated-C, and native execution machinery instead of performing host-side NumPy scatter work.

These rules preserve the compiler's existing row-major reshape, structural broadcasting, and root-relative alias semantics instead of introducing separate autodiff indexing rules.

## Fail-closed boundaries

The first phase intentionally rejects rather than guesses when a correct source transform would require semantics that the current bounded contract does not provide:

- non-scalar selected outputs;
- symbolic/dynamic shapes;
- integer gradients;
- mixed `f32`/`f64` backward slices, because there is no explicit cast primitive in this phase;
- ReLU, `prod`, `argmax`, forward mutation/write effects, and other unsupported backward operations;
- invalid or duplicate `wrt` indices and invalid return selection.

These are not claims that the operations are mathematically non-differentiable. They are explicit compiler capability boundaries. Future coverage should be added only with an executable, verifier-backed VJP/JVP rule rather than by silently approximating or materializing gradients outside the IR.

## Verification evidence

Focused regressions cover closed-form broadcast gradients, multi-path accumulation, reduction/reshape/view gradients, zero gradients for unused requested inputs, and deterministic rejection of unsupported or mixed-precision slices. A composed transpose → reverse → strided-slice loss must reconstruct its gradient through inverse aliases plus a real `copy_into` scatter and agree across reference, CPU, Loop, and native/OpenMP execution. Because `GraphBuilder.matmul()` lowers to supported `reshape`/`mul`/`sum` primitives rather than introducing a separate matmul IR opcode, matmul gradients also reuse the ordinary backend pipeline.

`run_gradient_consistency_campaign()` adds an independent numerical evidence layer. It deterministically generates smooth static `f64` scalar-loss graphs, computes analytic gradients through `differentiate_module()`, compares them with central finite differences of the original verified module, and shrinks the first reproducible divergence by operations, shape, and input values. Generated cases and minimized failures use the existing canonical repro artifact format. ReLU and other non-smooth rules are intentionally excluded so finite differences do not manufacture false failures. No performance claim is attached to the campaign.

## Dynamic specialization boundary

`compile_dynamic_gradient_module()` deliberately keeps unresolved symbolic dimensions out of `differentiate_module()`. Runtime inputs first bind the existing symbolic shape system, the forward module is specialized and reverified to exact concrete shapes, and only then is reverse-mode AD applied. The resulting concrete gradient module enters the normal native pipeline and is cached by the same runtime binding key used by dynamic compilation. This ordering keeps shape-dependent zero/one materialization and slice-gradient `copy_into` storage extents concrete, while preserving existing specialization budgets, compiler timeouts, borrowing, OpenMP, and native cache behavior.

`compile_adaptive_dynamic_gradient_module()` applies the same ordering but sends each concrete gradient module through the existing adaptive admission path. Structural limits are therefore measured on the executable gradient program, not estimated from the symbolic or forward module. Each complete multi-symbol binding caches one `AdaptiveExecutable`; within-budget bindings retain the native backend, while `CompileBudgetExceeded` retains the verified Loop fallback. Cache hits reuse the original backend decision, and non-budget verifier/autodiff/native failures remain fail-closed instead of being converted into fallback.

`compile_dynamic_vjp_module()` applies the same specialization-first rule to runtime-seeded tensor-output VJPs. Symbolic bindings are solved only from the original forward inputs; the appended cotangent is deliberately excluded from shape solving. After specialization, `vector_jacobian_product_module()` appends a concrete cotangent input whose type exactly matches the specialized selected output, and the resulting native executable is cached by the same complete binding key. A wrong cotangent shape or dtype therefore fails against the concrete VJP ABI rather than influencing symbolic inference.

## Runtime-seeded VJP and higher-order boundary

`vector_jacobian_product_module()` accepts any static floating selected output supported by the bounded backward slice. The transformed module preserves the original runtime inputs and appends one cotangent input whose tensor type exactly matches that selected output. Reverse propagation is otherwise the same verified tensor-IR transform used by `differentiate_module()`, so reference, CPU, Loop, generated-C, and native backends execute the resulting VJP without a gradient-specific runtime opcode.

This makes higher-order composition executable over transformed IR. For a scalar loss, `differentiate_module()` can first produce a tensor gradient; applying `vector_jacobian_product_module()` to that gradient module and supplying a runtime vector computes a reverse-over-reverse Hessian-vector product.

### Bounded `copy_into` VJP

A verified `copy_into(root, target, source)` is differentiable when `target` is a direct positive-stride `slice` of that exact fresh root. Source and side computations may depend on pre-write root/view values because the generation-aware primal tape materializes the backward-relevant aliases before the write. Under that bounded contract, the target is address/layout metadata rather than an independently differentiable tensor value. The output cotangent splits into two ordinary IR contributions: the source receives the cotangent gathered from the overwritten slice (then existing unbroadcast semantics recover a broadcast source shape), while the pre-write root receives a compiler-owned copy of the cotangent with that slice overwritten by exact zeros. This preserves untouched infinities and signed zeros without using a numeric `0/1` mask and keeps the resulting adjoint inside verified write-effect IR.

The transform deliberately rejects composed/non-slice writable targets. For a direct-slice write, reverse mode records a pre-write primal tape before cloning the effect: every backward-relevant root/view value that aliases the exact current root is materialized into compiler-owned storage for that generation. Reverse rules still accumulate cotangents into the original cloned SSA values; only rules that need primal data (currently multiplication) read the taped replacement after the write invalidates the original generation. Tape materializations are not inserted into the reverse traversal, so they preserve primal data without becoming extra mathematical operations. Each ordered `copy_into` generation captures its own tape, allowing dependencies such as `source = target * target` to differentiate across multiple writes. Materialization uses multiplication by exact positive one rather than a numeric mask, preserving infinities and signed zeros.

This remains a bounded differentiated-effect rule, not a claim of unrestricted mutation AD.

### Full-root `binary_inplace` VJP

Exact-shape floating `binary_inplace` effects now reuse the same generation-aware tape without partial-region machinery. For `add`, both the pre-write root and the different-root source receive the full output cotangent. For `mul`, the root receives `cotangent * source`, while the source receives `cotangent * taped_prewrite_root`. Because the tape is captured before every supported write effect, source-side computations derived from the pre-write root remain differentiable after the root generation advances.

### Adaptive dynamic VJP admission

`AdaptiveDynamicVJPExecutable` preserves the dynamic VJP ordering while applying the existing adaptive execution policy to the concrete transformed module. Original forward inputs alone solve symbolic bindings; the forward module is specialized first; the concrete selected output fixes the appended cotangent ABI; and only then does the resulting VJP enter `compile_adaptive_module()`. The structural budget therefore measures the executable VJP itself. Each binding deterministically caches either native execution or the verified Loop fallback, and repeated bindings reuse that exact backend decision.

### Resource-managed dynamic VJP lifecycle

Native and adaptive dynamic VJP specializations now participate in the same deterministic resource-managed retention protocol as ordinary and gradient dynamic execution. Cache hits refresh one LRU order, final managed native ownership performs the actual process-cache unload, externally retained native VJP handles may reacquire after eviction, and adaptive Loop decisions are evicted without falsely incrementing native-release accounting. The VJP facades therefore reuse the existing artifact identity and ownership registry rather than introducing a differentiation-specific cache.

### Dynamic Hessian-vector products

`DynamicHVPExecutable` carries reverse-over-reverse composition across the symbolic/runtime boundary without introducing a higher-order IR opcode. The contract is intentionally bounded to exactly one `wrt` input. Original forward inputs alone solve symbolic bindings; the scalar-loss forward module is specialized first; `differentiate_module()` constructs the concrete first gradient; and `vector_jacobian_product_module()` differentiates that gradient with an appended runtime vector whose exact tensor type is fixed by the specialized gradient output. The resulting HVP reuses the existing native specialization cache, compile budget, borrow-input, parallel, timeout, and deadline plumbing.

This path is executable through ordinary higher-order tensor IR, including differentiated-effect gradients such as slice-scatter `copy_into`; the dynamic wrapper adds specialization and ABI handling rather than a new differentiation engine.

## Next architectural frontier

The current writable-effect family has bounded reverse semantics for direct-slice `copy_into`, direct-slice `binary_into add/mul`, and full-root `binary_inplace add/mul`, all reusing ordinary tensor IR, verified storage generations, and one pre-write primal tape. Runtime-seeded VJPs cross symbolic specialization through native and adaptive dynamic execution and share the same managed lifecycle as ordinary and gradient specializations. Bounded single-input HVPs now also cross native dynamic specialization by composing the existing gradient and VJP transforms after concrete forward specialization. Non-direct partial targets, unsupported operators, multi-input/block Hessians, and full Hessian materialization remain fail-closed or out of scope rather than being farmed as variants. The next architectural gap is adaptive dynamic HVP admission/fallback so each concrete HVP is measured under the existing structural budget and deterministically caches either native execution or verified Loop fallback.