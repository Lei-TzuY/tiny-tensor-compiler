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

### Adaptive dynamic HVP admission

`AdaptiveDynamicHVPExecutable` preserves the same specialization order and single-`wrt` contract while reusing the existing adaptive VJP runtime-vector ABI and adaptive specialization cache. Original forward inputs solve the symbolic binding, the concrete forward module is transformed into its HVP, and only that executable HVP enters `compile_adaptive_module()`. Structural admission therefore measures the higher-order program itself. Each binding deterministically caches either native execution or the verified Loop fallback, with the appended vector remaining runtime ABI data rather than a source of symbolic bindings.

### Resource-managed dynamic HVP lifecycle

Native and adaptive dynamic HVP specializations now participate in the same deterministic resource-managed retention protocol as ordinary, gradient, and VJP dynamic execution. Cache hits refresh one LRU order, final managed native ownership performs the actual process-cache unload, externally retained native HVP handles may reacquire after eviction, and adaptive Loop decisions are evicted without falsely incrementing native-release accounting. The HVP facades reuse the existing artifact identity and ownership registry rather than introducing a higher-order-specific cache.

### Forward-mode Jacobian-vector products

`jacobian_vector_product_module()` introduces a bounded source-to-source tangent transform over the static floating subset. The transformed module preserves all original runtime inputs, appends one exact-type tangent input for each requested `wrt` input, and returns the tangent of the selected output. Constants receive exact zero tangents; `add` propagates tangent sums; `mul` applies the ordinary product rule with broadcast-capable tensor IR; and `sum`, reshape/view aliases, transpose, reverse, and positive-stride slice replay the same structural transform on the tangent value. Matmul therefore works compositionally through its existing primitive lowering rather than a JVP-specific opcode.

Forward mode now also has a bounded write-effect rule for verified `copy_into(root, target, source)` when `target` is a direct positive-stride slice of that exact fresh root. The tangent path emits the same ordinary `copy_into` over the independent tangent root/target/source values: untouched tangent elements remain unchanged, the written region is replaced by the source tangent, and the tangent root advances through the same storage-generation discipline as the primal root. Source tangents derived from pre-write slices are computed before the effect and remain valid through the write. Non-direct writable targets remain fail-closed.

The transform is executable across reference, CPU, Loop, and native execution. Symbolic shapes, mixed-precision tangent slices, unsupported pure operators, non-direct copy targets, and arithmetic write effects currently fail closed. These are compiler capability boundaries, not claims that the programs are mathematically non-differentiable.

### Dynamic forward-mode JVP specialization

`DynamicJVPExecutable` carries the bounded forward-mode transform across symbolic specialization without introducing a tangent-specific cache or backend. Original primal runtime inputs alone solve the complete symbolic binding; the forward module is specialized first; `jacobian_vector_product_module()` then appends one exact-type tangent input per requested `wrt` input to the concrete ABI. Tangent inputs never participate in symbolic inference. Multi-symbol and multi-`wrt` bindings therefore reuse the same native specialization cache, compile budget, borrowed-input, parallel, timeout, and deadline plumbing as the existing dynamic differentiation executables.

### Adaptive dynamic JVP admission

`AdaptiveDynamicJVPExecutable` preserves the dynamic JVP ordering while applying the existing adaptive execution policy to the concrete transformed module. Original primal inputs alone solve symbolic bindings; the forward module is specialized first; `jacobian_vector_product_module()` constructs the exact tangent ABI; and only then does the concrete JVP enter `compile_adaptive_module()`. The structural budget therefore measures the executable JVP itself. Each binding deterministically caches either native execution or the verified Loop fallback, including multi-`wrt` tangent ordering, while tangent inputs remain excluded from symbolic inference.

### Resource-managed dynamic JVP lifecycle

Native and adaptive dynamic JVP specializations now participate in the same deterministic resource-managed retention protocol as ordinary, gradient, VJP, and HVP dynamic execution. Cache hits refresh one LRU order, final managed native ownership performs the actual process-cache unload, externally retained native JVP handles may reacquire after eviction, and adaptive Loop decisions are evicted without falsely incrementing native-release accounting. Tangent inputs remain ABI data and do not alter the symbolic binding key or artifact identity.

### Generation-aware arithmetic-write JVP

Direct-slice `binary_into` now has bounded forward-mode semantics over exact floating `add` and `mul`. For `add`, the tangent path applies the same verified partial arithmetic write to the tangent root, updating the target region as `target_tangent + source_tangent` while preserving untouched elements. For `mul`, the transform materializes the target primal before the primal write advances storage generation, computes `target_tangent * source_primal + taped_target_primal * source_tangent`, and replaces the tangent target region through ordinary verified `copy_into`. Broadcasted sources therefore follow the same concrete tensor inference as the primal operation, and source tangents derived from pre-write slices remain valid because they are computed before the tangent write. Non-direct partial targets remain fail-closed.

### Full-root `binary_inplace` JVP

Exact-shape floating `binary_inplace` effects now have bounded forward-mode rules without introducing tangent-specific mutation IR. `add` produces `root_tangent + source_tangent`. `mul` snapshots the pre-write root primal before the primal generation advances and produces `root_tangent * source_primal + taped_prewrite_root * source_tangent`. Because the source must use a different storage root, its primal and tangent remain independently valid; source tangents derived from pre-write root computations are materialized before the write and therefore survive the generation advance.

### JVP correctness campaign

`run_jvp_consistency_campaign()` provides an independent numerical and algebraic evidence layer for forward mode. It deterministically generates smooth static `f64` tensor-output graphs, compares the analytic JVP with a central directional finite difference of the original verified module, and separately checks the adjoint identity `dot(J·v, c) == dot(v, Jᵀ·c)` by composing the existing runtime-seeded JVP and VJP transforms on the same primal graph. Generated directions and cotangents are deterministic, and failures preserve a stable signature while shrinking operations, tensor side length, primal inputs, tangent, and cotangent. The minimized case retains a canonical primal repro artifact plus the exact frozen direction vectors. Non-smooth operators remain outside this campaign so finite differences do not manufacture false failures; no performance claim is attached to the evidence.

### Joint primal-plus-JVP execution

`value_and_jacobian_vector_product_module()` is the first linearization-oriented vertical slice. It reuses the forward-mode transform to clone one verified primal ancestor graph, appends the requested runtime tangent inputs in the existing `wrt` order, propagates tangent values beside that single primal clone, and returns `(primal_output, tangent_output)` from the same transformed program. The joint program therefore avoids building and executing a separate primal module beside a JVP module, including across the verified writable-effect rules and pre-write primal snapshots.

### Reusable pushforward linearization

`compile_pushforward_linearization()` is the first retained-state linearization contract. For one concrete pure floating graph, the transform splits execution into two verified modules instead of rebuilding a joint primal+JVP program per query. The one-shot primal module preserves the original input ABI, evaluates the selected primal once, and returns that output plus only the non-input/non-constant intermediate primal values that later product-rule multiplications actually require. The reusable pushforward module receives frozen owned copies of the original primal inputs, those retained tape values, and a fresh tangent tuple; it contains tangent propagation but does not clone the original primal arithmetic/reduction chain.

`PushforwardLinearizationState` therefore services multiple `pushforward()` calls after one `linearize()` call. Caller-owned primal arrays are copied into retained state, so later mutation of those arrays cannot change derivative queries. The state intentionally holds the pushforward executable and retained tensors but not the primal executable; subsequent tangent queries have no execution path that recomputes the primal. Direct-slice `copy_into` now crosses this retained-state boundary: the reusable tangent program replays the same verified tangent-storage generation, while any nonlinear tape value that aliases the current root and would be invalidated by the primal write is materialized into an owned tensor before that generation advances. `binary_into` and `binary_inplace` remain fail-closed until their arithmetic pre-write dependencies use the same lifetime contract. Borrow-retained inputs, symbolic specialization, and performance claims are also outside this contract.

### Reusable pullback linearization

`compile_pullback_linearization()` mirrors the retained-state pushforward contract for reverse mode on the same bounded static floating subset. Its one-shot primal/tape module evaluates the selected primal once and returns only the nonlinear intermediate primals required by later derivative propagation, including generation-specific owned snapshots captured before a supported direct-slice `copy_into`. `PullbackLinearizationState` freezes owned copies of the original inputs and tape values, exposes the retained primal, and accepts repeated runtime cotangents through `pullback()`; the reusable reverse program propagates adjoints directly from those retained values, zeroing the overwritten root region and gathering the source cotangent for `copy_into`, without cloning or executing the original primal arithmetic/write chain.

Pushforward and pullback use the same bounded-graph validation and deterministic nonlinear tape-value selection. `compile_linearization()` validates that the independently constructed pushforward and pullback primal/tape modules are identical, then retains only one primal/tape executable and one frozen retained-value layout. For supported direct-slice `copy_into`, derivative-required aliases of the current root are copied into compiler-owned tape tensors before the write advances storage generation; the state then freezes those snapshots alongside caller inputs. `LinearizationState` exposes both derivative directions from that shared state and intentionally does not retain the primal executable, so repeated tangent and cotangent queries cannot replay the original forward graph or observe caller mutation after `linearize()`.

This first generation-aware retained-write contract is deliberately narrow. Direct positive-stride slice `copy_into` is supported; non-direct copy targets and arithmetic writes remain fail-closed. No mutable alias handle is retained across query lifetimes: only immutable owned tensor snapshots and the retained primal result cross the boundary.

## Next architectural frontier

Generation-aware direct-copy retention is now closed across reusable pushforward, reusable pullback, and the shared linearization state. The next architectural gap is arithmetic-write retained snapshots. Extend the same single tape/lifetime protocol to direct-slice `binary_into add/mul` and then full-root `binary_inplace add/mul`, with particular attention to multiplicative rules that require exact pre-write target/root primals after storage generation advances. Non-direct partial targets should remain fail-closed rather than becoming alias-variant farming. Symbolic retained-state specialization, borrowed retained inputs, and performance claims remain separate later concerns.