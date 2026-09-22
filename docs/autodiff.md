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

## Next architectural frontier

The autodiff phase now has source-to-source VJPs, alias/write-effect integration, backend differential regressions, independent finite-difference consistency evidence, and lazy runtime symbolic specialization before differentiation. The next architectural gap is deeper dynamic-gradient policy—especially multi-symbol/adaptive specialization evidence—or, separately, an explicit higher-order contract once transformed effectful gradient IR is deliberately accepted as differentiable input.