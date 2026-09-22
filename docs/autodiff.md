# Reverse-mode autodiff

`differentiate_module()` is a bounded source-to-source reverse-mode automatic-differentiation transform for verified tensor IR.

The transform does not introduce an autodiff runtime or a gradient-specific backend opcode. It rebuilds the selected forward ancestor slice and expresses every adjoint with ordinary verified tensor operations, then verifies the resulting module again. The transformed module therefore enters the existing reference, Buffer/Loop IR, generated-C, native, cache, and verification paths like any hand-written module.

## Supported first-phase contract

- The selected return value must be a static scalar `f32` or `f64` tensor.
- `wrt` identifies one or more runtime-input indices; gradients are returned in exactly that order.
- Every value on the backward-reachable slice must use the same exact floating dtype as the selected output.
- Supported backward operations are `add`, `mul`, `sum`, `reshape`, and whole-storage `view`, with `input` and `const` as leaves.
- Runtime inputs not on the selected loss slice are retained in the transformed module's input ABI. A requested but unused input receives an exact zero tensor gradient.
- Multiple reverse paths accumulate through ordinary tensor `add` operations.

## Broadcasting and reductions

Elementwise broadcasting is differentiated structurally. A contribution whose forward operand was broadcast is reduced across every inserted leading axis and every expanded size-one axis, then reshaped back to the exact operand type. This uses ordinary tensor `sum` and `reshape` operations rather than host-side NumPy precomputation.

The adjoint of `sum` reconstructs reduced singleton axes as needed and broadcasts the upstream gradient back to the source shape by multiplying with a typed all-ones constant. The adjoints of `reshape` and whole-storage `view` reshape the upstream gradient back to the source shape.

These rules preserve the compiler's existing row-major reshape and structural broadcasting semantics instead of introducing separate autodiff indexing rules.

## Fail-closed boundaries

The first phase intentionally rejects rather than guesses when a correct source transform would require semantics that the current bounded contract does not provide:

- non-scalar selected outputs;
- symbolic/dynamic shapes;
- integer gradients;
- mixed `f32`/`f64` backward slices, because there is no explicit cast primitive in this phase;
- ReLU, `prod`, `argmax`, transpose/slice/reverse aliases, mutation/write effects, and other unsupported backward operations;
- invalid or duplicate `wrt` indices and invalid return selection.

These are not claims that the operations are mathematically non-differentiable. They are explicit compiler capability boundaries. Future coverage should be added only with an executable, verifier-backed VJP/JVP rule rather than by silently approximating or materializing gradients outside the IR.

## Verification evidence

Focused regressions cover closed-form broadcast gradients, multi-path accumulation, reduction/reshape/view gradients, zero gradients for unused requested inputs, and deterministic rejection of unsupported or mixed-precision slices. Because `GraphBuilder.matmul()` lowers to supported `reshape`/`mul`/`sum` primitives rather than introducing a separate matmul IR opcode, a composed matmul loss is differentiated and compared across reference, CPU, Loop, and native execution. No performance claim is attached to autodiff; this milestone establishes transformation correctness and backend reuse.

## Next architectural frontier

This phase establishes program-to-program differentiation over a bounded pure subset. The next useful autodiff milestone should increase semantic depth rather than enumerate trivial operator rules: candidates include a verifier-backed finite-difference/gradient-consistency corpus, transpose/slice/reverse VJP rules with explicit scatter semantics, or higher-order differentiation once the transformed IR itself is deliberately accepted as an input contract.