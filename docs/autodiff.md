# Reverse-mode autodiff

`differentiate_module()` is a bounded source-to-source reverse-mode automatic-differentiation transform for verified tensor IR.

The transform does not introduce an autodiff runtime or a gradient-specific backend opcode. It rebuilds the selected forward ancestor slice and expresses every adjoint with ordinary verified tensor operations, then verifies the resulting module again. The transformed module therefore enters the existing reference, Buffer/Loop IR, generated-C, native, cache, and verification paths like any hand-written module.

## Supported first-phase contract

- The selected return value must be a static scalar `f32` or `f64` tensor.
- `wrt` identifies one or more runtime-input indices; gradients are returned in exactly that order.
- Every value on the backward-reachable slice must use the same exact floating dtype as the selected output.
- Supported backward operations are `add`, `mul`, `sum`, and `reshape`, with `input` and `const` as leaves.
- Runtime inputs not on the selected loss slice are retained in the transformed module's input ABI. A requested but unused input receives an exact zero tensor gradient.
- Multiple reverse paths accumulate through ordinary tensor `add` operations.

## Broadcasting and reductions

Elementwise broadcasting is differentiated structurally. A contribution whose forward operand was broadcast is reduced across every inserted leading axis and every expanded size-one axis, then reshaped back to the exact operand type. This uses ordinary tensor `sum` and `reshape` operations rather than host-side NumPy precomputation.

The adjoint of `sum` reconstructs reduced singleton axes as needed and broadcasts the upstream gradient back to the source shape by multiplying with a typed all-ones constant. The adjoint of `reshape` reshapes the upstream gradient back to the source shape.

These rules preserve the compiler's existing row-major reshape and structural broadcasting semantics instead of introducing separate autodiff indexing rules.

## Fail-closed boundaries

The first phase intentionally rejects rather than guesses when a correct source transform would require semantics that the current bounded contract does not provide:

- non-scalar selected outputs;
- symbolic/dynamic shapes;
- integer gradients;
- mixed `f32`/`f64` backward slices, because there is no explicit cast primitive in this phase;
- ReLU, `prod`, `argmax`, transpose/strided views, mutation/write effects, matmul, and other unsupported backward operations;
- invalid or duplicate `wrt` indices and invalid return selection.

These are not claims that the operations are mathematically non-differentiable. They are explicit compiler capability boundaries. Future coverage should be added only with an executable, verifier-backed VJP/JVP rule rather than by silently approximating or materializing gradients outside the IR.

## Verification evidence

Focused regressions cover closed-form broadcast gradients, multi-path accumulation, reduction/reshape gradients compiled and executed natively, zero gradients for unused requested inputs, and deterministic rejection of unsupported or mixed-precision slices.

The production candidate before this document passed the full Ubuntu/Windows × Python 3.11/3.13 CI matrix. Ubuntu Python 3.11 reported Ruff success and 650 passing tests. No performance claim is attached to autodiff; this milestone establishes transformation correctness and backend reuse.

## Next architectural frontier

This phase establishes program-to-program differentiation. The next useful autodiff milestone should increase semantic depth rather than enumerate trivial operator rules: candidates include a verifier-backed finite-difference/gradient-consistency corpus, compositional differentiation through a stabilized matmul lowering, or higher-order differentiation once the transformed IR itself is deliberately accepted as an input contract.