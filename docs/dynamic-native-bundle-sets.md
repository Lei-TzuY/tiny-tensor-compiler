# Finite dynamic native bundle sets

`native_bundle_set.py` packages an explicitly bounded family of runtime-symbolic tensor specializations into one compiler-free deployment directory.

This phase does **not** make native Buffer IR, Loop IR, generated C, or the native ABI symbolic. Each requested binding is specialized to one ordinary concrete tensor module, reverified, lowered through the existing concrete pipeline, and published as a child `native-bundle-v1` artifact. The outer `native-bundle-set-v1` manifest only describes and verifies how those concrete children map back to the original symbolic input contract.

## Build-time contract

`compile_dynamic_bundle_set(module, bindings, destination)` requires:

- a validated dynamic tensor `Module`;
- one or more explicit complete symbolic binding mappings;
- unique bindings;
- a unique concrete runtime input ABI for every binding;
- a destination that does not already exist.

For every requested binding the compiler:

1. freezes the symbolic module;
2. normalizes the complete binding with the existing symbolic engine;
3. specializes and reverifies a fully concrete module;
4. lowers through Buffer IR, memory planning, Loop IR, and elementwise fusion;
5. creates an ordinary verified serial child native bundle;
6. verifies that the child's manifest input ABI equals the concrete specialization;
7. records the child manifest SHA-256 and embedded ABI identity in the outer manifest.

The entire family is assembled in a sibling temporary directory and published with one final rename. A failed child build leaves no partially published outer package.

## Manifest model

The outer manifest records:

- schema `native-bundle-set-v1`;
- the current OS/platform/machine/pointer-width target identity;
- canonical sorted symbolic dimension names;
- the symbolic runtime input type template, including static, `SymbolicDim`, one-variable `AffineDim`, and canonical multi-symbol `LinearDim` extents;
- canonically ordered variants with exact bindings, child paths, child-manifest hashes, and child ABI hashes.

Loading is fail-closed. The loader rejects malformed symbolic terms, missing declared symbols, duplicate bindings, noncanonical ordering, ambiguous duplicate concrete input ABIs, wrong child paths, child-manifest substitution, binding-to-child-ABI mismatch, ABI-hash mismatch, and target mismatch.

The child bundle remains responsible for source/library content hashes and the ABI identity exported from the compiled shared library. The outer layer composes those existing checks rather than replacing them.

This is internal integrity and consistency validation, not cryptographic authenticity. A party capable of replacing an entire coherent package is outside this phase's trust model.

## Compiler-free runtime dispatch

`load_dynamic_bundle_set(path)` requires no tensor IR and performs no compiler lookup. It verifies the complete outer manifest before exposing a reusable `NativeBundleSetExecutable`.

Calling the executable:

1. derives the exact runtime input ABI from input count, shapes, and dtypes;
2. selects the unique packaged child whose concrete input ABI matches;
3. lazily loads that already-compiled child bundle;
4. executes through the existing `NativeBundleExecutable`, including ordered multi-output and caller-provided output buffers.

An input shape/dtype combination that was not explicitly packaged is an error. There is no lazy JIT fallback and no attempt to infer or compile a missing specialization at deployment time.

`specialize({...})` may also select a packaged child by its exact symbolic binding. Bindings not present in the package are rejected.

## Deliberate boundaries

This first bundle-set phase is intentionally finite and serial:

- no on-demand specialization compilation;
- no unresolved symbolic physical storage or loop bounds;
- no borrowed-input bundle ABI;
- no OpenMP bundle mode;
- no signed package/provenance claim;
- no registry, remote artifact transport, or cross-target package;
- no compatibility fallback for an unpackaged runtime shape.

The active writable-alias/storage work remains a separate subsystem. Bundle-set implementation is layered on existing verified concrete child bundles and does not modify storage-layout, view, input-binding, or mutability semantics.

## Evidence scope

Regression coverage includes compiler-free dispatch across multiple concrete symbolic bindings, lazy variant loading/reuse, explicit specialization selection, ordered multi-output with preallocated outputs, rejection of unpackaged runtime shapes, duplicate-binding and ambiguous-ABI refusal, binding-to-child-ABI verification, child-manifest substitution detection, malformed symbolic-template rejection, atomic outer publication, and destination-collision handling on both GCC-style and MSVC CI paths.

No runtime performance or deployment-size claim is made from CI timing.

## Next promotion

After this finite AOT-family layer is sealed, deployment work should move to a genuinely new boundary rather than adding more manifest fields: examples include an explicit provenance/authenticity model, controlled transport/registry semantics, or a separately verified parallel/AOT compatibility phase. Storage mutability remains owned by its existing parallel branch until that work converges.
