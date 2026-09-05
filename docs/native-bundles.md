# Verified native AOT bundles

The native-bundle surface turns one already-verified, concrete `LoopProgram` into a reusable platform-specific deployment artifact. It is intentionally separate from the internal persistent compilation cache: cache entries are an implementation detail keyed by compiler/source identity, while a bundle is an explicitly exported directory with a stable first schema and a self-described call ABI.

## API

```python
from tiny_tensor_compiler.native_bundle import compile_native_bundle, load_native_bundle

compile_native_bundle(program, "model.ttcbundle")
executable = load_native_bundle("model.ttcbundle")
try:
    result = executable(inputs=[runtime_input])
finally:
    executable.close()
```

`load_native_bundle()` does not look up or invoke a compiler and does not require the original `Module`, `CPUProgram`, `LoopProgram`, or generated source to be reconstructed. The bundle contains the generated source only as immutable provenance/integrity material alongside the compiled library.

## Bundle layout

A `native-bundle-v1` directory contains exactly the artifacts needed by the loader contract:

- `manifest.json`: schema, current target identity, ordered input/output tensor ABI, ABI digest, source/library names, and SHA-256 digests;
- `program.c`: the exact generated C source compiled for the bundle;
- the target shared library (`program.so`, `program.dylib`, or `program.dll`, according to the existing native backend).

The bundle is platform-specific. Target identity includes the Python process platform family, machine architecture, and pointer width. A bundle produced for another OS/architecture is rejected rather than treated as portable object code.

## Verification before execution

The loader performs all verification before configuring or invoking `tiny_tensor_run`:

1. require the exact `native-bundle-v1` manifest field set and current target identity;
2. require the expected source/library filenames;
3. verify SHA-256 for both source and shared library;
4. decode every ordered input/output `TensorType` and reject malformed/unsupported dtype or shape metadata;
5. recompute a canonical SHA-256 of the ordered input/output ABI and require it to match `manifest.json`;
6. load a private staged copy of the shared library and require its exported `tiny_tensor_bundle_abi_sha256()` identity to match the manifest ABI digest.

Step 6 binds a well-formed manifest ABI to the library that was actually compiled. Rewriting a valid manifest shape or dtype and recomputing only the manifest digest is therefore rejected before `tiny_tensor_run` is called.

These checks establish artifact consistency and corruption/tamper detection within the bundle contract. They are **not** a cryptographic authenticity or provenance claim: the format has no signature or trusted publisher key, and an actor able to replace the source, library, manifest, and embedded ABI identity together can construct another internally consistent bundle.

## Runtime contract

A loaded `NativeBundleExecutable` preserves the ordinary native ABI validation rules:

- runtime inputs must match the manifest's ordered shape/dtype contract;
- one output returns an `ndarray`, while multiple outputs return an ordered tuple;
- caller-provided `out=` arrays must match exact shape/dtype, be aligned, writable, and C-contiguous;
- outputs may not overlap runtime inputs or one another;
- `close()` unloads the private staged library copy and releases its staging directory.

The loader stages the shared library instead of loading it directly from the user-owned bundle directory, so normal executable lifetime does not require keeping the exported bundle itself locked by the process.

## Publication durability

`compile_native_bundle()` builds into a temporary sibling directory and publishes the completed directory with one final rename. An existing destination is rejected. Compiler failure, manifest failure, or any pre-publication exception removes the temporary build directory and never exposes a partially populated destination.

## Deliberate first-phase limits

`native-bundle-v1` accepts a concrete serial `LoopProgram`. It does not yet package unresolved symbolic/dynamic templates, `BorrowedLoopProgram` runtime alias contracts, OpenMP bundle mode, cross-platform binaries, signatures, or a fleet-oriented artifact registry.

Finite symbolic deployment families are handled by the separate `native-bundle-set-v1` layer. Those verified bundle-set directories can now be wrapped in the deterministic `native-bundle-archive-v1` single-file transport described in `docs/dynamic-bundle-archives.md`; the concrete child-bundle schema itself remains an ordinary directory and is unchanged by that transport layer.

These are separate milestones because each needs new executable metadata or lifecycle guarantees. In particular, authentication/signing must not be implied by the existing SHA-256 integrity fields, and archive transport does not make platform-specific native code cross-target portable.
