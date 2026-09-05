# Native ABI identity handshake

The ordinary native runtime verifies that a loaded shared library declares the exact concrete tensor ABI that the caller expects before `tiny_tensor_run` is invoked.

## Identity

The ABI identity is SHA-256 over canonical JSON containing the ordered input and output tensor types:

```json
{"inputs":[{"dtype":"i32","shape":[2,3]}],"outputs":[{"dtype":"i32","shape":[2,3]}]}
```

Object keys are sorted and JSON separators are compact. Input/output order, concrete shape, and dtype therefore contribute to identity. Internal storage layouts, compiler flags, and implementation details are not caller ABI fields; they remain covered by existing source/cache identities rather than this tensor ABI hash.

This canonical tensor-type encoding matches the ABI identity already used by native bundles. Ordinary native artifacts use the generic `tiny_tensor_abi_sha256` export, while bundle artifacts retain their bundle-specific export and trust/release policy.

## Build-source boundary

Public `generate_c()` output is unchanged. The native runtime appends the ABI-reporting export only to the C source actually sent to the platform compiler. This preserves the historical byte-for-byte public C-generation compatibility contract while still making every ordinary runtime-compiled artifact self-identifying.

The runtime loads the resulting `.so`, `.dylib`, or `.dll`, calls `tiny_tensor_abi_sha256()`, and compares the returned digest with the ABI derived from the verified `LoopProgram` before binding or invoking `tiny_tensor_run`.

The same handshake is used by:

- eager `compile_native()` and one-shot `execute_native()`;
- reusable native executables after process-cache eviction/reload;
- persistent native-cache artifacts;
- opt-in OpenMP native compilation, including process-pinned Windows artifacts.

## Persistent-cache failure handling

Persistent cache manifests already verify artifact file integrity with a library SHA-256. That does not prove the library has the ABI expected by the current caller.

If a staged persistent artifact has a valid manifest and matching file digest but reports the wrong tensor ABI, the runtime closes the staged copy, invalidates the persistent entry, recompiles once from the expected build source, and verifies the rebuilt artifact before execution. A second mismatch fails closed.

A freshly compiled process-local artifact with the wrong embedded ABI fails immediately; recompiling identical source would not repair that invariant violation.

## Evidence boundary

This handshake is an accidental/wrong-artifact correctness check. It is **not** publisher authentication, signature verification, provenance attestation, tamper resistance, or a general security guarantee. Native-bundle publisher trust, threshold authorization, release channels, and attestations remain separate layers.

It is also not a performance claim. CI validates deterministic identity, real GCC/MSVC shared-library loading, fail-closed mismatch behavior, persistent-cache rebuild behavior, and ordinary/OpenMP execution paths.
