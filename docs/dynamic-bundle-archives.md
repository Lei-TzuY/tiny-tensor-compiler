# Deterministic dynamic bundle-set archives

The archive transport wraps one already-verified finite `native-bundle-set-v1` directory in a single deterministic file without weakening any child bundle, ABI, target, or runtime-shape checks.

It is a transport boundary, not a new executable backend and not a package-authenticity system.

## API

```python
from tiny_tensor_compiler import (
    load_dynamic_bundle_set_archive,
    pack_dynamic_bundle_set_archive,
)

pack_dynamic_bundle_set_archive("family.ttcset", "family.ttca")
executable = load_dynamic_bundle_set_archive("family.ttca")
try:
    result = executable(inputs=[runtime_input])
finally:
    executable.close()
```

Packing still happens after compilation: `pack_dynamic_bundle_set_archive()` never invokes a compiler. Loading and dispatch are likewise compiler-free.

Verified archive bytes may also be distributed through the separate content-addressed registry layer documented in `content-addressed-bundle-registry.md`. That layer pins the exact archive SHA-256, streams/downloads with explicit transport bounds, and then reuses this archive loader unchanged for complete payload verification.

## Transport schema

`native-bundle-archive-v1` is a ZIP container with a deliberately narrow profile:

- top-level `archive.json` declares schema `native-bundle-archive-v1`, payload kind `dynamic-bundle-set`, and payload root `bundle`;
- every package byte lives under `bundle/` and preserves the verified bundle-set directory-relative path;
- entries are emitted in canonical sorted order after `archive.json`;
- every entry uses `ZIP_STORED`, a fixed DOS timestamp, fixed regular-file mode, and no encryption;
- the same verified source bundle tree therefore produces byte-identical archive bytes on repeated packing.

The archive remains platform-specific because its child native bundles remain platform-specific. Deterministic transport does not make object code portable across operating systems or machine architectures.

## Fail-closed packing

Before writing an archive, the packer fully verifies the source bundle set. Unlike ordinary runtime dispatch, which intentionally loads child variants lazily, archive publication forces each packaged binding through the existing child loader once so every child source/library SHA-256, ABI digest, embedded ABI identity, and target contract is checked.

The archive is assembled in a temporary sibling file. Before the final rename, the exact temporary archive bytes are loaded through the archive loader and fully revalidated again. This second check closes the source-validation/read time-of-check/time-of-use window: a source mutation during packing cannot publish an archive that only fails when a later consumer opens it.

An existing destination is rejected, and failed validation removes the temporary file instead of exposing a partial package.

## Safe extraction

The loader does not call `ZipFile.extractall()`. It validates the complete central-directory entry set first, then copies accepted regular files into a fresh private temporary directory.

The loader rejects:

- missing or unsupported `archive.json` schemas or payload kinds;
- absolute paths, `..` path components, noncanonical backslash paths, or entries outside `bundle/`;
- duplicate entry names;
- directory entries and symbolic-link entries;
- encrypted entries;
- compression methods other than `ZIP_STORED`;
- an empty payload.

After extraction, the loader fully verifies every packaged child through the existing bundle-set and concrete-bundle contracts before returning an executable. It then constructs a fresh `NativeBundleSetExecutable`, so normal runtime child loading remains lazy even though the transport itself was checked eagerly.

`NativeBundleSetArchiveExecutable.close()` first closes all loaded child executables and then removes the private extracted payload tree. The original archive remains caller-owned and untouched.

## Integrity and trust boundary

Archive validation composes existing internal consistency checks:

- bundle-set manifest shape/binding/child-manifest verification;
- concrete child source and shared-library SHA-256;
- canonical ordered ABI SHA-256;
- ABI identity exported by the loaded shared library;
- current target identity.

These checks detect corruption, partial replacement, path-confusion attempts, and incoherent package substitution. They do **not** authenticate who produced the archive. There is no signature, trusted publisher key, certificate chain, transparency log, or publisher-trust policy in this phase. A party able to replace an entire coherent archive can still construct another internally consistent package.

The content-addressed registry layer narrows remote substitution by requiring a caller-pinned SHA-256 for the exact archive bytes and by reusing this full verifier after download. It still does not establish publisher identity: a caller who is persuaded to trust a different digest can be directed to a different coherent archive.

## Evidence scope

Regression coverage exercises deterministic byte-for-byte repacking, compiler-free dispatch and explicit specialization, preallocated multi-output execution, child-library tamper detection, source-corruption refusal before publication, path traversal, duplicate-name and symlink rejection, unsupported transport schema rejection, destination collision handling, and full child validation on both GCC-style and MSVC CI paths.

No compression-ratio, deployment-size, network-transfer, or runtime-performance claim is inferred from CI timing.

## Next promotion

The local single-file transport and the first controlled content-addressed HTTP(S) distribution boundary are now separate completed layers. Further ZIP metadata/compression variants or mutable registry naming would be low-value format farming.

A later trust phase should add genuine publisher authenticity/provenance using a standard reviewable mechanism with explicit key/trust/revocation semantics. Until such a mechanism can be validated cross-platform without inventing a bespoke cryptographic stack, other independent compiler/runtime frontiers remain preferable to fake signing claims.
