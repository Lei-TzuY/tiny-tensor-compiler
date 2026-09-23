# Deterministic native bundle-set archives

The archive transport wraps either one already-verified finite `native-bundle-set-v1` directory or one verified `native-linearization-bundle-set-v1` retained-linearization family in a single deterministic file without weakening child bundle, ABI, target, retained-state, or runtime-shape checks.

It is a transport boundary, not a new executable backend and not by itself a package-authenticity system.

## API

```python
from tiny_tensor_compiler import (
    load_dynamic_bundle_set_archive,
    load_dynamic_linearization_bundle_set_archive,
    pack_dynamic_bundle_set_archive,
    pack_dynamic_linearization_bundle_set_archive,
)

pack_dynamic_bundle_set_archive("family.ttcset", "family.ttca")
executable = load_dynamic_bundle_set_archive("family.ttca")

pack_dynamic_linearization_bundle_set_archive(
    "linearizations.ttclin",
    "linearizations.ttcla",
)
linearizations = load_dynamic_linearization_bundle_set_archive(
    "linearizations.ttcla"
)
```

Packing still happens after compilation: neither archive packer invokes a compiler. Loading, finite dispatch, and retained-linearization queries are likewise compiler-free.

The content-addressed registry now dispatches both supported archive payload kinds through one digest-pinned HTTP/staging path while retaining payload-specific archive verification. Ordinary archives load through the `dynamic-bundle-set` verifier; retained-linearization archives load through the coordinated retained-state verifier and return compiler-free linearization executables. Publisher attestation, release-channel, threshold, and transparency APIs still target the ordinary payload and remain separate later trust-layer promotions for retained linearizations.

## Transport schema

`native-bundle-archive-v1` is a ZIP container with a deliberately narrow profile:

- top-level `archive.json` declares schema `native-bundle-archive-v1`, payload root `bundle`, and exactly one supported payload kind: `dynamic-bundle-set` or `retained-linearization-bundle-set`;
- every package byte lives under `bundle/` and preserves the verified bundle-set directory-relative path;
- entries are emitted in canonical sorted order after `archive.json`;
- every entry uses `ZIP_STORED`, a fixed DOS timestamp, fixed regular-file mode, and no encryption;
- the same verified source bundle tree therefore produces byte-identical archive bytes on repeated packing.

The archive remains platform-specific because its child native bundles remain platform-specific. Deterministic transport does not make object code portable across operating systems or machine architectures.

## Fail-closed packing

Before writing an archive, the selected packer fully verifies the source bundle set. Ordinary bundle sets force each packaged binding through the existing child loader once. Retained-linearization sets likewise force every binding through the coordinated primal/tape, pushforward, and pullback component loader, so child source/library SHA-256 values, ABI digests, embedded ABI identities, target identity, and the cross-component retained-state ABI are checked before publication.

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

After extraction, the loader fully verifies every packaged child through the payload kind's existing bundle-set contract before returning an executable. It then constructs a fresh dispatcher, so normal runtime child loading remains lazy even though the transport itself was checked eagerly.

`NativeBundleSetArchiveExecutable.close()` and `NativeLinearizationBundleSetArchiveExecutable.close()` first close all lazily loaded child/component executables and then remove the private extracted payload tree. The original archive remains caller-owned and untouched.

## Integrity and trust boundary

Archive validation composes existing internal consistency checks:

- bundle-set manifest shape/binding/child-manifest verification;
- retained-linearization component-role, child-manifest/ABI hash, and cross-component retained-state ABI verification;
- concrete child source and shared-library SHA-256;
- canonical ordered ABI SHA-256;
- ABI identity exported by the loaded shared library;
- current target identity.

These checks detect corruption, partial replacement, path-confusion attempts, and incoherent package substitution. The bare archive format still does **not** authenticate who produced the archive. A party able to replace an entire coherent archive can construct another internally consistent package.

The content-addressed registry narrows remote substitution by requiring a caller-pinned SHA-256 for the exact archive bytes and by reusing this full verifier after download. It still does not establish publisher identity on its own.

Publisher authorization is an optional layer above both contracts. `PublisherTrustPolicy` pins accepted Ed25519 public keys, and the attested registry APIs verify a detached authorization over the exact archive digest before a staged download can become a caller-visible destination or executable. See `bundle-publisher-attestations.md` for its threat model and limitations.

This layering is intentional: archive consistency, content-addressed byte identity, and publisher authorization remain distinct properties with independent failure modes.

## Evidence scope

Regression coverage exercises deterministic byte-for-byte repacking, compiler-free ordinary dispatch and retained primal/pushforward/pullback execution, explicit specialization, preallocated multi-output execution, child-library tamper detection, wrong payload-kind rejection, source-corruption refusal before publication, path traversal, canonical-equivalent path alias rejection, duplicate-name and symlink rejection, unsupported transport schema rejection, destination collision handling, cleanup, and full child validation on both GCC-style and MSVC CI paths.

No compression-ratio, deployment-size, network-transfer, or runtime-performance claim is inferred from CI timing.

## Next promotion

Deterministic local single-file transport and unsigned content-addressed registry delivery now cover both ordinary finite bundle sets and retained-linearization bundle sets under one narrow archive schema, safe extraction boundary, and shared digest-pinned transfer path. Further ZIP metadata/compression variants or a second retained-specific HTTP protocol would be low-value format/transport farming.

The next retained-linearization deployment milestone is publisher authorization parity: reuse the existing Ed25519 digest attestation and pinned trust policy while staging and loading through the retained archive verifier. Release-channel rollback/freshness, threshold policy, and transparency should remain later layers until this publisher-authenticated retained path is executable and fail-closed.
