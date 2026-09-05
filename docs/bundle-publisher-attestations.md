# Ed25519 bundle publisher attestations

The optional publisher-attestation layer authenticates one exact content-addressed native bundle archive to a caller-pinned Ed25519 publisher key. It composes with the existing archive and registry verifiers instead of replacing them.

The trust chain is deliberately split into two independent questions:

1. **Byte integrity:** the content-addressed registry requires the caller's exact `sha256:<digest>` and the downloaded archive must pass the complete archive/bundle/ABI verifier.
2. **Publisher authorization:** a detached Ed25519 attestation must authorize that exact archive digest and verify under a public key already pinned by the caller's `PublisherTrustPolicy`.

A valid signature never bypasses the archive verifier, and a valid archive digest never counts as a publisher signature.

## API

```python
from tiny_tensor_compiler import (
    PublisherTrustPolicy,
    load_attested_dynamic_bundle_set_registry,
    publish_attested_dynamic_bundle_set_archive,
    publisher_public_key_from_private_key,
)

# Private signing material remains publisher-controlled. This example assumes
# a 32-byte raw Ed25519 private seed was obtained from a secure key source.
public_key = publisher_public_key_from_private_key(private_seed)
policy = PublisherTrustPolicy((public_key,))

digest, publisher_id = publish_attested_dynamic_bundle_set_archive(
    "family.ttca",
    "https://registry.example",
    private_seed,
    token="registry-token",
)

executable = load_attested_dynamic_bundle_set_registry(
    "https://registry.example",
    digest,
    publisher_id,
    policy,
    token="registry-token",
)
try:
    result = executable(inputs=[runtime_input])
finally:
    executable.close()
```

The unsigned `publish_dynamic_bundle_set_archive()`, `fetch_dynamic_bundle_set_archive()`, and `load_dynamic_bundle_set_registry()` APIs are unchanged. Publisher authentication is an explicit opt-in rather than a silent change to the historical content-addressed transport contract.

## Key and publisher identity

The cryptographic primitive is Ed25519 from the `cryptography` package. The project does not implement an elliptic-curve or signature primitive itself.

Signing accepts one raw 32-byte Ed25519 private seed. Trust policies contain one or more raw 32-byte Ed25519 public keys. A stable publisher identifier is derived as:

```text
ed25519:<sha256(raw-public-key-bytes)>
```

The publisher identifier is a key fingerprint and routing identity; it is not a certificate, organization name, or proof of real-world identity.

`PublisherTrustPolicy` is caller-owned. A publisher is accepted only when its exact public key is already pinned in the policy. The policy may also name locally revoked publisher identifiers. Unknown or revoked publishers fail before registry network access begins.

## Detached attestation schema

The canonical detached envelope is ASCII JSON with sorted compact keys and one trailing newline:

```json
{"archive_digest":"sha256:<64 lowercase hex>","publisher_id":"ed25519:<64 lowercase hex>","schema":"ttc-ed25519-archive-attestation-v1","signature":"<128 lowercase hex>"}
```

The Ed25519 signature covers a domain-separated message containing the exact archive digest and publisher identifier:

```text
tiny-tensor-compiler\0native-bundle-archive-attestation-v1\0
+ archive_digest
+ \0
+ publisher_id
```

This prevents the signature from being reinterpreted as authorization for an unrelated application message. The envelope is bounded to 16 KiB and must use the exact canonical schema and JSON encoding before signature verification is attempted.

## Registry object contract

Detached attestations are stored immutably at:

```text
<registry>/v1/attestations/ed25519/<publisher-key-sha256>/<archive-sha256>
```

`publish_attested_dynamic_bundle_set_archive()` first uses the existing content-addressed archive publication path. It then PUTs the detached attestation with `If-None-Match: *`. An HTTP 409/412 is treated only as a possible idempotent pre-existing object. In every case the client GETs the stored attestation back and verifies the exact digest binding, publisher identity, pinned key, revocation policy, and Ed25519 signature before reporting publication success.

A registry that acknowledges the PUT but returns another publisher's valid signature therefore cannot produce a successful publication result.

## Fail-closed fetch and load

`fetch_attested_dynamic_bundle_set_archive()` refuses unknown or revoked publisher identities before opening a network connection. For a trusted publisher it:

1. creates a private sibling staging directory;
2. uses the existing content-addressed fetcher to download and fully verify the archive into staging;
3. downloads the bounded detached attestation through the same no-redirect/Bearer-token transport policy;
4. verifies exact archive digest, requested publisher identity, caller-pinned key, local revocation, and Ed25519 signature;
5. only then atomically publishes the staged archive at the caller's destination.

Failed trust or signature verification removes staging and does not publish the caller destination. A concurrently created destination is never removed by failure cleanup.

`load_attested_dynamic_bundle_set_registry()` applies the same trust boundary and then delegates to the existing compiler-free archive executable. The attestation layer does not introduce a compiler call or a second native execution path.

## Transport credentials are separate

HTTPS and the optional Bearer token protect the network transport/account boundary. They are not publisher signatures. Tokens are not included in the signed message, archive content address, manifests, or publisher identifier.

The existing redirect refusal remains in effect so credentials are not automatically forwarded to a different origin.

## What this protects

Assuming the caller's pinned public key is correct and the corresponding private signing key remains uncompromised, a registry or network party cannot replace the requested archive with another coherent archive and still satisfy both the pinned SHA-256 and publisher authorization checks.

Local revocation can immediately cause one previously pinned publisher key to fail closed for that caller.

## What this does not claim

This phase deliberately does **not** provide:

- freshness or rollback protection for an older correctly signed archive;
- trusted release names, versions, channels, or mutable tags;
- certificate-chain or organizational-identity semantics;
- remote/global revocation distribution or a key-rotation protocol;
- transparency logs or inclusion proofs;
- trusted timestamps or signing-time validation;
- threshold or multi-party signatures;
- TUF, Sigstore, in-toto, PKI, or hardware-backed key custody;
- secure generation/storage/backup of publisher private keys;
- protection after a trusted publisher private key is compromised;
- cross-target portability of native artifacts.

Those are separate supply-chain/trust problems and must not be inferred from a valid Ed25519 signature.

## Evidence scope

Regression coverage exercises deterministic canonical attestations, modified-signature and wrong-digest refusal, untrusted and revoked publisher rejection, invalid key/envelope handling, immutable attestation publication, post-PUT read-back verification, Bearer-authenticated real loopback HTTP transport, destination-not-published failure behavior, and compiler-free execution of a signed finite symbolic bundle family.

The full suite runs on Ubuntu and Windows with Python 3.11 and 3.13. The production candidate CI also installs and executes the standard `cryptography` Ed25519 implementation on each platform.

No cryptographic-strength proof, key-management certification, network-security audit, or runtime-performance claim is inferred from CI timing.

## Phase boundary

This closes the first caller-pinned publisher-authorization layer. Adding alternate encodings, signature algorithms, or key spellings without a new trust property would be format farming.

Further deployment-security work should only proceed when it adds a separately testable trust property such as standardized freshness/rollback metadata or externally verifiable transparency. Otherwise the project should promote on an independent compiler/runtime frontier rather than accumulating home-grown supply-chain protocol surface.
