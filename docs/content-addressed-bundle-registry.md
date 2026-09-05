# Content-addressed bundle registry transport

The registry layer distributes an already-verified deterministic `native-bundle-archive-v1` object through a deliberately small HTTP(S) contract. It does not compile tensor IR, reinterpret bundle manifests, or create a second native execution path.

Its identity is the exact archive bytes:

```text
sha256:<64 lowercase hex digits>
```

The caller pins that digest before a remote object is trusted. The downloaded bytes must match the pin and must independently pass the existing archive, bundle-set, child library, target, and ABI verification before execution.

## API

```python
from tiny_tensor_compiler import (
    digest_dynamic_bundle_set_archive,
    fetch_dynamic_bundle_set_archive,
    load_dynamic_bundle_set_registry,
    publish_dynamic_bundle_set_archive,
)

expected = digest_dynamic_bundle_set_archive("family.ttca")

published = publish_dynamic_bundle_set_archive(
    "family.ttca",
    "https://registry.example",
    token="registry-token",
)
assert published == expected

fetch_dynamic_bundle_set_archive(
    "https://registry.example",
    expected,
    "downloaded.ttca",
    token="registry-token",
)

executable = load_dynamic_bundle_set_registry(
    "https://registry.example",
    expected,
    token="registry-token",
)
try:
    result = executable(inputs=[runtime_input])
finally:
    executable.close()
```

The canonical object URL is:

```text
<registry>/v1/archives/sha256/<64 lowercase hex digits>
```

## Publication contract

`publish_dynamic_bundle_set_archive()` validates the local archive with the existing archive loader before uploading it. The upload is addressed by the local SHA-256 digest and uses `If-None-Match: *` so the protocol is naturally immutable rather than a mutable tag endpoint.

A successful HTTP response is not accepted as proof of publication. After PUT, the client GETs the content-addressed object back, verifies the exact SHA-256 bytes, and runs the existing full archive verification over those remote bytes. An HTTP 409/412 is accepted only as a possible idempotent pre-existing object and receives the same read-back verification.

Therefore a registry that acknowledges an upload but stores different bytes does not produce a successful publication result.

## Fetch contract

`fetch_dynamic_bundle_set_archive()` requires an expected canonical digest from the caller. It:

1. streams the object into a private sibling temporary file;
2. enforces a configurable maximum byte count while reading;
3. validates `Content-Length` when the server supplies it;
4. checks the exact SHA-256 digest;
5. passes the temporary archive through the existing complete archive/bundle/ABI verifier;
6. atomically publishes the verified temporary file at the requested local destination.

A failed transfer or verification removes the temporary artifact and never replaces an existing destination.

`load_dynamic_bundle_set_registry()` applies the same download and verification boundary, then owns the private downloaded archive and extracted archive payload for the lifetime of the returned executable. `close()` closes the underlying archive/bundle executables before removing the private download tree.

## Redirect and credential boundary

The client does not follow HTTP redirects. This is intentional: automatically following a redirect can silently move a Bearer credential or an expected immutable object lookup to a different origin.

An optional token is sent as:

```text
Authorization: Bearer <token>
```

Tokens are HTTP credentials only. They are not publisher signatures and are never written into manifests, archive bytes, cache identities, or content addresses.

Registry URLs may not embed credentials or contain query/fragment components. The production default requires HTTPS. Plain HTTP requires the explicit `allow_insecure_http=True` escape hatch, which exists so local/controlled development registries and the loopback regression server can exercise the real protocol without claiming transport security.

## Fail-closed cases

The client rejects or fails publication/fetch on:

- a noncanonical digest;
- HTTP authentication or missing-object errors;
- redirects;
- a declared or actual object larger than the configured transfer limit;
- malformed or mismatched `Content-Length`;
- truncated transport responses;
- any downloaded byte sequence whose SHA-256 differs from the caller pin;
- a remotely substituted object after a nominally successful PUT;
- any archive, child bundle, native library, target, or ABI inconsistency detected by the existing archive verifier;
- an existing local fetch destination.

The registry transport does not weaken the archive loader's path, ZIP-entry, child-library, manifest, target, or ABI checks.

## Trust boundary

Content addressing answers one narrow question: *did the client receive the exact bytes identified by the caller's digest?*

It does **not** answer who created or approved those bytes. This phase does not add:

- publisher public keys or signatures;
- certificate-chain or transparency-log semantics beyond the normal HTTPS transport stack;
- trusted release names/tags;
- remote registry server attestation;
- key rotation/revocation policy;
- cross-target portability.

A malicious party that can convince the caller to trust a different SHA-256 digest can still direct the caller to a different internally coherent archive. Publisher authenticity therefore remains a separate trust-layer milestone.

## Evidence scope

Regression coverage uses a real loopback `ThreadingHTTPServer`, not mocked `urlopen()` calls. It exercises PUT/GET, Bearer credentials, immutable 412 publication, post-upload read-back, digest substitution, transfer limits, redirect refusal without credential forwarding, 401/404 errors, truncated responses, compiler-free remote load, finite symbolic dispatch, and preallocated multi-output native execution.

The same suite runs on Ubuntu and Windows so the remotely loaded archive still reaches the existing GCC-style/MSVC child native execution paths.

No network throughput, registry scalability, CDN behavior, TLS hardening, or runtime-performance claim is inferred from CI timing.

## Next promotion

This closes the first controlled content-addressed distribution phase. Adding more HTTP verbs, mutable tags, or alternate checksum spellings would be protocol farming.

The next deployment/trust milestone should establish genuine publisher authenticity/provenance using a standard, reviewable cryptographic mechanism with explicit key/trust/revocation semantics and cross-platform evidence. If that cannot be done without introducing an unverified crypto stack, the project should promote on another independent executable frontier instead.
