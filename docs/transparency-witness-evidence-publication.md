# Transparency witness evidence publication

The evidence publication layer moves one already signed transparency-witness observation, plus the RFC 6962 consistency proofs a recipient may need, between clients without trusting the transport itself.

A publication is a canonical JSON envelope signed by a caller-pinned Ed25519 publication key. The envelope binds the publication signer, transparency-log identity, witness-policy identity, exact encoded witness observation, and an ordered consistency-proof map. The publication signature authenticates that envelope and proof set; it does **not** replace the transparency log signature, witness signature, or caller-supplied witness policy.

## Publication and verification

`create_transparency_witness_evidence_publication(...)` first verifies the supplied witness observation under the caller's log public key and witness policy. It then canonicalizes the proof map and signs the complete envelope with a domain-separated publication signature.

`verify_transparency_witness_evidence_publication(...)` independently checks:

- canonical publication framing and schema;
- the caller-pinned publication key against the encoded publisher identity;
- the exact witness-policy identity;
- the embedded witness observation, including its witness signature and signed transparency checkpoint;
- the transparency-log identity derived from that checkpoint;
- proof-node framing and deterministic proof ordering;
- the publication signature over the complete envelope.

The publication signer is therefore an authenticated distributor of specific evidence bytes. It is not promoted into a transparency-log root, witness-policy authority, or freshness oracle.

## Untrusted HTTP transport

`fetch_transparency_witness_evidence_publication(...)` deliberately returns raw, untrusted bytes. Authentication happens only when the caller verifies or accepts those bytes.

The fetcher is intentionally bounded:

- HTTPS is the default transport requirement;
- plain HTTP requires explicit `allow_insecure_http=True`, which is primarily useful for controlled/local deployments and tests;
- URLs must be absolute HTTP(S) URLs and may not embed credentials, query strings, or fragments;
- redirects are rejected rather than silently changing the publication origin;
- the response media type must be `application/vnd.tiny-tensor-compiler.transparency-witness-publication+json`;
- `Content-Length`, when present, must be well formed and fit the configured transfer limit;
- streamed bytes are independently capped even if the server omits or lies about `Content-Length`;
- truncated bodies and empty responses fail closed.

TLS can protect the network hop, but publication authenticity does not depend on trusting HTTP or HTTPS response contents. A transport attacker who changes the envelope, observation, or proof set cannot produce the pinned publication signature.

## Durable acceptance

`accept_transparency_witness_evidence_publication(...)` does not create a second state-transition policy. After verifying the publication, it checks that the destination `TransparencyWitnessEvidenceStore` is pinned to the same log and witness policy, then delegates the mutation to `TransparencyWitnessEvidenceStore.record(...)`.

The existing durable-store rules therefore remain authoritative:

- per-witness rollback refusal;
- exact RFC 6962 consistency-proof requirements against remembered checkpoint digests;
- fail-before-mutation behavior for missing, extra, malformed, or invalid proofs;
- same-size/different-root terminal fork evidence;
- refusal to overwrite or heal already stored fork evidence;
- cross-process locking, canonical persistence, re-verification, and atomic replacement.

A correctly signed old publication cannot roll a client back after that client has remembered a newer witness view. Likewise, a publication containing valid signed same-size conflicting views causes the same terminal fork state as locally supplied evidence.

## Bounded scope

Each publication carries one witness observation and zero or more consistency proofs. This phase intentionally does not define batching, subscription, gossip, peer discovery, automatic consistency-proof retrieval, witness discovery, background refresh, or a publication server/daemon.

It also does not establish:

- global checkpoint freshness;
- trusted wall-clock time or maximum checkpoint age;
- proof that all clients have fetched the same publication;
- universal split-view prevention;
- availability of any publisher or witness;
- automatic selection of the globally newest checkpoint.

Those require independent freshness signals or broader cross-client communication protocols. This layer establishes a smaller, executable property: evidence produced in one process can be transported as untrusted bytes, authenticated by another process, and admitted into that client's existing durable consistency state without weakening its rollback, fork, or append-only checks.

## Executable evidence

The regression suite uses a real loopback `ThreadingHTTPServer`, not a mocked URL fetch. It exercises authenticated fetch-and-accept, wrong publisher keys, tampered publications, redirect refusal, wrong media types, truncation, transfer limits, missing and damaged consistency proofs, durable rollback refusal, and fetched same-size fork evidence. The same suite runs on the repository's Ubuntu and Windows CI matrix.
