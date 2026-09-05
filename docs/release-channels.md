# Rollback-protected release channels

The content-addressed bundle registry and publisher-attestation APIs establish two separate facts before compiler-free execution:

1. the downloaded archive bytes match the requested SHA-256 digest and pass the native-bundle archive verifier; and
2. a caller-pinned Ed25519 publisher authorized that exact digest.

A valid signature over an old digest is still valid, so those layers alone do not provide freshness. The optional release-channel APIs add a deliberately bounded third property: **once a caller has successfully accepted and persisted a publisher/channel sequence, that caller will not later accept a lower signed sequence from the same publisher/channel through the same protected state store.**

## Signed channel head

A release head is canonical JSON containing:

- schema `ttc-ed25519-release-channel-v1`;
- the canonical `ed25519:<sha256-public-key>` publisher id;
- a normalized channel name such as `stable`;
- a non-negative monotonically increasing integer sequence; and
- the exact content-addressed `sha256:<64 lowercase hex>` archive digest.

The publisher signs a domain-separated canonical message with the same raw Ed25519 key model used by archive attestations. `verify_release_checkpoint()` verifies canonical encoding, pinned publisher authorization/revocation, the expected channel, and the detached signature before the checkpoint can participate in rollback state.

The sequence is publisher policy metadata. It is not wall-clock time and it is not inferred from registry ordering.

## Publisher update protocol

`publish_release_channel()` first verifies any existing signed head for that publisher/channel. A lower requested sequence is rejected, and reusing an existing sequence for a different archive digest is rejected. Publishing the exact same sequence and digest is idempotent.

The archive remains an immutable content-addressed object and is still published through the existing attested-archive path. Only after that succeeds does the publisher update the mutable channel head.

Channel updates use HTTP conditional requests:

- the first head uses `If-None-Match: *`;
- later heads use the `ETag` observed with the previously verified head in `If-Match`.

A conflicting `409`/`412` fails the publication and requires the caller to refetch and retry. This prevents cooperating/concurrent writers that observed different heads from silently overwriting one another through the client protocol. It is **not** a claim that a malicious registry provides globally consistent views to all clients.

## Caller rollback state

`ReleaseStateStore` is caller-owned persistent policy state keyed by `(publisher_id, channel)`. Each entry stores the highest accepted sequence and its archive digest.

Updates execute as one cross-process critical section:

1. acquire an operating-system file lock;
2. read and validate canonical state;
3. compare the candidate checkpoint with the current floor;
4. reject a lower sequence;
5. reject the same sequence bound to a different digest;
6. write canonical state to a private temporary file, flush it, and atomically replace the old file.

The implementation reuses the repository's existing POSIX/Windows file-lock primitive. A regression starts independent spawned processes that race sequence 10 and 11 updates and proves the persisted floor cannot finish at 10.

The state file is intentionally not self-authenticating. A caller relying on rollback protection must protect this local policy state from hostile deletion or modification using the surrounding operating-system/application trust boundary.

## Fetch and execution order

`fetch_release_channel_archive()` fails closed in this order:

1. validate the requested publisher against the caller's `PublisherTrustPolicy`;
2. fetch the signed channel head;
3. verify canonical metadata, expected publisher/channel, and Ed25519 signature;
4. compare the checkpoint with the local rollback floor **before any archive or archive-attestation request**;
5. stage and fetch the exact digest through the existing publisher-attested archive path;
6. recheck and record the checkpoint under the cross-process rollback lock, because another process may have accepted a newer sequence while the archive was downloading;
7. atomically publish the staged archive to the caller destination.

`load_release_channel_registry()` then loads the already verified archive through the existing compiler-free bundle-set executable. Release metadata does not bypass archive SHA-256 verification, archive structure verification, target/ABI checks, or publisher attestation.

## Security boundary

This phase proves a narrow rollback property. It does **not** claim:

- freshness on a caller's first observation of a channel;
- recovery if the caller's persisted rollback state is deleted or maliciously modified;
- registry availability or globally consistent views;
- trusted timestamps or clock-based expiry;
- transparency logs or gossip;
- TUF-style delegated roles, threshold signatures, or root-key rotation;
- PKI/organizational identity beyond caller-pinned Ed25519 keys; or
- general software-supply-chain security.

A registry can replay a previously valid signed checkpoint to a brand-new caller that has no prior state. Preventing that requires an external freshness or consistency mechanism and is intentionally outside this milestone.

## Executable evidence

The regression suite covers:

- deterministic canonical signed release metadata and domain/channel binding;
- signature tampering and wrong-channel rejection;
- persistent rollback and same-sequence/different-digest rejection;
- POSIX and Windows cross-process floor monotonicity;
- real loopback HTTP publication using conditional channel updates;
- two successive attested native bundle releases;
- compiler-free execution of the accepted current release; and
- replay of the older still-valid signed release head, rejected before any archive request.

These are correctness/security-semantics tests. They are not a performance, global freshness, or availability benchmark.
