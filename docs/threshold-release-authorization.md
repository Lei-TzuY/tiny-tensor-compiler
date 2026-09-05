# Threshold release authorization

This phase adds an opt-in `k`-of-`n` Ed25519 authorization layer for native bundle release channels. It is independent of the historical single-publisher release-channel schema and does not weaken or silently upgrade existing callers.

## Policy identity

`ThresholdReleasePolicy` is defined by a caller-pinned set of distinct Ed25519 public keys and an integer threshold `k` with `2 <= k <= n`. Signers are ordered by their stable `ed25519:<sha256(public-key)>` identifiers. The remote policy identity is the SHA-256 hash of one canonical descriptor containing that ordered signer set and the threshold.

Local revocations are deliberately excluded from the policy identifier. Revocation is a caller-side acceptance rule: the same remote channel can therefore be evaluated by a stricter local policy without creating a different registry address. Policy construction fails when local revocations leave fewer eligible signers than the configured threshold.

## Checkpoint authorization

A threshold checkpoint binds exactly:

- the threshold policy identifier;
- one normalized channel name;
- one monotonic non-negative sequence;
- one exact `sha256:` bundle-set archive digest.

Every signer signs the same domain-separated canonical message. The encoded checkpoint stores a signer-id-sorted list of signatures. Signer identifiers must be unique canonical members of the pinned policy; duplicate or unknown signers fail closed. Verification requires at least `k` cryptographically valid signatures from currently non-revoked policy members.

The practical security property is deliberately narrow: assuming the caller's pinned policy and local trust state are protected, compromise of fewer than `k` eligible private keys is insufficient to authorize a new accepted checkpoint for that policy. This is not a claim that the archive build process, registry, operating system, signer machines, or network are otherwise uncompromised.

## Registry and rollback composition

Threshold publication reuses the existing immutable content-addressed archive registry. The mutable channel head lives below `v1/channels/threshold-ed25519/<policy-hash>/<channel>` and uses the same conditional ETag update discipline as the single-publisher release layer.

Fetch/load follows this order:

1. download the small mutable threshold checkpoint;
2. verify canonical encoding, exact policy identity, channel, and `k`-of-`n` signatures;
3. consult `ThresholdReleaseStateStore` and reject a locally known rollback before requesting the archive;
4. fetch the exact content-addressed archive and run the existing archive verifier;
5. reacquire the cross-process state lock, recheck the checkpoint against the latest local floor, and persist it atomically;
6. only then expose the downloaded archive or compiler-free executable.

The second state check closes the race where another process accepts a newer release while an older archive is still downloading. Same-sequence/different-digest checkpoints are treated as equivocation/conflict and rejected.

## Deliberate non-goals

This layer does **not** provide first-contact freshness. A new caller with no persisted floor can still be shown an older correctly threshold-signed checkpoint by a malicious registry. It also does not provide:

- trusted timestamps or expiry;
- transparency logs, gossip, or global registry consistency;
- TUF-style delegated roles, root metadata, or threshold key rotation/recovery;
- PKI or organizational identity for signer keys;
- remote revocation distribution;
- registry availability;
- build provenance or reproducible-build guarantees;
- general software-supply-chain security.

`ThresholdReleaseStateStore` is caller-owned security state. Deleting or maliciously modifying that file can remove the local rollback guarantee, so applications that rely on this property must protect the state using their surrounding operating-system/application trust boundary.

## Evidence boundary

Regression coverage verifies canonical policy/checkpoint construction, key-order independence, threshold enforcement, duplicate/untrusted signer refusal, local revocation behavior, rollback/equivocation persistence, real conditional loopback HTTP publication, compiler-free execution of the accepted finite symbolic bundle family, and rejection of an older still-valid threshold-signed channel head before any archive request.

These tests establish the specified authorization and rollback semantics. They are not evidence for properties listed as non-goals above.
