# Release transparency witness quorum

This phase adds an optional caller-pinned witness endorsement gate on top of the existing append-only release-transparency checkpoint verifier.

The base transparency layer already verifies a signed log checkpoint, inclusion of exact authenticated release bytes, consistency from a caller's persisted checkpoint, and local rollback/fork refusal. A malicious log can still serve different individually consistent views to isolated first-contact clients. Witness endorsements add one bounded independent trust input: a caller may require `k` distinct configured Ed25519 witness keys to sign the exact checkpoint bytes before local transparency state is allowed to advance.

## Policy model

`TransparencyWitnessPolicy` contains:

- one to sixteen caller-pinned raw Ed25519 public keys;
- an integer threshold `k` between one and the number of configured keys;
- an optional caller-local set of revoked witness fingerprints.

Keys are canonicalized by the existing stable `ed25519:<sha256(public-key)>` fingerprint and sorted before policy identity is computed. The policy id commits to the schema, threshold, and complete ordered witness set. Local revocations do not change that identity; they affect which otherwise-valid endorsements count toward the current threshold.

A policy is rejected if keys are duplicated, the threshold is outside the configured key set, or current local revocations leave fewer eligible keys than the threshold.

## Endorsement model

`create_transparency_witness_quorum()` signs a domain-separated message containing:

- the canonical witness-quorum schema;
- the exact witness policy identity; and
- `SHA256(encoded_checkpoint)`, where `encoded_checkpoint` is the exact signed transparency-checkpoint byte string presented to the caller.

The resulting quorum envelope is canonical JSON with signatures sorted by witness identity. Creation rejects unknown, revoked, or duplicate witness keys and requires at least the configured threshold.

`verify_transparency_witness_quorum()` independently checks canonical framing, exact checkpoint-byte digest binding, exact policy identity, membership of every signer, Ed25519 validity of every listed signature, uniqueness/order, revocations, and the final non-revoked threshold.

This deliberately signs the exact checkpoint bytes rather than only the Merkle root. Re-encoding or substituting checkpoint metadata while keeping an equal root does not preserve the witness endorsement.

## State advancement ordering

`accept_witnessed_release_transparency()` verifies the witness quorum **before** calling the existing `accept_release_transparency()` path. Therefore malformed, wrong-policy, wrong-checkpoint, revoked, insufficient, or cryptographically invalid witness evidence cannot advance `TransparencyStateStore`.

Once the witness gate succeeds, the existing log-signature, inclusion-proof, consistency-proof, and persistent append-only state checks remain unchanged. This phase does not introduce a second transparency state implementation.

## Security boundary

A valid witness quorum proves only that at least `k` caller-pinned, non-revoked witness keys produced valid endorsements for the same exact checkpoint bytes under the configured policy.

It does **not** prove that those witnesses:

- independently reconstructed or monitored the log;
- verified consistency against their own prior checkpoints;
- exchanged checkpoints with one another or with other clients;
- possess independent organizational control simply because their keys differ;
- observed a globally unique log view; or
- provide freshness, timestamps, expiry, availability, or PKI identity.

Accordingly, this phase is **not** a general gossip protocol and does not claim complete split-view prevention. For a caller that deliberately pins independently controlled witnesses and requires a threshold, a malicious log can no longer satisfy that local acceptance policy by acting alone; enough configured witness keys must also endorse the presented checkpoint. The strength of that property depends on the caller's actual witness independence and key custody.

The next transparency promotion should add independently verifiable cross-view evidence—such as persisted witness-side consistency state, explicit checkpoint gossip/cross-comparison, or another bounded mechanism that makes witness behavior itself auditable—rather than merely increasing witness counts or adding more signature-envelope variants.

## Verification evidence

Focused regressions cover deterministic policy/quorum encoding, threshold enforcement, duplicate and unknown signers, exact checkpoint-byte binding, cryptographic signature tampering, local revocation effects, policy bounds, and fail-before-state-advance integration. The full Ubuntu/Windows × Python 3.11/3.13 CI matrix exercises the same public API.
