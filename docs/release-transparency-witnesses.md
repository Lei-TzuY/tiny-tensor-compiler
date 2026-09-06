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

## Optional stateful witness creation

`create_stateful_transparency_witness_quorum()` is the bounded stateful signing path. It preserves the existing quorum wire format and verifier, but selected witness keys are not invoked until their caller-supplied local append-only state accepts the checkpoint.

For every selected signer the caller supplies one distinct `TransparencyStateStore` for the same pinned log operator plus that witness's RFC 6962 consistency proof. Creation then:

1. verifies the signed checkpoint against the pinned log public key;
2. verifies signer membership, revocation status, and uniqueness under the witness policy;
3. verifies that every state store is distinct and pinned to the checkpoint's log;
4. calls `precheck()` for **all** selected witness stores before mutating any of them;
5. calls `record()` for every selected store, which repeats the append-only check while holding that store's own state lock; and only then
6. delegates to the existing canonical `create_transparency_witness_quorum()` encoder/signature domain.

The precheck-all phase means a deterministic malformed, rollback, fork, or invalid consistency proof supplied in the call cannot partially advance the other selected stores. Each later `record()` deliberately rechecks under its own file lock, so a concurrent state change between precheck and persistence cannot silently bypass that witness's local floor.

The stores are nevertheless separate persistent files, not one distributed transaction. If an external concurrent actor changes one store after the precheck phase, an earlier selected store may already have independently accepted the checkpoint before a later `record()` fails. In that case this function returns no quorum, but it does not claim to roll back a legitimate append-only state advance that another witness store already persisted. Callers that require transactional coordination across independently operated witnesses need a broader protocol.

First contact also retains the base transparency limitation: a witness store with no prior checkpoint has no independent freshness signal. Its first accepted checkpoint establishes a local floor for later consistency checks; it does not prove that no older or alternative checkpoint existed elsewhere.

## State advancement ordering

`accept_witnessed_release_transparency()` verifies the witness quorum **before** calling the existing `accept_release_transparency()` path. Therefore malformed, wrong-policy, wrong-checkpoint, revoked, insufficient, or cryptographically invalid witness evidence cannot advance the caller's `TransparencyStateStore`.

Once the witness gate succeeds, the existing log-signature, inclusion-proof, consistency-proof, and persistent append-only state checks remain unchanged. This layer does not introduce a second transparency verifier or client-state implementation.

## Security boundary

A plain quorum created by `create_transparency_witness_quorum()` proves only that at least `k` caller-pinned, non-revoked witness keys produced valid endorsements for the same exact checkpoint bytes under the configured policy.

A quorum returned by `create_stateful_transparency_witness_quorum()` additionally proves a local execution property for the supplied selected witnesses: before their keys were used by that call, each supplied witness state store accepted the signed checkpoint as an append-only continuation of its own persisted local floor (or as first contact when no floor existed).

Neither form proves that the witnesses:

- are independently operated or have independent key/state custody merely because their keys or state paths differ;
- reconstructed the full log or verified every release leaf;
- exchanged checkpoints with one another or with other clients;
- compared their local views against an external witness network;
- observed a globally unique or freshest log view; or
- provide trusted timestamps, expiry, availability, PKI identity, or recovery from hostile deletion/replacement of their local state.

Accordingly, stateful witness creation is **not** a general gossip protocol and does not claim complete split-view prevention. It converts configured witnesses from stateless checkpoint signers into signers with caller-supplied local append-only memory. The strength of that property still depends on real witness independence, state protection, proof delivery, and first-contact assumptions.

The next transparency promotion should add explicit cross-witness evidence—such as checkpoint exchange/cross-comparison, independently persisted witness observations that clients can compare, or another bounded mechanism that makes divergent views externally detectable—rather than adding more signature-envelope variants or merely increasing witness counts.

## Verification evidence

The original quorum regressions cover deterministic policy/quorum encoding, threshold enforcement, duplicate and unknown signers, exact checkpoint-byte binding, cryptographic signature tampering, local revocation effects, policy bounds, and fail-before-client-state-advance integration.

Stateful-witness regressions additionally cover first-contact persistence before signing; valid RFC 6962 growth; precheck-all refusal without deterministic partial state advance when one proof is damaged; rollback and same-size fork refusal; distinct-store and pinned-log requirements; and invalid checkpoint-signature refusal before witness state changes. The full Ubuntu/Windows × Python 3.11/3.13 CI matrix exercises the same state-file and verification behavior.
