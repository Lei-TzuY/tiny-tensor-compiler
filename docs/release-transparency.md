# Release transparency consistency

This phase adds an optional, caller-pinned transparency layer for already-authenticated release checkpoint bytes. It is deliberately separate from publisher and threshold release authorization: callers must first validate a release checkpoint with the existing release verifier, then may require evidence that those exact bytes are included in an append-only log.

## Checkpoint model

A transparency checkpoint contains a canonical schema identifier, the pinned log operator identity, a positive tree size, and a SHA-256 Merkle root. The checkpoint body is domain-separated and signed with Ed25519. The caller supplies the raw 32-byte log public key; the stable `ed25519:<sha256(public-key)>` fingerprint must match the checkpoint `log_id`.

The Merkle construction follows the RFC 6962 domain separation used by Certificate Transparency:

- leaf hash: `SHA256(0x00 || leaf_bytes)`;
- interior hash: `SHA256(0x01 || left_hash || right_hash)`.

For release transparency, `leaf_bytes` are the exact canonical release or threshold-release checkpoint bytes that the caller authenticated separately. Transparency does not reinterpret or re-authorize the release payload.

## Inclusion and consistency

`verify_transparency_inclusion()` verifies an audit path for one exact leaf index against a signed checkpoint root. `verify_transparency_consistency()` verifies that a newer tree head is an append-only extension of a previously accepted tree head for the same pinned log operator.

Proof inputs are bounded: tree sizes are limited to `2^63-1`, proof paths to 64 SHA-256 nodes, and every proof node must be exactly 32 raw bytes. Malformed, incomplete, extra-node, wrong-root, and cross-operator proofs fail closed.

## Persistent local floor

`TransparencyStateStore` persists one `(log_id, tree_size, root_hash)` floor under the same cross-process locking and atomic-replace durability machinery used by release rollback state. A later head is accepted only when:

- it uses the same caller-pinned log operator;
- a smaller tree size is never presented;
- the same tree size has exactly the same root;
- a larger tree supplies a valid consistency proof from the persisted head.

Re-accepting the exact same head is idempotent. Rollback, same-size forks, corrupt/non-canonical state, and invalid consistency evidence do not overwrite the accepted floor.

`accept_release_transparency()` composes checkpoint signature verification, release-byte inclusion verification, and persistent append-only state advancement. It does **not** call the publisher/threshold release verifier itself; this separation keeps release authorization and transparency evidence explicit.

## Optional caller-pinned witness quorum

The separate witness-quorum layer may require `k` distinct caller-pinned Ed25519 witness keys to endorse the exact signed transparency-checkpoint bytes before `TransparencyStateStore` is allowed to advance. `accept_witnessed_release_transparency()` verifies that quorum first and then delegates to the unchanged append-only acceptance path above.

Witness endorsement is intentionally a separate policy surface rather than a property silently implied by the base log verifier. See [`release-transparency-witnesses.md`](release-transparency-witnesses.md) for the canonical policy/quorum format, revocation behavior, state-ordering guarantee, and its narrower security claim.

## Security boundary

The base transparency layer proves a local append-only property relative to a caller-pinned operator key and previously persisted state. By itself it does **not** provide:

- first-contact freshness when no local transparency floor exists;
- witness cosigning, gossip, or cross-client checkpoint comparison;
- protection against a malicious log serving different individually consistent split views to isolated clients;
- trusted timestamps, expiry, or external time ordering;
- PKI/organizational identity for the log key;
- recovery if an attacker can delete or rewrite the caller's local state or replace pinned key material;
- registry availability, global consistency, or a full TUF-style update framework.

The optional witness-quorum layer adds caller-pinned multi-key endorsement of one exact checkpoint, but it still does not prove that witnesses independently monitored consistency or exchanged views. Stronger split-view resistance requires independently verifiable witness-side state, gossip/cross-comparison, trusted freshness signals, or a broader update-security protocol and is intentionally not claimed here.

## Verification evidence

Focused regressions cover canonical Ed25519 checkpoint verification, operator-key binding, inclusion proofs, non-power-of-two consistency growth, damaged proofs, rollback and same-size fork refusal, idempotent re-acceptance, persistent reopen, wrong pinned operator refusal, and bounded malformed proof/tree inputs. The full repository CI matrix exercises the state layer on Ubuntu and Windows with Python 3.11 and 3.13. Witness-specific evidence is documented separately in `release-transparency-witnesses.md`.
