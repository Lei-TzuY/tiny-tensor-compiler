# Threshold policy rotation

This phase adds a bounded forward-only rotation mechanism for the caller-pinned threshold release policies introduced by the threshold release authorization layer. It establishes an independently verifiable transition from one already trusted `ThresholdReleasePolicy` identity to another caller-pinned policy identity without turning remote metadata into a key-discovery mechanism.

## Trust model

The caller begins with one bootstrap `ThresholdReleasePolicy`. The caller is also responsible for obtaining and pinning every candidate next policy through an external trust/bootstrap mechanism. Rotation metadata never carries public keys and cannot make an unknown key trustworthy by itself.

A policy transition binds exactly:

- the current `threshold-ed25519:<sha256>` policy id;
- the candidate next policy id;
- one positive integer rotation epoch.

The current policy's configured `k`-of-`n` non-revoked Ed25519 members authorize that exact transition. Signatures are domain-separated, canonical, unique, and signer-id sorted. Local revocation policy is evaluated at verification time and remains caller-owned; as with threshold release authorization, revocations do not change the remote policy identity.

## Forward-only local state

`ThresholdPolicyRotationStateStore` is rooted in the caller-pinned bootstrap policy id. With no state file, the accepted state is `(epoch=0, bootstrap_policy_id)`. After that, every accepted transition must satisfy all of the following while the cross-process state lock is held:

1. the supplied current policy id equals the locally accepted current policy id;
2. the signed transition predecessor equals that current policy id;
3. the caller-pinned next policy id equals the signed target id;
4. the transition epoch is exactly `current_epoch + 1`;
5. the current policy supplies at least its threshold number of valid non-revoked signatures.

Only after all checks pass is the new `(epoch, policy_id)` state persisted with fsync plus atomic replace. Two competing same-epoch transitions therefore cannot both advance one protected local state: once one target is accepted, the other transition names a stale predecessor and fails.

The state file also records the bootstrap policy id. Reopening an existing state under a different bootstrap anchor fails closed rather than silently adopting the stored policy chain.

## Relationship to release checkpoints

Rotation deliberately does not merge policy state and release-channel rollback state into one object. After a transition is accepted, the caller uses the newly pinned `ThresholdReleasePolicy` with the existing threshold release checkpoint APIs. The policy rotation store answers only which policy identity the local trust chain has advanced to; the existing `ThresholdReleaseStateStore` independently protects per-policy/per-channel release sequences and archive digests.

This separation avoids changing the existing threshold release wire format or registry paths and prevents a policy transition from implicitly authorizing a particular release payload.

## Fail-closed boundaries

The implementation rejects:

- fewer than the current policy's threshold number of eligible signatures;
- duplicate, unknown, malformed, or locally revoked signers;
- noncanonical transition JSON or Ed25519 signature encoding;
- a transition signed by a policy other than the caller's current pinned policy;
- a transition targeting a policy other than the caller's pinned candidate next policy;
- epoch zero, replay, skipped epochs, and stale predecessors;
- no-op transitions whose source and target policy ids are identical;
- corrupt/noncanonical local state or state rooted in a different bootstrap policy.

## Security claim boundary

This is a **pin-to-pin forward authorization** property under protected caller-owned local state. It is not key discovery, key recovery, or a general root-metadata system. In particular it does not provide:

- first-contact freshness;
- a secure mechanism for discovering the candidate next public keys;
- recovery if the current policy can no longer produce its threshold;
- emergency override or offline root roles;
- delegated roles or TUF-style root metadata;
- trusted timestamps or expiry;
- transparency logs, gossip, or global consistency evidence;
- protection if an attacker can rewrite/delete the caller's local rotation state or replace the caller's pinned policy material;
- organizational/PKI identity for the signing keys;
- general software-supply-chain security.

The narrow property is: assuming the bootstrap/current policy material and local rotation state are protected, advancing to a different pinned policy requires a valid threshold authorization by the currently accepted policy, and the accepted local chain cannot move backward or fork through the rotation API.

## Evidence

Focused regressions cover canonical transition creation, signer-order independence, threshold enforcement, current-policy revocation, pinned-next-policy mismatch, transition tampering, predecessor mismatch, persistent bootstrap anchoring, one-step epoch advancement, replay/fork/skip rejection, corrupt state refusal, and use of the newly accepted policy to authorize a normal threshold release checkpoint.

The first implementation CI exercises these contracts together with the full existing repository test suite on Ubuntu and Windows with Python 3.11 and 3.13. This evidence establishes the specified local rotation semantics only; it is not evidence for the non-goals above.
