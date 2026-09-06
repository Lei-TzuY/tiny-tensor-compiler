# Durable transparency witness evidence

`TransparencyWitnessEvidenceStore` turns portable signed witness observations into caller-local durable consistency memory for one pinned transparency log and one caller-supplied witness policy.

This is deliberately narrower than a gossip or freshness service. The store never fetches observations, never discovers checkpoints, never contacts witnesses, and never decides which remote view is globally newest. A missing state file is first contact and provides no external split-view protection.

## Healthy state

A healthy state keeps at most one latest verified observation per witness. Every stored observation is persisted in its original signed encoding and is cryptographically re-verified whenever the state file is read; the JSON file is not a new trust root.

For a witness already present in the store, an exact repeated observation is idempotent and a smaller tree size is rejected as rollback. A larger observation replaces that witness only after all required append-only relations to the already remembered checkpoint views have been proven.

The state file is bound to the configured log identity and witness-policy identity, uses canonical JSON, is size bounded, and is updated under the repository's cross-process state lock with temporary-file write, `fsync`, and atomic replacement.

## Consistency-proof fanout

When a caller records an observation at a tree size different from remembered checkpoints, it must provide the exact RFC 6962 consistency proofs needed to relate the new view to every distinct remembered checkpoint digest at a different size.

Proofs are supplied as a mapping keyed by checkpoint digest. If multiple witnesses previously signed the same exact checkpoint bytes, that checkpoint requires only one proof. Missing proofs, extra proofs, malformed proof nodes, and invalid append-only relations fail before the state file is changed.

This design intentionally does not auto-fetch proofs. The caller remains responsible for obtaining evidence through an independent transport or deployment workflow.

## Terminal fork evidence

If two verified observations bind the same log tree size to different Merkle roots, the store persists the signed pair as terminal fork evidence. This applies both to cross-witness disagreement and to one witness equivocating by signing two divergent heads at the same size.

After terminal fork evidence is stored, subsequent `record()` calls fail closed instead of replacing or healing the evidence. The original signed observations remain available for external audit.

Same-size observations that agree on the root must also bind the exact same checkpoint bytes; the store does not treat a matching root alone as sufficient checkpoint identity.

## Security boundary

The store establishes these local properties:

- durable memory of previously verified witness views;
- per-witness rollback refusal;
- explicit append-only proof requirements across remembered tree sizes;
- deterministic persistence of signed same-size fork evidence;
- cross-process serialization and atomic state replacement;
- re-verification of persisted signed observations under the caller's current log key and witness policy.

It does **not** establish:

- global checkpoint freshness;
- network gossip or witness discovery;
- automatic consistency-proof retrieval;
- clock-based staleness guarantees;
- universal split-view prevention;
- proof that all clients or witnesses have observed the same head.

Those require an external communication or publication system. This phase provides the durable local evidence substrate such a system can build on without silently inventing those stronger claims.
