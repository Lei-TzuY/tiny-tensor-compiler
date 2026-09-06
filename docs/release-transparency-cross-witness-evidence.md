# Cross-witness transparency evidence

This phase turns the stateful transparency witness checkpoint state into portable,
independently signed evidence that callers can exchange and compare. It is a bounded
cryptographic evidence layer, not a network gossip protocol and not a claim that the
system globally prevents split views.

## Signed observation

A witness may create a `ttc-release-transparency-witness-observation-v1` envelope only
for the exact signed log checkpoint that its `TransparencyStateStore` currently records
as accepted. An empty state store or a checkpoint that does not match the persisted
current state is rejected before the witness signs anything.

The canonical observation envelope contains exactly:

- the exact encoded signed transparency checkpoint;
- the SHA-256 digest of those exact checkpoint bytes;
- the transparency witness policy identity;
- the witness identity;
- the observation schema identifier; and
- the witness Ed25519 signature.

The witness signature is domain-separated and binds the schema, policy identity,
witness identity, and exact checkpoint digest. Verification independently checks the
canonical observation framing, the embedded log-operator checkpoint signature, the
pinned log identity, witness policy membership and revocation state, and the witness
signature.

The observation therefore proves only that the named eligible witness signed a digest
of the embedded valid log checkpoint under the named witness policy. Creation adds the
stronger local rule that the checkpoint had already become that witness's persisted
current accepted state.

## Deterministic cross-witness comparison

`compare_transparency_witness_observations()` always re-verifies both encoded
observations. The inputs must come from two distinct eligible witnesses under one
policy and one pinned transparency log.

For equal tree sizes:

- equal roots and identical checkpoint bytes produce `same_checkpoint`;
- different roots produce `same_size_fork`.

A `same_size_fork` result retains both independently signed observation bytes. Those two
observations are portable equivocation evidence: two distinct eligible witnesses attest
to different signed log roots for the same tree size. The comparison does not need a
Merkle consistency proof for this case, and supplying one is rejected as irrelevant.

For different tree sizes, ordering by size is not enough. The comparison reports
`consistent_growth` only after the existing RFC 6962 consistency verifier accepts an
explicit consistency proof from the older checkpoint to the newer checkpoint. A
missing, malformed, or invalid proof is an ordinary verification failure. It is not
promoted to fork evidence merely because consistency could not be proved.

## Fail-closed boundaries

The evidence layer rejects:

- self-comparison of two observations signed by the same witness;
- observations for a different witness policy;
- witnesses absent from or revoked by the active policy;
- invalid log-operator or witness signatures;
- altered checkpoint bytes or checkpoint digests;
- non-canonical or duplicate-key observation JSON;
- same-size same-root observations that nevertheless bind different checkpoint bytes;
- cross-log comparisons; and
- irrelevant consistency proofs for same-size comparisons.

These checks do not weaken the existing `TransparencyStateStore` rollback/fork rules or
the existing quorum-verification rules. They create an additional portable evidence
surface rather than another endorsement threshold variant.

## Deliberate non-goals

This phase does **not** implement or claim:

- a network gossip daemon or peer-to-peer transport;
- witness discovery or rendezvous;
- timestamps, trusted freshness, maximum checkpoint age, or clock synchronization;
- global uniqueness of the latest checkpoint;
- transactional state updates across multiple witnesses;
- automatic evidence publication, aggregation, retention, or adjudication; or
- complete prevention or detection of every split-view attack.

Callers must transport observations and, for different tree sizes, obtain the required
consistency proof through some external mechanism. The implementation only defines and
verifies the cryptographic evidence that those higher-level mechanisms may exchange.

## Phase promotion

Further work on transparency should add a genuinely executable evidence capability,
such as bounded observation exchange/freshness policy or durable evidence aggregation,
with its transport and trust assumptions made explicit. Adding more envelope fields,
more signature-count variants, or unverified networking scaffolding would not constitute
a new milestone.
