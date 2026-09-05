# Deterministic verification corpus

This phase promotes the deterministic repro, differential, and metamorphic verification stack from one-failure-at-a-time discovery to a persistent multi-failure corpus that can be merged, deduplicated, stored, and replayed without introducing a second IR or repro format.

## Corpus model

A corpus is canonical versioned JSON with format name `tiny-tensor-verification-corpus` and version `1`. Every entry contains:

- `kind`: `differential` or `metamorphic`;
- the existing stable failure `signature`;
- an optional metamorphic `relation`;
- one or more `witness_seeds`, sorted and unique;
- already-minimized canonical repro artifacts from the existing `tiny-tensor-repro` format;
- an `entry_sha256` identity.

Differential entries contain exactly one minimized repro. Metamorphic entries contain exactly the minimized baseline/transformed repro pair and retain the relation that makes the pair meaningful.

The entry identity deliberately excludes witness seeds. It is SHA-256 over canonical JSON containing the entry kind, stable signature, relation, and ordered repro SHA-256 identities. If several generated seeds shrink to the same failure identity, the corpus retains one entry and unions their witness seeds in deterministic sorted order.

This is deduplication of exact minimized failure identities, not semantic bug clustering. Two repros that happen to represent the same underlying compiler defect but have different stable signatures or different minimized canonical repros remain separate entries.

## Collection

`collect_differential_corpus()` and `collect_metamorphic_corpus()` walk every seed in the requested bounded range. Each individual seed reuses the existing campaign implementation with `cases=1`, including its established failure classification and deterministic shrink order.

Unlike `run_differential_campaign()` and `run_metamorphic_campaign()`, the corpus collectors do not stop after the first failure. They retain every shrunk failure from the requested range and deduplicate only after the existing per-seed shrink has produced a canonical minimized artifact.

The original campaign APIs remain unchanged and keep their first-failure behavior. The corpus layer therefore adds persistence and multi-failure aggregation without creating a parallel generator, shrinker, candidate runner, or failure-signature implementation.

## Canonical persistence and fail-closed loading

`serialize_verification_corpus()` emits compact JSON with sorted keys. `save_verification_corpus()` writes exactly that UTF-8 document and returns its SHA-256 content digest; this phase does not claim crash-safe or transactional filesystem publication.

`load_verification_corpus()` validates before returning a corpus:

- exact top-level and entry key sets;
- exact format name and version;
- duplicate JSON object keys are rejected;
- entry kinds, relation namespaces, repro arity, signatures, and witness-seed ordering are checked;
- every nested repro must independently pass the existing canonical repro loader and must already be byte-for-byte canonical JSON;
- metamorphic repro pairs must carry equal expected-output count, shape, dtype, and raw bits;
- every entry identity is recomputed from the nested repro digests and compared with `entry_sha256`;
- duplicate entry identities are rejected in serialized input;
- the fully decoded corpus must serialize back to the exact input document.

`merge_verification_corpora()` is the controlled deduplication path. Entries with one exact identity must agree on kind, signature, relation, and repro bytes; their witness seeds are merged as a sorted set.

## Replay

`replay_verification_corpus()` reuses `replay_repro_case()` for every minimized artifact. `backend="reference"` provides a format/semantic-baseline replay; `backend="native"` sends the same stored minimized repros through the ordinary compiler/native path and retains the existing exact expected shape/dtype/raw-bit checks.

A mixed corpus can therefore become a deterministic regression gate without regenerating the original seed campaign. Corpus replay does not re-run shrinking and does not require the original candidate defect to still exist.

For metamorphic entries, each side is still an ordinary canonical repro whose expected output was captured independently by reference execution. Loader validation additionally proves the two expected-output records agree before the pair is accepted as a metamorphic corpus entry.

## Evidence boundary

The production implementation is exercised across Ubuntu and Windows with Python 3.11 and 3.13. Regression coverage includes exact failure deduplication, witness-seed union, differential/metamorphic corpus merging, canonical JSON and file round trips, tamper/duplicate/noncanonical refusal, clean empty collections, and mixed reference/native replay.

This phase does **not** claim code-coverage percentage, path coverage, statistical bug-discovery rate, fuzzing completeness, security-fuzzing effectiveness, corpus optimality, or performance improvement. The corpus records exact deterministic failures observed by the existing bounded campaigns.

## Phase promotion

Adding more seeds, more storage wrappers, or more algebraically equivalent corpus identities is not the next milestone. The next verification promotion should add a genuinely new oracle or selection dimension, such as cross-configuration/compiler metamorphism, measured coverage-guided case selection with explicit instrumentation evidence, or reduction-aware relations once that surface has stable ownership and semantics.
