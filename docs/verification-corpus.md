# Deterministic verification corpus

This subsystem promotes the deterministic repro, differential, metamorphic, and cross-configuration verification stack from one-failure-at-a-time discovery to a persistent multi-failure corpus that can be merged, deduplicated, stored, and replayed without introducing a second IR or repro format.

## Corpus versions and compatibility

The format name remains `tiny-tensor-verification-corpus`.

Version `1` is the historical differential/metamorphic corpus format. Its canonical serialization is preserved: a corpus containing only `differential` and `metamorphic` entries still serializes as version 1 with exactly the original entry key set and entry-identity algorithm.

Version `2` is used only when the corpus contains at least one `configuration` entry. A mixed corpus may therefore contain historical version-1-style differential/metamorphic entries alongside configuration entries without changing the existing entries' `entry_sha256` identities. A version-2 document with no configuration entry is rejected because its canonical representation is version 1.

Every entry retains:

- a stable `kind` and failure `signature`;
- an optional metamorphic `relation`;
- one or more `witness_seeds`, sorted and unique;
- already-minimized canonical repro artifacts from the existing `tiny-tensor-repro` format;
- an `entry_sha256` identity.

Differential entries contain exactly one minimized repro. Metamorphic entries contain exactly the minimized baseline/transformed repro pair and retain the relation that makes the pair meaningful.

Configuration entries also contain `baseline_configuration` and `failing_configuration`. The baseline is the canonical `serial-copied` configuration from the cross-configuration oracle, and the failing configuration must be one of the fixed verified native configurations (`serial-copied`, `parallel-copied`, `serial-borrowed`, or `parallel-borrowed`). Configuration entries contain one minimized repro.

## Failure identity and deduplication

For version-1-compatible differential/metamorphic entries, the entry identity remains SHA-256 over canonical JSON containing the entry kind, stable signature, relation, and ordered repro SHA-256 identities. Witness seeds are deliberately excluded so several seeds that shrink to one exact failure can be represented by one entry with a deterministic sorted witness union.

Configuration entry identity additionally includes the exact baseline/failing configuration pair. Two failures that minimize to identical repro bytes but diverge in different native configurations therefore remain distinct identities rather than being incorrectly deduplicated.

This remains deduplication of exact minimized failure identities, not semantic bug clustering. Two artifacts that happen to represent one underlying compiler defect but differ in stable signature, configuration pair, relation, or minimized canonical repro identity remain separate entries.

## Collection

`collect_differential_corpus()` and `collect_metamorphic_corpus()` walk every seed in the requested bounded range and reuse their existing `cases=1` campaign implementations, including established failure classification and deterministic shrinking.

`collect_configuration_corpus()` does the same for `run_configuration_metamorphic_campaign()`. It persists the already-minimized configuration-specific failure identity produced by that oracle, including the exact baseline/failing pair, rather than inventing a second configuration runner, generator, shrinker, or signature taxonomy.

Unlike the first-failure campaign APIs, the corpus collectors do not stop after the first failing seed. They retain every shrunk failure from the requested range and deduplicate only after the existing per-seed shrink has produced a canonical minimized artifact.

## Canonical persistence and fail-closed loading

`serialize_verification_corpus()` emits compact JSON with sorted keys. `save_verification_corpus()` writes exactly that UTF-8 document and returns its SHA-256 content digest; this subsystem does not claim crash-safe or transactional filesystem publication.

`load_verification_corpus()` validates before returning a corpus:

- exact top-level and version-appropriate entry key sets;
- exact format name and supported version;
- duplicate JSON object keys are rejected;
- version 1 cannot contain configuration entries;
- version 2 must contain at least one configuration entry;
- entry kinds, relation namespaces, configuration names, configuration/signature binding, repro arity, signatures, and witness-seed ordering are checked;
- every nested repro must independently pass the existing canonical repro loader and must already be byte-for-byte canonical JSON;
- metamorphic repro pairs must carry equal expected-output count, shape, dtype, and raw bits;
- every entry identity is recomputed from the nested repro digests and its identity fields and compared with `entry_sha256`;
- duplicate entry identities are rejected in serialized input;
- the fully decoded corpus must serialize back to the exact input document.

`merge_verification_corpora()` is the controlled deduplication path. Entries with one exact identity must agree on kind, signature, relation, repro bytes, and configuration pair; their witness seeds are merged as a sorted set.

## Replay

`replay_verification_corpus()` keeps the historical replay contract for differential/metamorphic entries: `backend="reference"` verifies the captured semantic baseline and `backend="native"` sends each stored minimized repro through the ordinary compiler/native path with exact expected output count, shape, dtype, and raw bits.

Configuration entries preserve the same regression philosophy: replay does **not** require the historical defect to still reproduce. Reference replay validates the stored minimized repro once. Native replay reconstructs the stored baseline/failing configuration pair and executes the minimized case under each distinct stored configuration; every execution must now agree exactly with the captured reference outputs. A repaired configuration-specific divergence therefore becomes a deterministic regression gate instead of a test that only passes while the old bug remains present.

The global native `parallel` replay option remains a caller choice for ordinary differential/metamorphic entries. Configuration entries ignore that global scheduling choice because their exact serial/OpenMP and copied/borrowed modes are part of the stored failure provenance.

Corpus replay does not re-run generation or shrinking and does not need the original witness seed to be regenerated.

## Evidence boundary

The production implementation is exercised across Ubuntu and Windows with Python 3.11 and 3.13. Configuration-corpus coverage includes version-1 serialization compatibility, version-2 canonical round trips, pair-aware identities, witness-seed deduplication, mixed v1/v2 corpus merging, fail-closed configuration metadata/version checks, reference replay, and real native replay of the stored configuration pair.

This subsystem does **not** claim code-coverage percentage, path coverage, statistical bug-discovery rate, fuzzing completeness, security-fuzzing effectiveness, semantic bug clustering, corpus optimality, or performance improvement. Configuration metadata records deterministic regression provenance for the existing cross-configuration oracle; it does not create a new correctness oracle by itself.

## Phase promotion

Adding more configuration aliases, corpus fields, or version numbers without a new independently testable property would be schema farming. After configuration-failure persistence is integrated, the next verification promotion should be selected fresh. Reduction-aware metamorphic relations are appropriate only once the reduction surface has stable ownership; cross-compiler metamorphism is appropriate only when two independent toolchains are genuinely executable in one supported environment. Otherwise the project should promote to another compiler/runtime frontier rather than expanding corpus syntax.
