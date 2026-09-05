# Deterministic verification corpus

This subsystem promotes the deterministic repro, differential, metamorphic, cross-configuration, and cross-compiler verification stack from one-failure-at-a-time discovery to a persistent multi-failure corpus that can be merged, deduplicated, stored, and replayed without introducing a second IR or repro format.

## Corpus versions and compatibility

The format name remains `tiny-tensor-verification-corpus`.

Version `1` is the historical differential/metamorphic corpus format. Its canonical serialization is preserved: a corpus containing only `differential` and `metamorphic` entries still serializes as version 1 with exactly the original entry key set and entry-identity algorithm.

Version `2` is used when the corpus contains at least one `configuration` entry and no compiler-pair entry. A mixed v2 corpus may contain historical version-1-style differential/metamorphic entries alongside configuration entries without changing the existing entries' `entry_sha256` identities. A version-2 document with no configuration entry is rejected because its canonical representation is version 1.

Version `3` is used only when the corpus contains at least one `compiler` entry. A v3 corpus may also contain historical differential/metamorphic entries and version-2 configuration entries; those older entries retain their existing payloads and `entry_sha256` identities. A version-3 document with no compiler entry is rejected because its canonical representation belongs to an older version.

Every entry retains:

- a stable `kind` and failure `signature`;
- an optional metamorphic `relation`;
- one or more `witness_seeds`, sorted and unique;
- already-minimized canonical repro artifacts from the existing `tiny-tensor-repro` format;
- an `entry_sha256` identity.

Differential entries contain exactly one minimized repro. Metamorphic entries contain exactly the minimized baseline/transformed repro pair and retain the relation that makes the pair meaningful.

Configuration entries also contain `baseline_configuration` and `failing_configuration`. The baseline is the canonical `serial-copied` configuration from the cross-configuration oracle, and the failing configuration must be one of the fixed verified native configurations (`serial-copied`, `parallel-copied`, `serial-borrowed`, or `parallel-borrowed`). Configuration entries contain one minimized repro.

Compiler entries contain exactly one minimized repro plus `baseline_compiler` and `failing_compiler`. The persisted baseline is the canonical `gcc` compiler and the failing compiler must be one of the canonical cross-compiler oracle names (`gcc` or `clang`). The corpus records names only: absolute executable paths, command aliases, diagnostics, temporary paths, and other host-specific compiler details are deliberately excluded from stable persistence.

## Failure identity and deduplication

For version-1-compatible differential/metamorphic entries, the entry identity remains SHA-256 over canonical JSON containing the entry kind, stable signature, relation, and ordered repro SHA-256 identities. Witness seeds are deliberately excluded so several seeds that shrink to one exact failure can be represented by one entry with a deterministic sorted witness union.

Configuration entry identity additionally includes the exact baseline/failing configuration pair. Compiler entry identity similarly includes the canonical baseline/failing compiler pair. Two failures that minimize to identical repro bytes but belong to different stored configuration or compiler pairs therefore remain distinct identities rather than being incorrectly deduplicated.

This remains deduplication of exact minimized failure identities, not semantic bug clustering. Two artifacts that happen to represent one underlying compiler defect but differ in stable signature, stored pair, relation, or minimized canonical repro identity remain separate entries.

## Collection

`collect_differential_corpus()` and `collect_metamorphic_corpus()` walk every seed in the requested bounded range and reuse their existing `cases=1` campaign implementations, including established failure classification and deterministic shrinking.

`collect_configuration_corpus()` does the same for `run_configuration_metamorphic_campaign()`. It persists the already-minimized configuration-specific failure identity produced by that oracle, including the exact baseline/failing pair, rather than inventing a second configuration runner, generator, shrinker, or signature taxonomy.

`collect_cross_compiler_corpus()` reuses `run_cross_compiler_metamorphic_campaign()` one seed at a time and persists its minimized compiler-aware failure. Collection intentionally uses the canonical ordered GCC/Clang oracle pair; custom runners may inject deterministic failures for testing, but persistent identity still records only the canonical compiler names. Passing a custom runner together with a native cache directory remains invalid because cache ownership belongs to the default native runner.

Unlike the first-failure campaign APIs, the corpus collectors do not stop after the first failing seed. They retain every shrunk failure from the requested range and deduplicate only after the existing per-seed shrink has produced a canonical minimized artifact.

## Canonical persistence and fail-closed loading

`serialize_verification_corpus()` emits compact JSON with sorted keys. `save_verification_corpus()` writes exactly that UTF-8 document and returns its SHA-256 content digest; this subsystem does not claim crash-safe or transactional filesystem publication.

`load_verification_corpus()` validates before returning a corpus:

- exact top-level and version-appropriate entry key sets;
- exact format name and supported version;
- duplicate JSON object keys are rejected;
- version 1 cannot contain configuration or compiler entries;
- version 2 must contain at least one configuration entry and cannot contain compiler entries;
- version 3 must contain at least one compiler entry;
- entry kinds, relation namespaces, stored configuration/compiler names, pair/signature binding, repro arity, signatures, and witness-seed ordering are checked;
- every nested repro must independently pass the existing canonical repro loader and must already be byte-for-byte canonical JSON;
- metamorphic repro pairs must carry equal expected-output count, shape, dtype, and raw bits;
- every entry identity is recomputed from the nested repro digests and its identity fields and compared with `entry_sha256`;
- duplicate entry identities are rejected in serialized input;
- the fully decoded corpus must serialize back to the exact input document.

`merge_verification_corpora()` is the controlled deduplication path. Entries with one exact identity must agree on kind, signature, relation, repro bytes, and any stored configuration/compiler pair; their witness seeds are merged as a sorted set. Mixing v1/v2/v3 data therefore promotes only the enclosing document version and never rehashes older entry identities.

## Replay

`replay_verification_corpus()` keeps the historical replay contract for differential/metamorphic entries: `backend="reference"` verifies the captured semantic baseline and `backend="native"` sends each stored minimized repro through the ordinary compiler/native path with exact expected output count, shape, dtype, and raw bits.

Configuration entries preserve the same regression philosophy: replay does **not** require the historical defect to still reproduce. Reference replay validates the stored minimized repro once. Native replay reconstructs the stored baseline/failing configuration pair and executes the minimized case under each distinct stored configuration; every execution must now agree exactly with the captured reference outputs.

Compiler entries use the same repaired-regression model. Reference replay validates the canonical repro without needing either native compiler. Native replay reconstructs the stored canonical compiler pair from the current environment and executes the case through each distinct stored compiler; every execution must agree exactly with the captured reference outputs. A fixed GCC/Clang divergence therefore becomes a durable regression gate instead of a test that succeeds only while the original divergence remains present.

A global native `compiler=` override is rejected when compiler entries are present because it would erase the stored compiler-pair provenance. The global native `parallel` replay option remains a caller choice for ordinary differential/metamorphic entries. Configuration entries ignore that global scheduling choice because their exact serial/OpenMP and copied/borrowed modes are stored provenance; compiler entries use the ordinary serial native path selected by the cross-compiler oracle.

Corpus replay does not re-run generation or shrinking and does not need the original witness seed to be regenerated.

## Evidence boundary

The corpus schema, compatibility, validation, merge, and reference-replay implementation is exercised across Ubuntu and Windows with Python 3.11 and 3.13. Compiler-corpus coverage additionally requires executable `gcc` and `clang` on Ubuntu and performs real compile/load/execute replay through both toolchains. Windows does not claim a second independent compiler pair; it verifies the platform-independent v3 schema/harness while the repository's ordinary matrix continues to exercise MSVC native execution separately.

This subsystem does **not** claim code-coverage percentage, path coverage, statistical bug-discovery rate, fuzzing completeness, security-fuzzing effectiveness, semantic bug clustering, corpus optimality, compiler conformance/completeness, or performance improvement. Stored configuration/compiler metadata is deterministic regression provenance for existing independently exercised oracles; persistence does not create a new correctness oracle by itself.

## Phase promotion

Version 3 closes persistence for the current differential, metamorphic, native-configuration, and same-host GCC/Clang compiler-divergence failure classes. Adding more corpus fields, compiler aliases, version numbers, or larger seed counts without a new independently testable property would be schema farming.

The next verification promotion should be selected fresh from a genuinely new evidence dimension. Reduction-aware relations remain appropriate only after the independently owned reduction surface converges; otherwise a new semantic/IR invariant, target/backend oracle, or another compiler/runtime frontier should be preferred over expanding corpus syntax.
