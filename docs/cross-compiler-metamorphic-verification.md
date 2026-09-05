# Deterministic cross-compiler metamorphic verification

This subsystem adds one same-host compiler-divergence oracle on top of the existing deterministic differential generator, exact result comparator, shrinker, and canonical repro artifact. The oracle itself does not introduce another tensor IR, generator, repro schema, or native execution engine; compiler-specific failures may now be persisted through the separately versioned verification-corpus v3 layer.

## Compiler relation

`run_cross_compiler_metamorphic_campaign()` executes one identical generated tensor module and identical runtime inputs with two explicitly distinct native C compiler configurations. The canonical ordered pair is:

1. `gcc` — baseline;
2. `clang` — candidate.

Each configuration enters the ordinary `compile_module(..., compiler=...)` path, so the comparison covers the same verified Tensor IR -> Buffer/Loop lowering -> storage/layout verification -> fusion -> generated C -> native compilation -> dynamic loading -> execution pipeline. Only the selected compiler command differs.

Outputs must agree exactly in count, shape, dtype, and raw C-order bits through the existing differential comparator. The configuration order is deterministic and part of the failure identity.

The Ubuntu CI integration test resolves both `gcc` and `clang` with `shutil.which()`, requires both commands to exist, and executes a bounded real native campaign through them. Missing GCC or Clang on that supported evidence environment is therefore a test failure, not a skipped or silently degraded verification mode.

Windows continues to run the pure harness, validation, signature, shrinking, and persistent-corpus schema regressions, but this subsystem does not claim two independent native compiler toolchains on the Windows CI image. Existing MSVC execution remains covered by the repository's ordinary full regression matrix.

## Oracle boundary

This oracle detects **compiler divergence**, not absolute semantic correctness. If GCC and Clang produce the same wrong result, the relation passes; the existing reference-vs-native differential campaign remains the semantic ground-truth oracle.

Likewise, this is not a cross-platform equivalence claim. GCC-vs-Clang is evaluated on one Ubuntu environment so target ABI, OS, architecture, and runtime platform stay fixed while the C compiler changes.

The first compiler-specific exception or result mismatch stops the bounded campaign. Exception signatures retain only the exception type and compiler pair, excluding compiler diagnostics, temporary paths, and other unstable message text. Result mismatch signatures reuse the existing exact mismatch taxonomy.

## Deterministic shrinking and repros

A failing generated case is minimized with the existing deterministic `_shrink_spec()` order. A candidate shrink is accepted only when it preserves the exact same compiler-aware failure observation, including the same failing compiler.

The original and minimized cases use the existing canonical `tiny-tensor-repro` format. Compiler selection is verification-run configuration rather than high-level tensor-program semantics, so compiler provenance is not embedded into the repro artifact.

## Persistent regression corpus

`collect_cross_compiler_corpus()` reuses the existing campaign one seed at a time, retains each already-minimized compiler failure, and deduplicates exact failure identities through the common deterministic verification corpus. Compiler persistence does not create another generator, shrinker, comparison rule, or signature taxonomy.

A corpus containing a compiler entry serializes as verification-corpus version 3. Each compiler entry stores one minimized canonical repro plus the canonical `baseline_compiler` and `failing_compiler` names. Absolute compiler paths, command aliases, diagnostics, and temporary filesystem details are not stable corpus identity.

Historical differential/metamorphic v1 entries and native-configuration v2 entries remain byte-compatible when merged into a v3 document: their entry payloads and `entry_sha256` identities are not rehashed merely because the enclosing document version is newer.

Reference corpus replay validates the minimized compiler repro without requiring native toolchains. Native replay reconstructs the stored canonical compiler pair from the current environment and requires every distinct stored compiler execution to agree exactly with the captured reference result. A repaired historical divergence therefore becomes a regression gate; replay does not require the old compiler bug to continue reproducing.

The persistent compiler pair cannot be replaced by a global `compiler=` replay override. This prevents a caller from silently discarding the provenance that makes a compiler-corpus entry meaningful.

## Regression and execution evidence

Coverage proves:

- the default compiler pair and ordering are stable;
- an injected reference runner observes the exact compiler order deterministically;
- Ubuntu requires executable GCC and Clang and performs real compile/load/execute work through both commands;
- a compiler-only result mismatch shrinks deterministically to one canonical repro;
- compiler-specific exception signatures exclude unstable message/path text;
- invalid seeds, case counts, compiler pairs, duplicate names/commands, and ambiguous custom-runner/cache options fail closed;
- compiler failures can be collected, deduplicated, canonically serialized as corpus v3, merged with older v1/v2 entries without changing their identities, and fail-closed loaded;
- Ubuntu native corpus replay reconstructs the stored GCC/Clang pair and requires both current toolchains to match the captured reference bits;
- the full existing Windows regression matrix remains compatible while real two-toolchain evidence remains explicitly Ubuntu-specific.

The original oracle implementation/test head `85cd6cf901059b5dba953f47a0c8aa7d886b59a9` passed CI #886 / run `33988966502` across Ubuntu/Windows × Python 3.11/3.13. Ubuntu Python 3.11 passed Ruff and **479 tests**, including the real GCC/Clang campaign. Persistent v3 corpus integration is validated separately by its own exact-head CI before merge.

## Exclusions and phase promotion

This subsystem intentionally does not:

- claim that GCC or Clang is independently correct;
- compare different operating systems or target ABIs;
- persist arbitrary compiler commands or host paths;
- add compiler-name aliases or arbitrary compiler matrices;
- add reduction-aware relations while the independent single-axis reduction branch retains ownership;
- claim fuzzing completeness, compiler-conformance completeness, code/path coverage percentages, statistical discovery rates, or performance improvements.

With compiler-divergence persistence closed by corpus v3, simply adding more aliases, more seeds, a third compiler, or another corpus version without a new independently testable property would be low-value farming. A later verification promotion should be selected fresh from genuinely new evidence dimensions: reduction-aware relations after reduction ownership converges, a semantic/IR invariant not reducible to the existing oracles, or another backend/target execution frontier.
