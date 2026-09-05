# Deterministic cross-compiler metamorphic verification

This phase adds one same-host compiler-divergence oracle on top of the existing deterministic differential generator, exact result comparator, shrinker, and canonical repro artifact. It does not introduce another tensor IR, generator, repro schema, corpus version, or native execution engine.

## Compiler relation

`run_cross_compiler_metamorphic_campaign()` executes one identical generated tensor module and identical runtime inputs with two explicitly distinct native C compiler configurations. The default ordered pair is:

1. `gcc` — baseline;
2. `clang` — candidate.

Each configuration enters the ordinary `compile_module(..., compiler=...)` path, so the comparison covers the same verified Tensor IR -> Buffer/Loop lowering -> storage/layout verification -> fusion -> generated C -> native compilation -> dynamic loading -> execution pipeline. Only the selected compiler command differs.

Outputs must agree exactly in count, shape, dtype, and raw C-order bits through the existing differential comparator. The configuration order is deterministic and part of the failure identity.

The Ubuntu CI integration test resolves both `gcc` and `clang` with `shutil.which()`, requires both commands to exist, and executes a bounded real native campaign through them. Missing GCC or Clang on that supported evidence environment is therefore a test failure, not a skipped or silently degraded verification mode.

Windows continues to run the pure harness, validation, signature, and shrinking regressions, but this phase does not claim two independent native compiler toolchains on the Windows CI image. Existing MSVC execution remains covered by the repository's ordinary full regression matrix.

## Oracle boundary

This oracle detects **compiler divergence**, not absolute semantic correctness. If GCC and Clang produce the same wrong result, the relation passes; the existing reference-vs-native differential campaign remains the semantic ground-truth oracle.

Likewise, this is not a cross-platform equivalence claim. GCC-vs-Clang is evaluated on one Ubuntu environment so target ABI, OS, architecture, and runtime platform stay fixed while the C compiler changes.

The first compiler-specific exception or result mismatch stops the bounded campaign. Exception signatures retain only the exception type and compiler pair, excluding compiler diagnostics, temporary paths, and other unstable message text. Result mismatch signatures reuse the existing exact mismatch taxonomy.

## Deterministic shrinking and repros

A failing generated case is minimized with the existing deterministic `_shrink_spec()` order. A candidate shrink is accepted only when it preserves the exact same compiler-aware failure observation, including the same failing compiler.

The original and minimized cases use the existing canonical `tiny-tensor-repro` format. Compiler selection is verification-run configuration rather than high-level tensor-program semantics, so this phase does not embed compiler provenance into the repro artifact.

The deterministic verification corpus remains at its existing versions. Persisting compiler-specific failures would require a separate explicit schema/version decision rather than overloading the version-2 configuration entry, whose semantics currently describe serial/OpenMP and copied/borrowed execution configurations.

## Regression and execution evidence

Coverage proves:

- the default compiler pair and ordering are stable;
- an injected reference runner observes the exact compiler order deterministically;
- Ubuntu requires executable GCC and Clang and performs real compile/load/execute work through both commands;
- a compiler-only result mismatch shrinks deterministically to one canonical repro;
- compiler-specific exception signatures exclude unstable message/path text;
- invalid seeds, case counts, compiler pairs, duplicate names/commands, and ambiguous custom-runner/cache options fail closed;
- the full existing Windows regression matrix remains compatible even though the real two-toolchain smoke is Ubuntu-specific.

Core implementation/test head `85cd6cf901059b5dba953f47a0c8aa7d886b59a9` passed CI #886 / run `33988966502` across Ubuntu/Windows × Python 3.11/3.13. Ubuntu Python 3.11 passed Ruff and **479 tests**, including the real GCC/Clang campaign.

## Exclusions and phase promotion

This phase intentionally does not:

- claim that GCC or Clang is independently correct;
- compare different operating systems or target ABIs;
- persist compiler failures into a new verification-corpus schema;
- add compiler-name aliases or arbitrary compiler matrices;
- add reduction-aware relations while the independent single-axis reduction branch retains ownership;
- claim fuzzing completeness, compiler-conformance completeness, code/path coverage percentages, statistical discovery rates, or performance improvements.

Simply adding more compiler aliases, more seeds, or a third compiler without a new independently testable property would be low-value farming. A later verification promotion should be selected fresh from genuinely new evidence dimensions: compiler-failure persistence through an explicit corpus-version decision, reduction-aware relations after reduction ownership converges, or another semantic/IR oracle that is not reducible to the existing differential, metamorphic, configuration, structural-selection, and compiler-divergence layers.
