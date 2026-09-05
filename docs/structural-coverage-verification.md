# Structural coverage-guided verification

This phase adds a deterministic **measured structural coverage** layer on top of the existing verification generators and correctness oracles.

It does not claim source-line coverage, branch coverage, fuzzing completeness, model coverage, or performance improvement. The purpose is narrower: given one deterministic candidate seed range, measure a stable set of compiler-relevant structural features for each generated case, select a fixed execution budget by explicit feature gain, and then execute the already-established correctness oracles on exactly those selected seeds.

## Measurement surface

`measure_structural_coverage(seed)` first materializes the existing deterministic differential case and then runs the real compiler analysis path:

```text
generated case specification
-> tensor IR
-> Buffer IR
-> memory/layout planning
-> Loop IR
-> elementwise fusion
-> measured structural feature set
```

The measured feature vocabulary currently records bounded facts from four layers:

- generator grammar: dtype, extent class, generated operations, adjacent operation transitions, and the deterministic metamorphic relation assigned to the seed;
- tensor IR: operation kinds that actually appear after case materialization;
- Loop/layout IR: views, `copy_into`, contiguous/non-contiguous layouts, non-zero offsets, signed negative strides, and kernel extent classes;
- fused execution structure: unfused kernel opcode or canonical fused-expression family, terminal ReLU, and fused semantic step kinds, plus contiguous/non-contiguous kernel-input layout classes.

Feature names are sorted and unique inside one `StructuralCoverageObservation`. They are an in-memory selection contract for this verification phase, not a persistent public interchange schema.

## Deterministic fixed-budget selection

`select_structural_coverage_seeds()` measures every seed in a caller-supplied consecutive candidate range and applies deterministic greedy set cover:

1. compute the set of features not yet covered by every remaining case;
2. select the case with the largest new-feature gain;
3. break equal-gain ties by the numerically lower seed;
4. continue until the exact caller-supplied budget is filled.

The result records the selected observations, the complete feature union of the candidate range, the union covered by the selected budget, and any remaining uncovered features.

The selector intentionally keeps filling the requested fixed budget after feature saturation. This preserves a simple and reproducible execution-cost contract instead of silently changing the number of oracle executions when feature buckets happen to overlap.

## Existing oracles remain authoritative

Coverage measurement is a **selection dimension**, not a new semantic oracle.

Three wrappers execute selected seeds through the existing verification engines:

- `run_coverage_guided_differential_campaign()` uses the reference-vs-candidate differential oracle;
- `run_coverage_guided_metamorphic_campaign()` uses the deterministic IR metamorphic oracle;
- `run_coverage_guided_configuration_campaign()` uses the four-way serial/OpenMP × copied/borrowed native configuration oracle.

Selected seeds execute in deterministic selection order. The wrappers stop at the first underlying failure and preserve that oracle's original typed failure object, seed, signature, and repro semantics rather than translating it into a coverage-specific failure identity.

Native compiler/cache options retain the existing fail-closed runner contract. A custom runner cannot be combined with options intended only for the default native runner.

## Correctness gap exposed by instrumentation

The first production run of the measured candidate range, CI #870 on `050c581a84a912e5b249d2ff8d102659efc5d7f8`, did not fail inside the selector. It exposed a pre-existing layout-planning bug.

A generated case composed a negative-stride reverse view with `view()` targeting the **same logical shape**. `StorageLayout.reshaped()` treated every `view(shape)` operation as a shape-changing reshape and therefore required contiguous input storage. That incorrectly rejected an identity view over an already valid signed-stride alias.

The production fix keeps the strict shape-changing rule but distinguishes the identity case:

- if source and target shapes are exactly equal, the view preserves the existing offset and signed strides unchanged;
- if the shape actually changes, zero-copy view reshape still requires contiguous storage;
- ordinary `reshape()` remains a materializing copy and can therefore consume a non-contiguous logical source.

Deterministic regressions cover reverse → identity view native execution, continued refusal of reverse → shape-changing zero-copy view, and successful reverse → copy reshape materialization.

This is exactly why the measurement layer lowers real cases rather than classifying only the generator grammar: the new seed range reached a real compiler/layout boundary that the previous fixed regression set had not exercised.

## Validation

Production exact head `d772e6e01cf74779da3964d4ddf5dd87fb5263e7` passed CI #872 / run `33986903878` on:

- Ubuntu, Python 3.11;
- Ubuntu, Python 3.13;
- Windows/MSVC, Python 3.11;
- Windows/MSVC, Python 3.13.

Every matrix cell passed Ruff plus the complete pytest suite. Ubuntu Python 3.11 completed **465 tests**. The suite includes a real native coverage-guided configuration campaign, so the selection layer is not validated only with reference-runner stubs.

## Evidence boundary

This phase deliberately does **not** claim:

- source-code line, branch, path, or MC/DC coverage;
- exhaustive semantic or state-space coverage;
- randomized or mutation-guided fuzzing;
- that the current feature vocabulary is complete or permanently versioned;
- that a selected budget is minimal or globally optimal;
- that fewer selected cases have better wall-clock performance than another test plan;
- any change to deterministic verification corpus version 1.

The measured feature vocabulary can evolve with compiler architecture. If it later becomes a persistent interchange or regression-baseline format, that requires an explicit versioning decision rather than silently treating these strings as a stable file format.

## Phase promotion

Adding more feature-name synonyms, increasing the candidate range, or merely raising the selection budget would be low-value farming.

The next verification milestone should add a genuinely new oracle/persistence dimension. The strongest CPU-verifiable candidates are:

1. reduction-aware metamorphic relations with carefully stated exactness rules now that `sum` has stable frontend/IR/native ownership;
2. an explicitly versioned corpus extension for configuration-metamorphic failures while retaining corpus-v1 compatibility;
3. cross-compiler metamorphism only when two independent native toolchains can actually be executed in the same supported environment.

Which one should be selected must be decided from the fresh repository state after this measured-coverage phase is merged and its exact merged-main CI is green.