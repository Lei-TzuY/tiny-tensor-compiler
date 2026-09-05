# Deterministic cross-configuration metamorphic verification

This phase adds an executable verification dimension across already-supported native runtime configurations. It reuses the deterministic differential generator, exact output comparator, shrink order, and canonical repro artifact instead of introducing another IR, repro schema, or corpus format.

## Configuration relation

For every deterministic generated case, `run_configuration_metamorphic_campaign()` executes one identical tensor module and identical runtime inputs in the fixed order:

1. `serial-copied` — serial native execution with ordinary copied inputs;
2. `parallel-copied` — verified OpenMP scheduling with ordinary copied inputs;
3. `serial-borrowed` — serial native execution with verified borrowed inputs;
4. `parallel-borrowed` — verified OpenMP scheduling with verified borrowed inputs.

`serial-copied` is the baseline. Every later configuration must return the same output count, shape, dtype, and raw C-order bits. The default runner uses `compile_module()` with the configuration's existing `parallel` and `borrow_inputs` controls, so the relation crosses Buffer/Loop lowering, alias/layout verification, fusion, generated C, the platform compiler, native loading, OpenMP scheduling where selected, and copied-versus-borrowed input binding.

The configuration order is part of the deterministic contract. An injected `configuration_runner` is available for focused regression tests, but custom runners cannot be mixed with native compiler/cache options.

## Oracle boundary

This oracle detects **configuration divergence**, not absolute semantic correctness. If all four configurations produce the same wrong result, the configuration relation passes; the existing differential campaign remains the reference-vs-native ground-truth oracle. Likewise, the ordinary metamorphic campaign remains responsible for equivalence between distinct IR programs.

The first configuration-specific exception or result mismatch stops the bounded campaign. Exception signatures retain only the exception type, never exception text, temporary native paths, or compiler diagnostics. Result mismatch signatures reuse the existing exact differential mismatch taxonomy.

This phase does not claim cross-compiler metamorphism. CI proves the relation on the platform's configured native toolchain: the GCC-style path on Ubuntu and MSVC on Windows. Comparing multiple compilers within one platform invocation requires a separate toolchain-availability and selection contract.

## Deterministic shrinking and repros

A failing generated `_CaseSpec` is minimized with the existing metamorphic shrink helper and the same deterministic order already established by differential verification:

1. delete generated operations left-to-right;
2. reduce the square tensor side length;
3. zero complete inputs;
4. zero individual elements in stable input/flat-index order.

A shrink candidate is retained only when it reproduces the exact same configuration-aware stable failure identity, including the same failing configuration. The original and minimized cases are serialized with the existing canonical `tiny-tensor-repro` format. There is no third artifact schema and no configuration-specific IR encoding.

The deterministic verification corpus remains version 1 and is intentionally unchanged in this slice. Persisting configuration-metamorphic failures would require an explicit corpus-format decision rather than silently overloading the existing `differential` / `metamorphic` entry kinds.

## Regression and execution evidence

Regression coverage proves:

- the four configuration descriptors and their order are stable;
- an injected reference runner observes that exact order and is deterministic across repeated campaigns;
- a real native bounded campaign agrees across serial/OpenMP and copied/borrowed input combinations;
- a configuration-only mismatch shrinks deterministically to one canonical repro;
- configuration-specific exception signatures exclude unstable message/path text;
- invalid seeds/case counts and ambiguous custom-runner/native-option combinations fail closed.

Production head `6d673993fe22f324e6f5053b957eb614502412ee` passed CI #864 / run `33985309737` on Ubuntu and Windows with Python 3.11 and 3.13. Every matrix cell passed Ruff and the full pytest suite. Ubuntu Python 3.11 completed **454 tests**; the Windows cells execute the real MSVC native/OpenMP paths, including the `parallel-borrowed` configuration.

## Exclusions and phase promotion

This phase intentionally does not:

- change deterministic verification corpus v1;
- add reduction-aware relations while the independent single-axis reduction surface remains unconverged;
- claim cross-compiler equivalence without two executable toolchains in one verified environment;
- claim code/path coverage, fuzzing completeness, statistical bug-discovery rates, or performance improvements;
- alter compiler, alias, OpenMP, borrowed-input, native trust/release, or persistent-cache correctness rules to make a relation pass.

Simply adding more equivalent configuration spellings or increasing the CI seed count would be low-value farming. The next verification promotion should add a genuinely new selection/oracle dimension with evidence—most plausibly measured coverage-guided deterministic case selection, configuration-failure persistence through an explicitly versioned corpus extension, cross-compiler metamorphism once toolchain availability is executable and verified, or reduction-aware relations after that subsystem has stable ownership and semantics.