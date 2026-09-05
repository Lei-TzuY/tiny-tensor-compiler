# Deterministic differential repro campaigns

This phase builds directly on the canonical repro-case format introduced by the deterministic IR repro milestone. It adds a bounded seeded case generator, a reference-versus-candidate differential runner, and deterministic shrinking that emits the same canonical repro artifact used by the existing replay tooling.

The goal is reproducible compiler-correctness discovery, not coverage-guided fuzzing or performance measurement.

## Stable seed contract

`generate_differential_case(seed)` accepts an unsigned 64-bit integer and uses an implementation-owned SplitMix64 sequence. Case generation does not depend on Python's `random` module, hash randomization, process entropy, platform word size, or iteration order of unordered containers.

For one seed, the generated module, exact input values, reference output bits, and canonical serialized repro document are therefore intended to remain byte-identical across the supported Python and operating-system CI matrix for this phase.

The current bounded grammar deliberately stays on the already-stable pure, non-reduction surface:

- dtype: `i32` or `f32`;
- square static shapes with side length 0 through 4;
- one through six generated operations;
- `add`;
- `mul`;
- `relu`;
- `view`;
- `reshape`;
- `reverse` on either axis;
- 2-D transpose.

The active single-axis reduction work is intentionally excluded from this generator. Mutation (`copy_into`), release/signature policy, security boundaries, and other unrelated subsystems are also excluded rather than being sampled accidentally.

## Differential execution

`run_differential_campaign()` walks consecutive seeds in deterministic order and stops at the first failure.

The semantic oracle is `execute_reference()`. By default the candidate is the ordinary compiled native path produced by `compile_module()`, so generated cases cross optimization, Buffer IR, memory planning, Loop IR, fusion, generated C, the platform compiler, and native execution. A caller may instead inject a candidate runner for another backend or a focused regression.

Results are compared exactly:

1. output count;
2. shape;
3. dtype;
4. canonical C-order output bytes.

A discrepancy receives a stable signature such as `mismatch:shape:0` or `mismatch:bytes:0`. Candidate failures from the explicitly classified runtime/type/value/OS/arithmetic/assertion families receive an exception-type signature such as `exception:builtins.RuntimeError`.

The harness does not catch `Exception` blindly. Unknown infrastructure or control-flow failures propagate instead of being mislabeled as compiler repros.

## Deterministic shrinking

When one seed fails, shrinking preserves the exact failure signature. Candidates that merely fail differently are rejected.

The shrink order is fixed:

1. greedily delete generated operations in program order;
2. try smaller square shapes from zero upward;
3. try replacing each complete runtime input with zeros;
4. replace remaining nonzero input elements with zero in input order and row-major element order.

The minimized result is serialized with `capture_repro_case()`. It is therefore directly consumable by `load_repro_case()` / `replay_repro_case()` and inherits the existing module fingerprint, exact input, and reference-output validation rules instead of defining a second repro format.

## Correctness regression discovered by the campaign

During development, seed 4 generated a legal graph containing a view and adjacent integer multiplies. Reference execution succeeded, but native compilation failed after elementwise fusion with:

`loop kernels do not permit output/input storage aliasing`

The Loop verifier was correct. The fusion planner still used the historical buffer-ID test when deciding whether the composed fused output aliased a leaf input. After the view/alias phases, distinct logical value IDs may share one physical storage root, so ID inequality is no longer sufficient evidence of non-aliasing.

The planner now evaluates composed binary-DAG and trailing-ReLU output/input safety by storage-root identity. If the final output root equals any fused leaf root, fusion is refused and the original individually verified kernels remain. The Loop verifier was not weakened.

A direct native replay regression for the generated seed-4 artifact remains in the suite so this integration bug cannot silently return.

## Validation boundary

The production head `61fe8bba9c85a706a81d54c35ea481159b90548e` passed CI #830 / run `33981407094` on Ubuntu and Windows with Python 3.11 and 3.13. Ruff and the full test suite passed in every matrix cell; Ubuntu Python 3.11 reported 428 passing tests.

This evidence establishes deterministic generation, deterministic same-signature shrinking, canonical repro integration, native differential execution, and the storage-root-aware fusion regression fix. It does **not** establish a fuzz-coverage percentage, bug-discovery rate, security-fuzzing claim, performance improvement, or statistical quality claim.

## Phase boundary

The next testing promotion should add genuinely new verification leverage rather than simply increasing the seed count or enumerating more already-covered operations. Candidates include deterministic multi-failure corpus deduplication/minimization, metamorphic relations that do not require a second backend, or carefully extending the generator into new compiler surfaces after their ownership and semantics have stabilized.