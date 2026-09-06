# Reduction-aware metamorphic verification

This phase adds a deterministic semantic oracle for the existing reduction surface without changing reduction lowering, storage planning, generated C, or native execution semantics.

## Why this is a separate campaign

The historical seeded differential grammar intentionally stays on the stable pure non-reduction surface. Its seed-to-graph mapping is already part of deterministic repro and corpus workflows, so reduction coverage is added through a domain-separated campaign instead of silently changing old seed identities.

For each seed, `generate_reduction_metamorphic_case()` materializes one baseline module and one transformed module, stores both as canonical repro documents, and selects exactly one reduction relation from a dedicated relation stream. `run_reduction_metamorphic_campaign()` validates the relation with the reference interpreter first, then compares candidate execution of the two modules. A failure is shrunk only while preserving its exact relation-scoped signature.

## Supported relations

The bounded first relation set is:

- `sum_reshape_invariance`
- `prod_reshape_invariance`
- `sum_all_axes_equivalence`
- `prod_all_axes_equivalence`
- `sum_axis1_transpose_map`
- `prod_axis1_transpose_map`
- `sum_keepdims_view_equivalence`
- `prod_keepdims_view_equivalence`
- `argmax_axis1_transpose_map`
- `argmax_keepdims_view_equivalence`

These relations are deliberately chosen so the compared reductions retain the same canonical logical scan order. The campaign therefore uses exact dtype/shape/byte comparison rather than a floating-point tolerance or an associativity assumption.

The sum/prod reshape relation changes only shape, not C-order element order. Full reduction and explicit `(0, 1)` reduction visit the same rank-2 logical order. Transpose axis-map relations move the reduced axis while preserving the per-output source sequence. `keepdims=True` changes only the result view shape after the same reduction. The two argmax relations preserve both the compared source sequence and index coordinates for the mapped axis.

The campaign does **not** claim arbitrary reduction reassociation, commutativity-based reorder equivalence, tolerance-based floating-point algebra, or a general theorem prover.

## Argmax and empty domains

`argmax` requires a non-empty reduced axis. Seeds whose generated side is zero therefore cannot select an argmax relation. Deterministic shrinking also refuses candidates that reduce an argmax failure to side zero. This keeps minimization inside the valid semantic domain instead of converting the original failure into an unrelated empty-reduction exception.

## Comparator contract

Reduction verification exposed an older verification-infrastructure ceiling: the canonical differential comparator handled `i32` and `f32`, while `argmax` correctly returns `i64` indices. The shared canonical-byte comparator now also supports `i64` through explicit little-endian `<i8` normalization.

This is intentionally a narrow capability expansion for index-valued results. It does not broaden the generated differential input grammar or implicitly add other output dtypes.

## Failure and repro contract

The campaign records:

- seed and selected relation;
- original baseline/transformed canonical repro documents;
- minimized baseline/transformed canonical repro documents;
- stable relation-scoped mismatch or exception signature;
- original/minimized generated-operation counts;
- deterministic shrink evaluation count.

Reference execution must establish the relation before any candidate comparison. If the reference pair disagrees, the relation implementation itself is treated as invalid and the campaign fails immediately rather than recording a compiler candidate failure.

Candidate exceptions are normalized by relation, baseline/transformed side, and exception type. Exception message text is deliberately excluded from the stable signature.

## Native evidence

With no custom candidate runner, the campaign uses the existing native compile/execute path. The first production-inclusive candidate passed the full Ubuntu/Windows × Python 3.11/3.13 CI matrix after the shared comparator learned `i64`; Ubuntu Python 3.11 ran 697 tests successfully.

This is correctness evidence for the oracle and the existing native reduction implementations. It is not a performance, conformance-suite completeness, or statistical fuzzing claim.

## Phase boundary

This phase adds a qualitatively new reduction semantic oracle; it should not be followed by farming extra seeds or trivial relation synonyms. Future reduction verification work should add a distinct proof dimension only when the semantics are precise—for example, a later direct-matmul/reduction integration oracle after that compiler-core work converges, or configuration/cross-compiler reduction relations that preserve exact evaluation order. Mutation-aware metamorphism should likewise wait while active mutation branches still own that compiler-core surface.
