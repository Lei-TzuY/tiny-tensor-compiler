# Deterministic metamorphic verification

This phase adds a second executable compiler-verification oracle on top of the deterministic differential/repro infrastructure.

The existing differential campaign compares one generated module against the tensor-IR reference evaluator. The metamorphic campaign instead constructs **two different IR programs that are required to be semantically equivalent**, validates that relation once with reference semantics, and then compares the two executions of the candidate backend directly. Candidate-vs-candidate comparison can therefore expose lowering, alias, optimization, code-generation, or native-execution regressions whose two sides happen to share the same reference input/output contract.

## Deterministic generation

`generate_metamorphic_case(seed)` reuses the exact bounded `_CaseSpec` grammar and 64-bit seed normalization from the differential generator. Relation selection uses a separate domain-separated SplitMix64 stream, so adding or reordering metamorphic relations cannot perturb the byte identity of an existing `generate_differential_case(seed)` artifact.

The bounded relation set is:

- reverse axis 0 twice;
- reverse axis 1 twice;
- transpose twice;
- an identity whole-storage view after an explicit owning C-order materialization;
- flatten-and-reshape round trip;
- ReLU idempotence;
- `reverse(0) -> transpose` commuting with `transpose -> reverse(1)` for the square generated domain.

Each side is serialized with the existing canonical repro format. A `MetamorphicCase` therefore carries two independently loadable/replayable artifacts rather than introducing another persistence schema.

## Oracle boundary

Before a relation is admitted to candidate comparison, both modules are executed with `execute_reference()` and compared with the same exact shape/dtype/raw-bit comparison used by the differential harness. A reference mismatch is a harness error and raises immediately; it is not reported as a candidate failure.

After reference validation, the ordinary candidate path runs both modules. The default is the existing native compiler path, so a campaign crosses optimization, Buffer IR, storage-root/layout planning, Loop IR, fusion, generated C, the platform C compiler, and native execution. An injected candidate runner can be used for focused regressions.

A candidate failure is reported only when:

- one side raises one of the explicitly classified candidate exceptions; or
- both sides execute but differ in output arity, shape, dtype, or raw output bits.

Failure signatures include the relation name and the failing side where appropriate, for example `metamorphic:<relation>:transformed-exception:<type>` or `metamorphic:<relation>:mismatch:<kind>`. Exception messages and native temporary paths are deliberately excluded from the signature.

## Deterministic shrinking

The first failing seed is minimized while preserving the exact relation-aware failure signature. Shrinking uses the same deterministic order as the differential campaign:

1. greedily delete generated operations left-to-right;
2. try smaller square side lengths;
3. try zeroing complete inputs;
4. try zeroing individual elements in stable input/flat-index order.

The relation itself is held fixed during shrinking. Both the original and minimized baseline/transformed programs are emitted through the canonical repro schema, so a minimized pair remains independently replayable and fingerprintable.

This is a reproducer reducer, not a proof of global minimality.

## Correctness issue exposed by the new oracle

The first native campaign exposed a real storage-layout bug on deterministic seed 2. The generated graph contained a singleton `(1, 1)` tensor whose axis had been reversed before a whole-storage view. Its layout was `StorageLayout(offset=0, strides=(-1, 1))`.

The old `StorageLayout.is_contiguous()` implementation required the stride tuple to be byte-for-byte equal to canonical C strides. That is stricter than logical C-contiguity: a size-one axis contributes no movement, so its stride value cannot change element order; an empty tensor has no element order at all.

The corrected predicate therefore:

- ignores stride values on axes whose extent is exactly one;
- treats zero-element shapes as contiguous once rank is consistent;
- still requires every extent-greater-than-one axis to have its exact canonical C stride.

Focused regressions retain rejection of negative or otherwise incorrect strides on non-singleton axes. This fix allows the seed-2 singleton view to be reshaped without materialization while preserving the existing prohibition on zero-copy reshape of genuinely non-contiguous storage.

## Evidence and exclusions

Production head CI #844 (`33982828032`) passed Ruff and the full pytest matrix on Ubuntu and Windows with Python 3.11 and 3.13. Ubuntu Python 3.11 completed **441 tests**. The matrix includes direct replay of the seed-2 baseline/transformed native artifacts and the bounded native metamorphic campaign.

This phase intentionally does not:

- add reduction operations; the independent `feat/single-axis-sum` surface owns that work;
- add mutation/copy-into relations;
- claim randomized fuzzing or broad statistical coverage;
- schedule an unbounded seed campaign in CI;
- weaken view-generation, storage-root, overlap, alias-lifetime, or native trust/release invariants;
- claim performance improvements.

## Phase promotion

Adding more algebraically similar relation spellings or simply increasing the seed count would be low-value farming. The next verification promotion should add a genuinely new testing dimension—such as persistent curated repro corpora, coverage-guided campaign selection, cross-configuration/compiler metamorphism, or reduction-aware relations after the reduction branch converges—while new compiler features should continue to follow their independently owned architectural surfaces.
