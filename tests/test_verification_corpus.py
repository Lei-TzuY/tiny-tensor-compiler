from __future__ import annotations

import json

import numpy as np
import pytest

from tiny_tensor_compiler import execute_reference
from tiny_tensor_compiler.repro import load_repro_case
from tiny_tensor_compiler.verification_corpus import (
    VerificationCorpusError,
    collect_differential_corpus,
    collect_metamorphic_corpus,
    load_verification_corpus,
    load_verification_corpus_file,
    merge_verification_corpora,
    replay_verification_corpus,
    save_verification_corpus,
    serialize_verification_corpus,
    verification_corpus_sha256,
)


def _failing_runner(_module, _inputs):
    raise RuntimeError("synthetic deterministic failure")


def _failing_transformed_runner(module, inputs):
    if module.function.name.endswith("_transformed"):
        raise RuntimeError("synthetic transformed failure")
    return execute_reference(module, inputs=inputs)


def test_differential_collection_deduplicates_same_minimized_failure():
    corpus = collect_differential_corpus(
        start_seed=4,
        cases=2,
        candidate_runner=_failing_runner,
    )

    assert len(corpus.entries) == 1
    entry = corpus.entries[0]
    assert entry.kind == "differential"
    assert entry.signature == "exception:builtins.RuntimeError"
    assert entry.relation is None
    assert entry.witness_seeds == (4, 5)
    assert len(entry.repros) == 1

    minimized = load_repro_case(entry.repros[0])
    assert [op.opcode for op in minimized.module.function.ops] == ["input", "input", "return"]
    assert minimized.inputs[0].shape == (0, 0)
    assert minimized.inputs[0].dtype == np.dtype(np.int32)


def test_metamorphic_corpora_merge_same_relation_failure_deterministically():
    first = collect_metamorphic_corpus(
        start_seed=23,
        cases=1,
        candidate_runner=_failing_transformed_runner,
    )
    second = collect_metamorphic_corpus(
        start_seed=27,
        cases=1,
        candidate_runner=_failing_transformed_runner,
    )

    merged = merge_verification_corpora(first, second)

    assert len(first.entries) == 1
    assert len(second.entries) == 1
    assert len(merged.entries) == 1
    entry = merged.entries[0]
    assert entry.kind == "metamorphic"
    assert entry.relation == "relu_idempotence"
    assert entry.signature == (
        "metamorphic:relu_idempotence:transformed-exception:builtins.RuntimeError"
    )
    assert entry.witness_seeds == (23, 27)
    assert len(entry.repros) == 2


def test_corpus_serialization_and_file_round_trip_are_canonical(tmp_path):
    corpus = collect_differential_corpus(
        start_seed=4,
        cases=2,
        candidate_runner=_failing_runner,
    )

    document = serialize_verification_corpus(corpus)
    loaded = load_verification_corpus(document)
    path = tmp_path / "verification-corpus.json"
    saved_digest = save_verification_corpus(path, corpus)

    assert loaded == corpus
    assert path.read_text(encoding="utf-8") == document
    assert load_verification_corpus_file(path) == corpus
    assert saved_digest == verification_corpus_sha256(document)
    assert saved_digest == verification_corpus_sha256(serialize_verification_corpus(loaded))


def test_corpus_loader_rejects_duplicate_tampered_and_noncanonical_entries():
    corpus = collect_differential_corpus(
        start_seed=4,
        cases=2,
        candidate_runner=_failing_runner,
    )
    document = serialize_verification_corpus(corpus)
    payload = json.loads(document)

    duplicate = dict(payload)
    duplicate["entries"] = [payload["entries"][0], payload["entries"][0]]
    with pytest.raises(VerificationCorpusError, match="duplicate entry"):
        load_verification_corpus(
            json.dumps(duplicate, sort_keys=True, separators=(",", ":"))
        )

    tampered = json.loads(document)
    tampered["entries"][0]["entry_sha256"] = "0" * 64
    with pytest.raises(VerificationCorpusError, match="SHA-256 mismatch"):
        load_verification_corpus(
            json.dumps(tampered, sort_keys=True, separators=(",", ":"))
        )

    with pytest.raises(VerificationCorpusError, match="not canonical"):
        load_verification_corpus(json.dumps(payload, indent=2, sort_keys=True))

    noncanonical_repro = json.loads(document)
    repro = noncanonical_repro["entries"][0]["repros"][0]
    noncanonical_repro["entries"][0]["repros"][0] = " " + repro
    with pytest.raises(VerificationCorpusError, match="repro artifact is not canonical"):
        load_verification_corpus(
            json.dumps(noncanonical_repro, sort_keys=True, separators=(",", ":"))
        )


def test_mixed_corpus_replays_reference_and_native():
    differential = collect_differential_corpus(
        start_seed=4,
        cases=1,
        candidate_runner=_failing_runner,
    )
    metamorphic = collect_metamorphic_corpus(
        start_seed=23,
        cases=1,
        candidate_runner=_failing_transformed_runner,
    )
    document = serialize_verification_corpus(
        merge_verification_corpora(differential, metamorphic)
    )

    reference = replay_verification_corpus(document, backend="reference")
    native = replay_verification_corpus(document, backend="native")

    assert reference.entry_count == 2
    assert reference.repro_count == 3
    assert native == reference


def test_clean_collection_is_empty_and_configuration_fails_closed():
    def reference_runner(module, inputs):
        return execute_reference(module, inputs=inputs)

    differential = collect_differential_corpus(
        start_seed=0,
        cases=3,
        candidate_runner=reference_runner,
    )
    metamorphic = collect_metamorphic_corpus(
        start_seed=0,
        cases=3,
        candidate_runner=reference_runner,
    )

    assert differential.entries == ()
    assert metamorphic.entries == ()

    with pytest.raises(TypeError, match="cases"):
        collect_differential_corpus(start_seed=0, cases=True)
    with pytest.raises(ValueError, match="positive"):
        collect_metamorphic_corpus(start_seed=0, cases=0)
    with pytest.raises(ValueError, match="64-bit"):
        collect_differential_corpus(start_seed=(1 << 64) - 1, cases=2)
    with pytest.raises(ValueError, match="backend"):
        replay_verification_corpus(
            serialize_verification_corpus(differential),
            backend="unknown",
        )
