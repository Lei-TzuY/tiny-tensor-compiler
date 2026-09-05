from __future__ import annotations

import json

import pytest

from tiny_tensor_compiler import execute_reference
from tiny_tensor_compiler.verification_corpus import (
    VerificationCorpusError,
    collect_configuration_corpus,
    collect_differential_corpus,
    load_verification_corpus,
    merge_verification_corpora,
    replay_verification_corpus,
    serialize_verification_corpus,
)


def _fail_parallel_borrowed(configuration, module, inputs):
    if configuration.name == "parallel-borrowed":
        raise RuntimeError("synthetic configuration failure")
    return execute_reference(module, inputs=inputs)


def _fail_serial_borrowed(configuration, module, inputs):
    if configuration.name == "serial-borrowed":
        raise RuntimeError("synthetic configuration failure")
    return execute_reference(module, inputs=inputs)


def _fail_candidate(_module, _inputs):
    raise RuntimeError("synthetic deterministic failure")


def test_existing_corpus_stays_byte_compatible_version_one_shape():
    corpus = collect_differential_corpus(
        start_seed=4,
        cases=1,
        candidate_runner=_fail_candidate,
    )

    payload = json.loads(serialize_verification_corpus(corpus))

    assert payload["version"] == 1
    assert set(payload) == {"entries", "format", "version"}
    assert set(payload["entries"][0]) == {
        "entry_sha256",
        "kind",
        "relation",
        "repros",
        "signature",
        "witness_seeds",
    }


def test_configuration_collection_deduplicates_and_serializes_as_version_two():
    corpus = collect_configuration_corpus(
        start_seed=4,
        cases=2,
        configuration_runner=_fail_parallel_borrowed,
    )

    assert len(corpus.entries) == 1
    entry = corpus.entries[0]
    assert entry.kind == "configuration"
    assert entry.relation is None
    assert entry.baseline_configuration == "serial-copied"
    assert entry.failing_configuration == "parallel-borrowed"
    assert entry.signature == (
        "configuration:serial-copied->parallel-borrowed:exception:builtins.RuntimeError"
    )
    assert entry.witness_seeds == (4, 5)
    assert len(entry.repros) == 1

    document = serialize_verification_corpus(corpus)
    payload = json.loads(document)
    assert payload["version"] == 2
    assert payload["entries"][0]["baseline_configuration"] == "serial-copied"
    assert payload["entries"][0]["failing_configuration"] == "parallel-borrowed"
    assert load_verification_corpus(document) == corpus


def test_configuration_pair_participates_in_failure_identity_and_merge():
    serial_borrowed = collect_configuration_corpus(
        start_seed=4,
        cases=1,
        configuration_runner=_fail_serial_borrowed,
    )
    parallel_borrowed = collect_configuration_corpus(
        start_seed=4,
        cases=1,
        configuration_runner=_fail_parallel_borrowed,
    )

    assert serial_borrowed.entries[0].repros == parallel_borrowed.entries[0].repros
    assert serial_borrowed.entries[0].entry_sha256 != parallel_borrowed.entries[0].entry_sha256
    assert len(merge_verification_corpora(serial_borrowed, parallel_borrowed).entries) == 2


def test_mixed_v1_and_configuration_corpus_promotes_document_without_rehashing_v1_entry():
    differential = collect_differential_corpus(
        start_seed=4,
        cases=1,
        candidate_runner=_fail_candidate,
    )
    original_identity = differential.entries[0].entry_sha256
    configuration = collect_configuration_corpus(
        start_seed=4,
        cases=1,
        configuration_runner=_fail_parallel_borrowed,
    )

    merged = merge_verification_corpora(differential, configuration)
    document = serialize_verification_corpus(merged)
    loaded = load_verification_corpus(document)

    assert json.loads(document)["version"] == 2
    assert len(loaded.entries) == 2
    differential_entry = next(entry for entry in loaded.entries if entry.kind == "differential")
    assert differential_entry.entry_sha256 == original_identity
    assert differential_entry.baseline_configuration is None
    assert differential_entry.failing_configuration is None


def test_v2_loader_fails_closed_on_configuration_metadata_and_version_mismatch():
    corpus = collect_configuration_corpus(
        start_seed=4,
        cases=1,
        configuration_runner=_fail_parallel_borrowed,
    )
    payload = json.loads(serialize_verification_corpus(corpus))

    unknown = json.loads(json.dumps(payload))
    unknown["entries"][0]["failing_configuration"] = "unknown-mode"
    with pytest.raises(VerificationCorpusError, match="unsupported failing native configuration"):
        load_verification_corpus(json.dumps(unknown, sort_keys=True, separators=(",", ":")))

    v1 = collect_differential_corpus(
        start_seed=4,
        cases=1,
        candidate_runner=_fail_candidate,
    )
    promoted_without_configuration = json.loads(serialize_verification_corpus(v1))
    promoted_without_configuration["version"] = 2
    with pytest.raises(VerificationCorpusError, match="version 2 requires a configuration entry"):
        load_verification_corpus(
            json.dumps(promoted_without_configuration, sort_keys=True, separators=(",", ":"))
        )


def test_configuration_corpus_replays_reference_and_stored_native_pair():
    corpus = collect_configuration_corpus(
        start_seed=4,
        cases=1,
        configuration_runner=_fail_parallel_borrowed,
    )
    document = serialize_verification_corpus(corpus)

    reference = replay_verification_corpus(document, backend="reference")
    native = replay_verification_corpus(document, backend="native")

    assert reference.entry_count == 1
    assert reference.repro_count == 1
    assert native == reference


def test_configuration_collection_fails_closed_on_ambiguous_runner_options():
    with pytest.raises(ValueError, match="compiler and cache_dir"):
        collect_configuration_corpus(
            start_seed=0,
            cases=1,
            configuration_runner=_fail_parallel_borrowed,
            compiler="cc",
        )
