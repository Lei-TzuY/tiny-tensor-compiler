from __future__ import annotations

import json
import shutil
import sys

import numpy as np
import pytest

from tiny_tensor_compiler import execute_reference
from tiny_tensor_compiler.verification_corpus import (
    VerificationCorpusError,
    collect_configuration_corpus,
    collect_cross_compiler_corpus,
    collect_differential_corpus,
    load_verification_corpus,
    merge_verification_corpora,
    replay_verification_corpus,
    serialize_verification_corpus,
)


def _fail_clang(configuration, module, inputs):
    result = execute_reference(module, inputs=inputs)
    if configuration.name == "clang":
        return np.zeros_like(np.asarray(result))
    return result


def _fail_gcc(configuration, module, inputs):
    if configuration.name == "gcc":
        raise RuntimeError("synthetic baseline compiler failure")
    return execute_reference(module, inputs=inputs)


def _fail_parallel_borrowed(configuration, module, inputs):
    if configuration.name == "parallel-borrowed":
        raise RuntimeError("synthetic configuration failure")
    return execute_reference(module, inputs=inputs)


def _fail_candidate(_module, _inputs):
    raise RuntimeError("synthetic deterministic failure")


def test_cross_compiler_collection_deduplicates_and_serializes_as_version_three():
    corpus = collect_cross_compiler_corpus(
        start_seed=5,
        cases=2,
        compiler_runner=_fail_clang,
    )

    assert len(corpus.entries) == 1
    entry = corpus.entries[0]
    assert entry.kind == "compiler"
    assert entry.relation is None
    assert entry.baseline_compiler == "gcc"
    assert entry.failing_compiler == "clang"
    assert entry.signature.startswith("compiler:gcc->clang:mismatch:")
    assert entry.witness_seeds == (5, 6)
    assert len(entry.repros) == 1

    document = serialize_verification_corpus(corpus)
    payload = json.loads(document)
    assert payload["version"] == 3
    assert payload["entries"][0]["baseline_compiler"] == "gcc"
    assert payload["entries"][0]["failing_compiler"] == "clang"
    assert load_verification_corpus(document) == corpus


def test_compiler_pair_participates_in_failure_identity_and_merge():
    gcc_failure = collect_cross_compiler_corpus(
        start_seed=5,
        cases=1,
        compiler_runner=_fail_gcc,
    )
    clang_failure = collect_cross_compiler_corpus(
        start_seed=5,
        cases=1,
        compiler_runner=_fail_clang,
    )

    assert gcc_failure.entries[0].repros == clang_failure.entries[0].repros
    assert gcc_failure.entries[0].entry_sha256 != clang_failure.entries[0].entry_sha256
    assert len(merge_verification_corpora(gcc_failure, clang_failure).entries) == 2


def test_mixed_v1_v2_v3_corpus_promotes_without_rehashing_older_entries():
    differential = collect_differential_corpus(
        start_seed=4,
        cases=1,
        candidate_runner=_fail_candidate,
    )
    configuration = collect_configuration_corpus(
        start_seed=4,
        cases=1,
        configuration_runner=_fail_parallel_borrowed,
    )
    compiler = collect_cross_compiler_corpus(
        start_seed=5,
        cases=1,
        compiler_runner=_fail_clang,
    )
    old_identities = {
        differential.entries[0].entry_sha256,
        configuration.entries[0].entry_sha256,
    }

    merged = merge_verification_corpora(differential, configuration, compiler)
    document = serialize_verification_corpus(merged)
    loaded = load_verification_corpus(document)

    assert json.loads(document)["version"] == 3
    loaded_old_identities = {
        entry.entry_sha256 for entry in loaded.entries if entry.kind != "compiler"
    }
    assert loaded_old_identities == old_identities


def test_v3_loader_fails_closed_on_compiler_metadata_and_version_mismatch():
    corpus = collect_cross_compiler_corpus(
        start_seed=5,
        cases=1,
        compiler_runner=_fail_clang,
    )
    payload = json.loads(serialize_verification_corpus(corpus))

    unknown = json.loads(json.dumps(payload))
    unknown["entries"][0]["failing_compiler"] = "unknown-compiler"
    with pytest.raises(VerificationCorpusError, match="unsupported failing compiler"):
        load_verification_corpus(json.dumps(unknown, sort_keys=True, separators=(",", ":")))

    wrong_baseline = json.loads(json.dumps(payload))
    wrong_baseline["entries"][0]["baseline_compiler"] = "clang"
    with pytest.raises(VerificationCorpusError, match="baseline must be the canonical gcc compiler"):
        load_verification_corpus(
            json.dumps(wrong_baseline, sort_keys=True, separators=(",", ":"))
        )

    v2_without_compiler = collect_configuration_corpus(
        start_seed=4,
        cases=1,
        configuration_runner=_fail_parallel_borrowed,
    )
    promoted_without_compiler = json.loads(serialize_verification_corpus(v2_without_compiler))
    promoted_without_compiler["version"] = 3
    with pytest.raises(VerificationCorpusError, match="version 3 requires a compiler entry"):
        load_verification_corpus(
            json.dumps(promoted_without_compiler, sort_keys=True, separators=(",", ":"))
        )


def test_reference_replay_accepts_compiler_corpus_without_native_toolchains():
    corpus = collect_cross_compiler_corpus(
        start_seed=5,
        cases=1,
        compiler_runner=_fail_clang,
    )
    result = replay_verification_corpus(
        serialize_verification_corpus(corpus),
        backend="reference",
    )

    assert result.entry_count == 1
    assert result.repro_count == 1


@pytest.mark.skipif(sys.platform == "win32", reason="GCC/Clang corpus replay evidence is verified on Ubuntu CI")
def test_compiler_corpus_replays_stored_pair_through_real_gcc_and_clang(tmp_path):
    assert shutil.which("gcc") is not None, "Ubuntu compiler-corpus replay requires gcc"
    assert shutil.which("clang") is not None, "Ubuntu compiler-corpus replay requires clang"
    corpus = collect_cross_compiler_corpus(
        start_seed=5,
        cases=1,
        compiler_runner=_fail_clang,
    )

    result = replay_verification_corpus(
        serialize_verification_corpus(corpus),
        backend="native",
        cache_dir=tmp_path,
    )

    assert result.entry_count == 1
    assert result.repro_count == 1


def test_compiler_corpus_rejects_ambiguous_collection_and_replay_options(tmp_path):
    with pytest.raises(ValueError, match="cache_dir"):
        collect_cross_compiler_corpus(
            start_seed=5,
            cases=1,
            compiler_runner=_fail_clang,
            cache_dir=tmp_path,
        )

    corpus = collect_cross_compiler_corpus(
        start_seed=5,
        cases=1,
        compiler_runner=_fail_clang,
    )
    with pytest.raises(ValueError, match="compiler override"):
        replay_verification_corpus(
            serialize_verification_corpus(corpus),
            backend="native",
            compiler="cc",
        )
