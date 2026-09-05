from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .configuration_metamorphic import (
    NATIVE_CONFIGURATIONS,
    ConfigurationMetamorphicFailure,
    ConfigurationRunner,
    _native_configuration_runner,
    run_configuration_metamorphic_campaign,
)
from .cross_compiler_metamorphic import (
    CROSS_COMPILER_CONFIGURATIONS,
    CompilerRunner,
    CrossCompilerMetamorphicFailure,
    _native_compiler_runner,
    run_cross_compiler_metamorphic_campaign,
)
from .differential import (
    CandidateRunner,
    DifferentialFailure,
    _compare_results,
    _require_seed,
    run_differential_campaign,
)
from .metamorphic import (
    METAMORPHIC_RELATIONS,
    MetamorphicFailure,
    run_metamorphic_campaign,
)
from .repro import (
    ReproCaseError,
    ReproMismatchError,
    load_repro_case,
    replay_repro_case,
    repro_case_sha256,
)

_FORMAT_NAME = "tiny-tensor-verification-corpus"
_FORMAT_VERSION_V1 = 1
_FORMAT_VERSION_V2 = 2
_FORMAT_VERSION_V3 = 3
_UINT64_MAX = (1 << 64) - 1
_CONFIGURATION_NAMES = frozenset(configuration.name for configuration in NATIVE_CONFIGURATIONS)
_CONFIGURATION_BY_NAME = {configuration.name: configuration for configuration in NATIVE_CONFIGURATIONS}
_BASELINE_CONFIGURATION = NATIVE_CONFIGURATIONS[0].name
_COMPILER_NAMES = frozenset(configuration.name for configuration in CROSS_COMPILER_CONFIGURATIONS)
_COMPILER_BY_NAME = {configuration.name: configuration for configuration in CROSS_COMPILER_CONFIGURATIONS}
_BASELINE_COMPILER = CROSS_COMPILER_CONFIGURATIONS[0].name


class VerificationCorpusError(ValueError):
    """Raised when a deterministic verification corpus is malformed or unverifiable."""


@dataclass(frozen=True)
class VerificationCorpusEntry:
    """One deduplicated minimized compiler failure plus deterministic seed provenance."""

    kind: str
    signature: str
    relation: str | None
    witness_seeds: tuple[int, ...]
    repros: tuple[str, ...]
    entry_sha256: str
    baseline_configuration: str | None = None
    failing_configuration: str | None = None
    baseline_compiler: str | None = None
    failing_compiler: str | None = None


@dataclass(frozen=True)
class VerificationCorpus:
    """Canonical ordered collection of minimized deterministic verification failures."""

    entries: tuple[VerificationCorpusEntry, ...]


@dataclass(frozen=True)
class VerificationCorpusReplayResult:
    """Summary of one fail-closed corpus replay."""

    entry_count: int
    repro_count: int


def collect_differential_corpus(
    *,
    start_seed: int,
    cases: int,
    candidate_runner: CandidateRunner | None = None,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    parallel: bool = False,
) -> VerificationCorpus:
    """Run every requested differential seed and retain deduplicated minimized failures."""
    first_seed = _require_campaign_range(start_seed, cases)
    entries: list[VerificationCorpusEntry] = []
    for offset in range(cases):
        result = run_differential_campaign(
            start_seed=first_seed + offset,
            cases=1,
            candidate_runner=candidate_runner,
            compiler=compiler,
            cache_dir=cache_dir,
            parallel=parallel,
        )
        if result.failure is not None:
            entries.append(_entry_from_differential_failure(result.failure))
    return VerificationCorpus(_merge_entries(entries))


def collect_metamorphic_corpus(
    *,
    start_seed: int,
    cases: int,
    candidate_runner: CandidateRunner | None = None,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    parallel: bool = False,
) -> VerificationCorpus:
    """Run every requested metamorphic seed and retain deduplicated minimized failures."""
    first_seed = _require_campaign_range(start_seed, cases)
    entries: list[VerificationCorpusEntry] = []
    for offset in range(cases):
        result = run_metamorphic_campaign(
            start_seed=first_seed + offset,
            cases=1,
            candidate_runner=candidate_runner,
            compiler=compiler,
            cache_dir=cache_dir,
            parallel=parallel,
        )
        if result.failure is not None:
            entries.append(_entry_from_metamorphic_failure(result.failure))
    return VerificationCorpus(_merge_entries(entries))


def collect_configuration_corpus(
    *,
    start_seed: int,
    cases: int,
    configuration_runner: ConfigurationRunner | None = None,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> VerificationCorpus:
    """Persist deduplicated failures from the verified native-configuration oracle."""
    first_seed = _require_campaign_range(start_seed, cases)
    entries: list[VerificationCorpusEntry] = []
    for offset in range(cases):
        result = run_configuration_metamorphic_campaign(
            start_seed=first_seed + offset,
            cases=1,
            configuration_runner=configuration_runner,
            compiler=compiler,
            cache_dir=cache_dir,
        )
        if result.failure is not None:
            entries.append(_entry_from_configuration_failure(result.failure))
    return VerificationCorpus(_merge_entries(entries))


def collect_cross_compiler_corpus(
    *,
    start_seed: int,
    cases: int,
    compiler_runner: CompilerRunner | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> VerificationCorpus:
    """Persist deduplicated failures from the canonical GCC/Clang compiler oracle."""
    first_seed = _require_campaign_range(start_seed, cases)
    entries: list[VerificationCorpusEntry] = []
    for offset in range(cases):
        result = run_cross_compiler_metamorphic_campaign(
            start_seed=first_seed + offset,
            cases=1,
            compiler_runner=compiler_runner,
            cache_dir=cache_dir,
        )
        if result.failure is not None:
            entries.append(_entry_from_cross_compiler_failure(result.failure))
    return VerificationCorpus(_merge_entries(entries))


def merge_verification_corpora(*corpora: VerificationCorpus) -> VerificationCorpus:
    """Merge corpora by canonical failure identity and union sorted witness seeds."""
    entries: list[VerificationCorpusEntry] = []
    for corpus in corpora:
        if not isinstance(corpus, VerificationCorpus):
            raise TypeError("merge_verification_corpora requires VerificationCorpus values")
        entries.extend(corpus.entries)
    return VerificationCorpus(_merge_entries(entries))


def serialize_verification_corpus(corpus: VerificationCorpus) -> str:
    """Serialize one corpus as canonical versioned JSON.

    Pure differential/metamorphic corpora retain the historical byte-compatible
    version-1 format. Configuration-specific entries promote the document to
    version 2, while compiler-pair entries explicitly promote it to version 3.
    """
    if not isinstance(corpus, VerificationCorpus):
        raise TypeError("serialize_verification_corpus requires a VerificationCorpus")
    entries = _merge_entries(corpus.entries)
    if any(entry.kind == "compiler" for entry in entries):
        version = _FORMAT_VERSION_V3
    elif any(entry.kind == "configuration" for entry in entries):
        version = _FORMAT_VERSION_V2
    else:
        version = _FORMAT_VERSION_V1
    payload = {
        "entries": [_entry_payload(entry) for entry in entries],
        "format": _FORMAT_NAME,
        "version": version,
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def load_verification_corpus(document: str) -> VerificationCorpus:
    """Decode and fail-closed validate one canonical deterministic verification corpus."""
    payload = _parse_document(document)
    _require_exact_keys(payload, {"entries", "format", "version"}, "verification corpus")
    if payload["format"] != _FORMAT_NAME:
        raise VerificationCorpusError(
            f"unsupported verification corpus format: {payload['format']!r}"
        )
    version = payload["version"]
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version not in {_FORMAT_VERSION_V1, _FORMAT_VERSION_V2, _FORMAT_VERSION_V3}
    ):
        raise VerificationCorpusError(f"unsupported verification corpus version: {version!r}")

    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list):
        raise VerificationCorpusError("verification corpus entries must be a list")

    entries: list[VerificationCorpusEntry] = []
    seen: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        entry = _entry_from_payload(raw_entry, index, version)
        if entry.entry_sha256 in seen:
            raise VerificationCorpusError(
                f"verification corpus contains duplicate entry {entry.entry_sha256}"
            )
        seen.add(entry.entry_sha256)
        entries.append(entry)

    has_configuration = any(entry.kind == "configuration" for entry in entries)
    has_compiler = any(entry.kind == "compiler" for entry in entries)
    if version == _FORMAT_VERSION_V1 and (has_configuration or has_compiler):
        raise VerificationCorpusError(
            "verification corpus version 1 cannot contain configuration or compiler entries"
        )
    if version == _FORMAT_VERSION_V2 and has_compiler:
        raise VerificationCorpusError("verification corpus version 2 cannot contain compiler entries")
    if version == _FORMAT_VERSION_V2 and not has_configuration:
        raise VerificationCorpusError("verification corpus version 2 requires a configuration entry")
    if version == _FORMAT_VERSION_V3 and not has_compiler:
        raise VerificationCorpusError("verification corpus version 3 requires a compiler entry")

    corpus = VerificationCorpus(tuple(entries))
    canonical = serialize_verification_corpus(corpus)
    if canonical != document:
        raise VerificationCorpusError("verification corpus is not canonical")
    return corpus


def verification_corpus_sha256(document: str) -> str:
    """Return the SHA-256 identity of one canonical verified corpus document."""
    corpus = load_verification_corpus(document)
    canonical = serialize_verification_corpus(corpus)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_verification_corpus(
    path: str | os.PathLike[str],
    corpus: VerificationCorpus,
) -> str:
    """Persist exactly one canonical corpus document and return its content digest."""
    document = serialize_verification_corpus(corpus)
    Path(path).write_text(document, encoding="utf-8")
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def load_verification_corpus_file(path: str | os.PathLike[str]) -> VerificationCorpus:
    """Load one corpus document from UTF-8 text storage."""
    return load_verification_corpus(Path(path).read_text(encoding="utf-8"))


def replay_verification_corpus(
    document: str,
    *,
    backend: str = "reference",
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    parallel: bool = False,
) -> VerificationCorpusReplayResult:
    """Replay every minimized corpus artifact through one supported backend.

    Native replay of configuration/compiler entries executes their stored
    canonical pair. Every execution must now agree with the captured reference
    result, making a fixed historical divergence a deterministic regression gate.
    Global ``parallel`` applies only to ordinary differential/metamorphic entries.
    """
    if backend not in {"reference", "native"}:
        raise ValueError("backend must be 'reference' or 'native'")
    if backend == "reference" and (compiler is not None or cache_dir is not None or parallel):
        raise ValueError("compiler, cache_dir, and parallel are native-backend options")

    corpus = load_verification_corpus(document)
    has_compiler_entries = any(entry.kind == "compiler" for entry in corpus.entries)
    if backend == "native" and has_compiler_entries and compiler is not None:
        raise ValueError("compiler override is incompatible with stored compiler corpus entries")

    configuration_runner = None
    if backend == "native" and any(entry.kind == "configuration" for entry in corpus.entries):
        configuration_runner = _native_configuration_runner(
            compiler=compiler,
            cache_dir=cache_dir,
        )
    compiler_runner = None
    if backend == "native" and has_compiler_entries:
        compiler_runner = _native_compiler_runner(cache_dir=cache_dir)

    repro_count = 0
    for entry in corpus.entries:
        if backend == "native" and entry.kind == "configuration":
            assert configuration_runner is not None
            _replay_configuration_entry(entry, configuration_runner)
            repro_count += 1
            continue
        if backend == "native" and entry.kind == "compiler":
            assert compiler_runner is not None
            _replay_compiler_entry(entry, compiler_runner)
            repro_count += 1
            continue

        for repro in entry.repros:
            replay_repro_case(
                repro,
                backend=backend,
                compiler=compiler,
                cache_dir=cache_dir,
                parallel=parallel,
            )
            repro_count += 1
    return VerificationCorpusReplayResult(
        entry_count=len(corpus.entries),
        repro_count=repro_count,
    )


def _entry_from_differential_failure(failure: DifferentialFailure) -> VerificationCorpusEntry:
    return _make_entry(
        kind="differential",
        signature=failure.signature,
        relation=None,
        witness_seeds=(failure.seed,),
        repros=(failure.minimized_repro,),
    )


def _entry_from_metamorphic_failure(failure: MetamorphicFailure) -> VerificationCorpusEntry:
    return _make_entry(
        kind="metamorphic",
        signature=failure.signature,
        relation=failure.relation,
        witness_seeds=(failure.seed,),
        repros=(failure.minimized_baseline_repro, failure.minimized_transformed_repro),
    )


def _entry_from_configuration_failure(
    failure: ConfigurationMetamorphicFailure,
) -> VerificationCorpusEntry:
    return _make_entry(
        kind="configuration",
        signature=failure.signature,
        relation=None,
        witness_seeds=(failure.seed,),
        repros=(failure.minimized_repro,),
        baseline_configuration=failure.baseline_configuration,
        failing_configuration=failure.failing_configuration,
    )


def _entry_from_cross_compiler_failure(
    failure: CrossCompilerMetamorphicFailure,
) -> VerificationCorpusEntry:
    return _make_entry(
        kind="compiler",
        signature=failure.signature,
        relation=None,
        witness_seeds=(failure.seed,),
        repros=(failure.minimized_repro,),
        baseline_compiler=failure.baseline_compiler,
        failing_compiler=failure.failing_compiler,
    )


def _make_entry(
    *,
    kind: str,
    signature: str,
    relation: str | None,
    witness_seeds: Sequence[int],
    repros: Sequence[str],
    baseline_configuration: str | None = None,
    failing_configuration: str | None = None,
    baseline_compiler: str | None = None,
    failing_compiler: str | None = None,
) -> VerificationCorpusEntry:
    normalized_seeds = tuple(sorted(set(witness_seeds)))
    normalized_repros = tuple(repros)
    repro_digests = tuple(_require_canonical_repro(repro) for repro in normalized_repros)
    identity = _entry_identity(
        kind,
        signature,
        relation,
        repro_digests,
        baseline_configuration,
        failing_configuration,
        baseline_compiler,
        failing_compiler,
    )
    entry = VerificationCorpusEntry(
        kind=kind,
        signature=signature,
        relation=relation,
        witness_seeds=normalized_seeds,
        repros=normalized_repros,
        entry_sha256=identity,
        baseline_configuration=baseline_configuration,
        failing_configuration=failing_configuration,
        baseline_compiler=baseline_compiler,
        failing_compiler=failing_compiler,
    )
    return _validate_entry(entry)


def _validate_entry(entry: VerificationCorpusEntry) -> VerificationCorpusEntry:
    if not isinstance(entry, VerificationCorpusEntry):
        raise TypeError("verification corpus entries must be VerificationCorpusEntry values")
    if entry.kind not in {"differential", "metamorphic", "configuration", "compiler"}:
        raise VerificationCorpusError(f"unsupported verification corpus entry kind: {entry.kind!r}")
    if not isinstance(entry.signature, str) or not entry.signature:
        raise VerificationCorpusError("verification corpus entry signature must be non-empty text")

    seeds = entry.witness_seeds
    if not isinstance(seeds, tuple) or not seeds:
        raise VerificationCorpusError("verification corpus witness seeds must be a non-empty tuple")
    if tuple(sorted(set(seeds))) != seeds:
        raise VerificationCorpusError("verification corpus witness seeds must be sorted and unique")
    for seed in seeds:
        try:
            _require_seed(seed)
        except (TypeError, ValueError) as exc:
            raise VerificationCorpusError(f"invalid verification corpus witness seed: {seed!r}") from exc

    if not isinstance(entry.repros, tuple):
        raise VerificationCorpusError("verification corpus repros must be a tuple")
    if entry.kind == "differential":
        _require_no_configuration_metadata(entry)
        _require_no_compiler_metadata(entry)
        if entry.relation is not None:
            raise VerificationCorpusError("differential corpus entries must not carry a relation")
        if len(entry.repros) != 1:
            raise VerificationCorpusError("differential corpus entries require exactly one repro")
        if entry.signature.startswith(("metamorphic:", "configuration:", "compiler:")):
            raise VerificationCorpusError("differential corpus signature uses another oracle namespace")
    elif entry.kind == "metamorphic":
        _require_no_configuration_metadata(entry)
        _require_no_compiler_metadata(entry)
        if entry.relation not in METAMORPHIC_RELATIONS:
            raise VerificationCorpusError(
                f"unsupported metamorphic corpus relation: {entry.relation!r}"
            )
        if len(entry.repros) != 2:
            raise VerificationCorpusError("metamorphic corpus entries require exactly two repros")
        prefix = f"metamorphic:{entry.relation}:"
        if not entry.signature.startswith(prefix):
            raise VerificationCorpusError(
                "metamorphic corpus signature does not match its relation"
            )
    elif entry.kind == "configuration":
        _require_no_compiler_metadata(entry)
        if entry.relation is not None:
            raise VerificationCorpusError("configuration corpus entries must not carry a relation")
        if len(entry.repros) != 1:
            raise VerificationCorpusError("configuration corpus entries require exactly one repro")
        if entry.baseline_configuration != _BASELINE_CONFIGURATION:
            raise VerificationCorpusError(
                "configuration corpus baseline must be the canonical serial-copied configuration"
            )
        if entry.failing_configuration not in _CONFIGURATION_NAMES:
            raise VerificationCorpusError(
                f"unsupported failing native configuration: {entry.failing_configuration!r}"
            )
        prefix = (
            f"configuration:{entry.baseline_configuration}->{entry.failing_configuration}:"
        )
        if not entry.signature.startswith(prefix):
            raise VerificationCorpusError(
                "configuration corpus signature does not match its stored configuration pair"
            )
    else:
        _require_no_configuration_metadata(entry)
        if entry.relation is not None:
            raise VerificationCorpusError("compiler corpus entries must not carry a relation")
        if len(entry.repros) != 1:
            raise VerificationCorpusError("compiler corpus entries require exactly one repro")
        if entry.baseline_compiler != _BASELINE_COMPILER:
            raise VerificationCorpusError("compiler corpus baseline must be the canonical gcc compiler")
        if entry.failing_compiler not in _COMPILER_NAMES:
            raise VerificationCorpusError(
                f"unsupported failing compiler: {entry.failing_compiler!r}"
            )
        prefix = f"compiler:{entry.baseline_compiler}->{entry.failing_compiler}:"
        if not entry.signature.startswith(prefix):
            raise VerificationCorpusError(
                "compiler corpus signature does not match its stored compiler pair"
            )

    repro_digests = tuple(_require_canonical_repro(repro) for repro in entry.repros)
    if entry.kind == "metamorphic":
        _require_equivalent_expected_outputs(entry.repros)
    expected_identity = _entry_identity(
        entry.kind,
        entry.signature,
        entry.relation,
        repro_digests,
        entry.baseline_configuration,
        entry.failing_configuration,
        entry.baseline_compiler,
        entry.failing_compiler,
    )
    if entry.entry_sha256 != expected_identity:
        raise VerificationCorpusError(
            "verification corpus entry SHA-256 mismatch: "
            f"expected {expected_identity}, found {entry.entry_sha256}"
        )
    return entry


def _require_no_configuration_metadata(entry: VerificationCorpusEntry) -> None:
    if entry.baseline_configuration is not None or entry.failing_configuration is not None:
        raise VerificationCorpusError(
            f"{entry.kind} corpus entries must not carry native configuration metadata"
        )


def _require_no_compiler_metadata(entry: VerificationCorpusEntry) -> None:
    if entry.baseline_compiler is not None or entry.failing_compiler is not None:
        raise VerificationCorpusError(
            f"{entry.kind} corpus entries must not carry compiler-pair metadata"
        )


def _merge_entries(entries: Sequence[VerificationCorpusEntry]) -> tuple[VerificationCorpusEntry, ...]:
    merged: dict[str, VerificationCorpusEntry] = {}
    for raw_entry in entries:
        entry = _validate_entry(raw_entry)
        previous = merged.get(entry.entry_sha256)
        if previous is None:
            merged[entry.entry_sha256] = entry
            continue
        if (
            previous.kind != entry.kind
            or previous.signature != entry.signature
            or previous.relation != entry.relation
            or previous.repros != entry.repros
            or previous.baseline_configuration != entry.baseline_configuration
            or previous.failing_configuration != entry.failing_configuration
            or previous.baseline_compiler != entry.baseline_compiler
            or previous.failing_compiler != entry.failing_compiler
        ):
            raise VerificationCorpusError(
                f"verification corpus identity collision: {entry.entry_sha256}"
            )
        merged[entry.entry_sha256] = VerificationCorpusEntry(
            kind=entry.kind,
            signature=entry.signature,
            relation=entry.relation,
            witness_seeds=tuple(sorted(set(previous.witness_seeds + entry.witness_seeds))),
            repros=entry.repros,
            entry_sha256=entry.entry_sha256,
            baseline_configuration=entry.baseline_configuration,
            failing_configuration=entry.failing_configuration,
            baseline_compiler=entry.baseline_compiler,
            failing_compiler=entry.failing_compiler,
        )
    return tuple(merged[identity] for identity in sorted(merged))


def _entry_payload(entry: VerificationCorpusEntry) -> dict[str, Any]:
    validated = _validate_entry(entry)
    payload: dict[str, Any] = {
        "entry_sha256": validated.entry_sha256,
        "kind": validated.kind,
        "relation": validated.relation,
        "repros": list(validated.repros),
        "signature": validated.signature,
        "witness_seeds": list(validated.witness_seeds),
    }
    if validated.kind == "configuration":
        payload["baseline_configuration"] = validated.baseline_configuration
        payload["failing_configuration"] = validated.failing_configuration
    elif validated.kind == "compiler":
        payload["baseline_compiler"] = validated.baseline_compiler
        payload["failing_compiler"] = validated.failing_compiler
    return payload


def _entry_from_payload(raw: Any, index: int, version: int) -> VerificationCorpusEntry:
    record = _require_mapping(raw, f"verification corpus entry #{index}")
    kind = record.get("kind")
    expected_keys = {
        "entry_sha256",
        "kind",
        "relation",
        "repros",
        "signature",
        "witness_seeds",
    }
    if version >= _FORMAT_VERSION_V2 and kind == "configuration":
        expected_keys |= {"baseline_configuration", "failing_configuration"}
    if version == _FORMAT_VERSION_V3 and kind == "compiler":
        expected_keys |= {"baseline_compiler", "failing_compiler"}
    _require_exact_keys(record, expected_keys, f"verification corpus entry #{index}")

    raw_seeds = record["witness_seeds"]
    if not isinstance(raw_seeds, list):
        raise VerificationCorpusError(
            f"verification corpus entry #{index} witness_seeds must be a list"
        )
    raw_repros = record["repros"]
    if not isinstance(raw_repros, list) or any(not isinstance(item, str) for item in raw_repros):
        raise VerificationCorpusError(
            f"verification corpus entry #{index} repros must be a list of strings"
        )
    entry_sha256 = record["entry_sha256"]
    if not _is_sha256(entry_sha256):
        raise VerificationCorpusError(
            f"verification corpus entry #{index} SHA-256 must be 64 lowercase hexadecimal characters"
        )
    entry = VerificationCorpusEntry(
        kind=record["kind"],
        signature=record["signature"],
        relation=record["relation"],
        witness_seeds=tuple(raw_seeds),
        repros=tuple(raw_repros),
        entry_sha256=entry_sha256,
        baseline_configuration=record.get("baseline_configuration"),
        failing_configuration=record.get("failing_configuration"),
        baseline_compiler=record.get("baseline_compiler"),
        failing_compiler=record.get("failing_compiler"),
    )
    return _validate_entry(entry)


def _entry_identity(
    kind: str,
    signature: str,
    relation: str | None,
    repro_digests: tuple[str, ...],
    baseline_configuration: str | None = None,
    failing_configuration: str | None = None,
    baseline_compiler: str | None = None,
    failing_compiler: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "kind": kind,
        "relation": relation,
        "repro_sha256": list(repro_digests),
        "signature": signature,
    }
    if kind == "configuration":
        payload["baseline_configuration"] = baseline_configuration
        payload["failing_configuration"] = failing_configuration
    elif kind == "compiler":
        payload["baseline_compiler"] = baseline_compiler
        payload["failing_compiler"] = failing_compiler
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _replay_configuration_entry(
    entry: VerificationCorpusEntry,
    runner: ConfigurationRunner,
) -> None:
    repro = entry.repros[0]
    case = load_repro_case(repro)
    names = [entry.baseline_configuration]
    if entry.failing_configuration != entry.baseline_configuration:
        names.append(entry.failing_configuration)

    for name in names:
        assert name is not None
        configuration = _CONFIGURATION_BY_NAME[name]
        actual = runner(configuration, case.module, case.inputs)
        mismatch = _compare_results(actual, case.expected_outputs)
        if mismatch is not None:
            raise ReproMismatchError(
                f"configuration corpus replay {name} diverged from captured reference: {mismatch}"
            )


def _replay_compiler_entry(
    entry: VerificationCorpusEntry,
    runner: CompilerRunner,
) -> None:
    repro = entry.repros[0]
    case = load_repro_case(repro)
    names = [entry.baseline_compiler]
    if entry.failing_compiler != entry.baseline_compiler:
        names.append(entry.failing_compiler)

    for name in names:
        assert name is not None
        configuration = _COMPILER_BY_NAME[name]
        actual = runner(configuration, case.module, case.inputs)
        mismatch = _compare_results(actual, case.expected_outputs)
        if mismatch is not None:
            raise ReproMismatchError(
                f"compiler corpus replay {name} diverged from captured reference: {mismatch}"
            )


def _require_canonical_repro(document: str) -> str:
    if not isinstance(document, str):
        raise VerificationCorpusError("verification corpus repro artifact must be a string")
    try:
        load_repro_case(document)
        payload = json.loads(document)
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if canonical != document:
            raise VerificationCorpusError("repro artifact is not canonical")
        return repro_case_sha256(document)
    except VerificationCorpusError:
        raise
    except (ReproCaseError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VerificationCorpusError(f"invalid repro artifact: {exc}") from exc


def _require_equivalent_expected_outputs(repros: tuple[str, ...]) -> None:
    baseline = load_repro_case(repros[0]).expected_outputs
    transformed = load_repro_case(repros[1]).expected_outputs
    if len(baseline) != len(transformed):
        raise VerificationCorpusError("metamorphic repro pair has different expected output counts")
    for left, right in zip(baseline, transformed, strict=True):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise VerificationCorpusError(
                "metamorphic repro pair has different expected output shape or dtype"
            )
        left_bytes = np.array(left, copy=True, order="C").tobytes(order="C")
        right_bytes = np.array(right, copy=True, order="C").tobytes(order="C")
        if left_bytes != right_bytes:
            raise VerificationCorpusError(
                "metamorphic repro pair has different expected output bits"
            )


def _require_campaign_range(start_seed: int, cases: int) -> int:
    first_seed = _require_seed(start_seed)
    if not isinstance(cases, int) or isinstance(cases, bool):
        raise TypeError("cases must be an integer")
    if cases <= 0:
        raise ValueError("cases must be positive")
    if first_seed + cases - 1 > _UINT64_MAX:
        raise ValueError("seed campaign exceeds the 64-bit seed range")
    return first_seed


def _parse_document(document: str) -> Mapping[str, Any]:
    if not isinstance(document, str):
        raise TypeError("verification corpus document must be a JSON string")
    try:
        payload = json.loads(document, object_pairs_hook=_object_without_duplicates)
    except json.JSONDecodeError as exc:
        raise VerificationCorpusError(f"invalid verification corpus JSON: {exc.msg}") from exc
    return _require_mapping(payload, "verification corpus")


def _require_mapping(raw: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise VerificationCorpusError(f"{context} must be a JSON object")
    if any(not isinstance(key, str) for key in raw):
        raise VerificationCorpusError(f"{context} keys must be strings")
    return raw


def _require_exact_keys(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(raw)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise VerificationCorpusError(
            f"{context} has unexpected keys: missing={missing}, extra={extra}"
        )


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationCorpusError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
