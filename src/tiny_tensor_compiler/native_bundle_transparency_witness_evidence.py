from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .native_bundle_release import _canonical_json, _state_lock
from .native_bundle_transparency import (
    NativeBundleTransparencyRollbackError,
    log_id_from_public_key,
    verify_transparency_consistency,
)
from .native_bundle_transparency_witness import (
    NativeBundleTransparencyWitnessError,
    TransparencyWitnessPolicy,
)
from .native_bundle_transparency_witness_observation import (
    TransparencyWitnessObservation,
    verify_transparency_witness_observation,
)

_EVIDENCE_STATE_SCHEMA = "ttc-release-transparency-witness-evidence-state-v1"
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_STATE_BYTES = 2 * 1024 * 1024
_MAX_OBSERVATIONS = 16


@dataclass(frozen=True)
class TransparencyWitnessEvidenceSnapshot:
    """Caller-local durable view of verified witness observations or terminal fork evidence."""

    status: str
    policy_id: str
    log_id: str
    observations: tuple[TransparencyWitnessObservation, ...] = ()
    fork_evidence: tuple[TransparencyWitnessObservation, TransparencyWitnessObservation] | None = None

    def __post_init__(self) -> None:
        if self.status not in {"healthy", "forked"}:
            raise ValueError("unsupported transparency witness evidence state")
        if self.status == "healthy":
            if self.fork_evidence is not None:
                raise ValueError("healthy witness evidence state cannot contain fork evidence")
            witness_ids = tuple(item.witness_id for item in self.observations)
            if not witness_ids or witness_ids != tuple(sorted(witness_ids)):
                raise ValueError("healthy witness evidence observations must be non-empty and sorted")
            if len(set(witness_ids)) != len(witness_ids):
                raise ValueError("healthy witness evidence must contain one observation per witness")
            if len(self.observations) > _MAX_OBSERVATIONS:
                raise ValueError("healthy witness evidence exceeds witness limit")
            return

        if self.observations:
            raise ValueError("forked witness evidence state cannot contain healthy observations")
        if self.fork_evidence is None:
            raise ValueError("forked witness evidence state requires signed fork evidence")
        first, second = self.fork_evidence
        ordered = tuple(sorted((first, second), key=_observation_sort_key))
        if (first, second) != ordered:
            raise ValueError("fork evidence observations must be deterministically ordered")
        if first.checkpoint.log_id != second.checkpoint.log_id:
            raise ValueError("fork evidence observations use different log operators")
        if first.checkpoint.tree_size != second.checkpoint.tree_size:
            raise ValueError("fork evidence observations must have the same tree size")
        if first.checkpoint.root_hash == second.checkpoint.root_hash:
            raise ValueError("fork evidence observations must bind divergent tree roots")


class TransparencyWitnessEvidenceStore:
    """Durable caller-local cross-witness consistency memory for one pinned log and policy.

    The store does not fetch observations or consistency proofs and makes no freshness
    claim. It only remembers caller-supplied, cryptographically verified evidence under
    a cross-process lock. A missing file is first contact and carries no external
    split-view protection.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        log_public_key: bytes,
        policy: TransparencyWitnessPolicy,
    ) -> None:
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError("transparency witness evidence path must be path-like")
        if not isinstance(policy, TransparencyWitnessPolicy):
            raise TypeError("policy must be a TransparencyWitnessPolicy")
        self._path = Path(path).expanduser().resolve()
        if self._path.exists() and not self._path.is_file():
            raise ValueError("transparency witness evidence path must name a file")
        self._log_id = log_id_from_public_key(log_public_key)
        self._log_public_key = log_public_key
        self._policy = policy

    @property
    def path(self) -> Path:
        return self._path

    @property
    def log_id(self) -> str:
        return self._log_id

    @property
    def policy_id(self) -> str:
        return self._policy.policy_id

    def current(self) -> TransparencyWitnessEvidenceSnapshot | None:
        with _state_lock(self._path):
            return _read_state(
                self._path,
                log_public_key=self._log_public_key,
                policy=self._policy,
                expected_log_id=self._log_id,
            )

    def record(
        self,
        encoded_observation: bytes,
        *,
        consistency_proofs: Mapping[str, Sequence[bytes]] | None = None,
    ) -> TransparencyWitnessEvidenceSnapshot:
        observation = verify_transparency_witness_observation(
            encoded_observation,
            log_public_key=self._log_public_key,
            policy=self._policy,
        )
        proofs = _normalize_proofs(consistency_proofs)
        with _state_lock(self._path):
            previous = _read_state(
                self._path,
                log_public_key=self._log_public_key,
                policy=self._policy,
                expected_log_id=self._log_id,
            )
            if previous is not None and previous.status == "forked":
                raise NativeBundleTransparencyWitnessError(
                    "transparency witness evidence store contains terminal fork evidence"
                )

            existing = () if previous is None else previous.observations
            prior_same_witness = next(
                (item for item in existing if item.witness_id == observation.witness_id),
                None,
            )
            if (
                prior_same_witness is not None
                and observation.checkpoint.tree_size < prior_same_witness.checkpoint.tree_size
            ):
                raise NativeBundleTransparencyRollbackError(
                    "transparency witness observation rollback is below that witness's stored tree size"
                )

            if prior_same_witness is not None and (
                prior_same_witness.encoded_observation == observation.encoded_observation
            ):
                if proofs:
                    raise NativeBundleTransparencyWitnessError(
                        "transparency witness consistency proof set does not match required checkpoints"
                    )
                return previous

            fork_pair = _find_same_size_fork(existing, observation)
            if fork_pair is not None:
                snapshot = TransparencyWitnessEvidenceSnapshot(
                    status="forked",
                    policy_id=self._policy.policy_id,
                    log_id=self._log_id,
                    fork_evidence=tuple(sorted(fork_pair, key=_observation_sort_key)),
                )
                _write_state(self._path, snapshot)
                return snapshot

            for item in existing:
                if item.checkpoint.tree_size != observation.checkpoint.tree_size:
                    continue
                if item.checkpoint.root_hash != observation.checkpoint.root_hash:
                    raise RuntimeError("same-size fork should have been captured before comparison")
                if item.checkpoint_digest != observation.checkpoint_digest:
                    raise NativeBundleTransparencyWitnessError(
                        "same-size same-root observations bind different checkpoint bytes"
                    )

            proof_targets: dict[str, TransparencyWitnessObservation] = {}
            for item in existing:
                if item.checkpoint_digest == observation.checkpoint_digest:
                    continue
                if item.checkpoint.tree_size == observation.checkpoint.tree_size:
                    continue
                proof_targets.setdefault(item.checkpoint_digest, item)

            required_keys = set(proof_targets)
            if set(proofs) != required_keys:
                raise NativeBundleTransparencyWitnessError(
                    "transparency witness consistency proof set does not match required checkpoints"
                )
            for digest in sorted(proof_targets):
                other = proof_targets[digest]
                older, newer = sorted(
                    (other, observation),
                    key=lambda item: item.checkpoint.tree_size,
                )
                verify_transparency_consistency(
                    older.checkpoint,
                    newer.checkpoint,
                    proofs[digest],
                )

            by_witness = {item.witness_id: item for item in existing}
            by_witness[observation.witness_id] = observation
            snapshot = TransparencyWitnessEvidenceSnapshot(
                status="healthy",
                policy_id=self._policy.policy_id,
                log_id=self._log_id,
                observations=tuple(by_witness[key] for key in sorted(by_witness)),
            )
            _write_state(self._path, snapshot)
            return snapshot


def _find_same_size_fork(
    existing: tuple[TransparencyWitnessObservation, ...],
    observation: TransparencyWitnessObservation,
) -> tuple[TransparencyWitnessObservation, TransparencyWitnessObservation] | None:
    for item in existing:
        if item.checkpoint.tree_size != observation.checkpoint.tree_size:
            continue
        if item.checkpoint.root_hash != observation.checkpoint.root_hash:
            return item, observation
    return None


def _normalize_proofs(
    consistency_proofs: Mapping[str, Sequence[bytes]] | None,
) -> dict[str, tuple[bytes, ...]]:
    if consistency_proofs is None:
        return {}
    if not isinstance(consistency_proofs, Mapping):
        raise TypeError("consistency_proofs must be a mapping keyed by checkpoint digest")
    normalized: dict[str, tuple[bytes, ...]] = {}
    for digest, proof in consistency_proofs.items():
        if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
            raise NativeBundleTransparencyWitnessError(
                "transparency witness consistency proof key is not a checkpoint digest"
            )
        if isinstance(proof, (bytes, bytearray, str)) or not isinstance(proof, Sequence):
            raise TypeError("transparency witness consistency proof must be a sequence of hash nodes")
        nodes = tuple(proof)
        if any(not isinstance(node, bytes) for node in nodes):
            raise TypeError("transparency witness consistency proof nodes must be bytes")
        normalized[digest] = nodes
    return normalized


def _read_state(
    path: Path,
    *,
    log_public_key: bytes,
    policy: TransparencyWitnessPolicy,
    expected_log_id: str,
) -> TransparencyWitnessEvidenceSnapshot | None:
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise NativeBundleTransparencyWitnessError(
            "failed to read transparency witness evidence state"
        ) from exc
    if not raw or len(raw) > _MAX_STATE_BYTES:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness evidence state size is invalid"
        )
    try:
        decoded: Any = json.loads(raw.decode("ascii"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness evidence state is not valid JSON"
        ) from exc
    fields = {
        "fork_evidence",
        "log_id",
        "observations",
        "policy_id",
        "schema",
        "status",
    }
    if not isinstance(decoded, dict) or set(decoded) != fields:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness evidence state fields are invalid"
        )
    if decoded.get("schema") != _EVIDENCE_STATE_SCHEMA:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness evidence state schema is invalid"
        )
    if decoded.get("log_id") != expected_log_id:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness evidence state log identity mismatch"
        )
    if decoded.get("policy_id") != policy.policy_id:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness evidence state policy identity mismatch"
        )
    if decoded.get("status") not in {"healthy", "forked"}:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness evidence state status is invalid"
        )
    if _canonical_json(decoded) != raw:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness evidence state JSON is not canonical"
        )

    observations_field = decoded.get("observations")
    fork_field = decoded.get("fork_evidence")
    if not isinstance(observations_field, list):
        raise NativeBundleTransparencyWitnessError(
            "transparency witness evidence state observations are invalid"
        )
    observations = tuple(
        _verify_stored_observation(value, log_public_key=log_public_key, policy=policy)
        for value in observations_field
    )

    if decoded["status"] == "healthy":
        if fork_field is not None:
            raise NativeBundleTransparencyWitnessError(
                "healthy transparency witness evidence state contains fork evidence"
            )
        try:
            return TransparencyWitnessEvidenceSnapshot(
                status="healthy",
                policy_id=policy.policy_id,
                log_id=expected_log_id,
                observations=observations,
            )
        except ValueError as exc:
            raise NativeBundleTransparencyWitnessError(
                "transparency witness evidence state observations are invalid"
            ) from exc

    if observations:
        raise NativeBundleTransparencyWitnessError(
            "forked transparency witness evidence state contains healthy observations"
        )
    if not isinstance(fork_field, list) or len(fork_field) != 2:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness evidence state fork evidence is invalid"
        )
    fork = tuple(
        _verify_stored_observation(value, log_public_key=log_public_key, policy=policy)
        for value in fork_field
    )
    try:
        return TransparencyWitnessEvidenceSnapshot(
            status="forked",
            policy_id=policy.policy_id,
            log_id=expected_log_id,
            fork_evidence=(fork[0], fork[1]),
        )
    except ValueError as exc:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness evidence state fork evidence is invalid"
        ) from exc


def _verify_stored_observation(
    value: Any,
    *,
    log_public_key: bytes,
    policy: TransparencyWitnessPolicy,
) -> TransparencyWitnessObservation:
    if not isinstance(value, str) or not value:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness evidence state observation encoding is invalid"
        )
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness evidence state observation is not ASCII"
        ) from exc
    return verify_transparency_witness_observation(
        encoded,
        log_public_key=log_public_key,
        policy=policy,
    )


def _write_state(path: Path, snapshot: TransparencyWitnessEvidenceSnapshot) -> None:
    payload = {
        "fork_evidence": (
            None
            if snapshot.fork_evidence is None
            else [item.encoded_observation.decode("ascii") for item in snapshot.fork_evidence]
        ),
        "log_id": snapshot.log_id,
        "observations": [
            item.encoded_observation.decode("ascii") for item in snapshot.observations
        ],
        "policy_id": snapshot.policy_id,
        "schema": _EVIDENCE_STATE_SCHEMA,
        "status": snapshot.status,
    }
    encoded = _canonical_json(payload)
    if len(encoded) > _MAX_STATE_BYTES:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness evidence state exceeds size limit"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.write-",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            try:
                directory = os.open(path.parent, os.O_RDONLY)
            except OSError:
                directory = -1
            if directory >= 0:
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise NativeBundleTransparencyWitnessError(
            "failed to persist transparency witness evidence state"
        ) from exc


def _observation_sort_key(
    observation: TransparencyWitnessObservation,
) -> tuple[str, str]:
    return observation.witness_id, observation.checkpoint_digest


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NativeBundleTransparencyWitnessError(
                "transparency witness evidence state contains duplicate object keys"
            )
        result[key] = value
    return result
