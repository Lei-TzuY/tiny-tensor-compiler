from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .native_bundle_attestation import (
    normalize_publisher_id,
    publisher_id_from_public_key,
    publisher_public_key_from_private_key,
)
from .native_bundle_release import _canonical_json, _state_lock
from .native_bundle_threshold import NativeBundleThresholdError, ThresholdReleasePolicy

_TRANSITION_SCHEMA = "ttc-threshold-policy-transition-v1"
_STATE_SCHEMA = "ttc-threshold-policy-rotation-state-v1"
_DOMAIN = b"tiny-tensor-compiler\x00threshold-policy-transition-v1\x00"
_POLICY_RE = re.compile(r"threshold-ed25519:([0-9a-f]{64})\Z")
_SIGNATURE_RE = re.compile(r"[0-9a-f]{128}\Z")
_MAX_TRANSITION_BYTES = 16 * 1024
_MAX_STATE_BYTES = 64 * 1024
_MAX_SIGNERS = 16


class NativeBundlePolicyRotationError(NativeBundleThresholdError):
    """Raised when threshold policy rotation metadata or local state is invalid."""


@dataclass(frozen=True, order=True)
class ThresholdPolicyTransition:
    """One forward authorization from a current threshold policy to a pinned next policy."""

    from_policy_id: str
    to_policy_id: str
    epoch: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "from_policy_id", _normalize_policy_id(self.from_policy_id))
        object.__setattr__(self, "to_policy_id", _normalize_policy_id(self.to_policy_id))
        object.__setattr__(self, "epoch", _validate_epoch(self.epoch))
        if self.from_policy_id == self.to_policy_id:
            raise ValueError("threshold policy transition must change policy identity")


class ThresholdPolicyRotationStateStore:
    """Caller-owned forward-only policy-id/epoch state rooted in one pinned bootstrap policy."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        bootstrap_policy: ThresholdReleasePolicy,
    ) -> None:
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError("threshold policy rotation state path must be path-like")
        self._path = Path(path).expanduser().resolve()
        if self._path.exists() and not self._path.is_file():
            raise ValueError("threshold policy rotation state path must name a file")
        self._bootstrap_policy = _require_policy(bootstrap_policy)
        # Validate any existing state immediately against the caller-pinned bootstrap anchor.
        with _state_lock(self._path):
            _read_state(self._path, self._bootstrap_policy.policy_id)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def bootstrap_policy_id(self) -> str:
        return self._bootstrap_policy.policy_id

    def current(self) -> tuple[int, str]:
        """Return the accepted rotation epoch and current policy id."""
        with _state_lock(self._path):
            return _read_state(self._path, self.bootstrap_policy_id)

    @property
    def epoch(self) -> int:
        return self.current()[0]

    @property
    def current_policy_id(self) -> str:
        return self.current()[1]

    def accept_transition(
        self,
        encoded: bytes,
        current_policy: ThresholdReleasePolicy,
        next_policy: ThresholdReleasePolicy,
    ) -> ThresholdPolicyTransition:
        """Verify and atomically persist exactly one next-epoch current->next transition."""
        current_policy = _require_policy(current_policy)
        next_policy = _require_policy(next_policy)
        with _state_lock(self._path):
            epoch, policy_id = _read_state(self._path, self.bootstrap_policy_id)
            if current_policy.policy_id != policy_id:
                raise NativeBundlePolicyRotationError(
                    "current threshold policy does not match locally accepted rotation state"
                )
            transition = verify_threshold_policy_transition(
                encoded,
                current_policy,
                next_policy,
                expected_epoch=epoch + 1,
            )
            _write_state(
                self._path,
                self.bootstrap_policy_id,
                transition.epoch,
                transition.to_policy_id,
            )
            return transition


def create_threshold_policy_transition(
    private_keys: Sequence[bytes],
    current_policy: ThresholdReleasePolicy,
    next_policy: ThresholdReleasePolicy,
    epoch: int,
) -> bytes:
    """Create canonical current-policy k-of-n authorization for one pinned next policy id."""
    current_policy = _require_policy(current_policy)
    next_policy = _require_policy(next_policy)
    transition = ThresholdPolicyTransition(
        current_policy.policy_id,
        next_policy.policy_id,
        epoch,
    )
    signatures: list[dict[str, str]] = []
    seen: set[str] = set()
    message = _transition_message(transition)
    for private_key in private_keys:
        private = _private_key(private_key)
        public_key = publisher_public_key_from_private_key(private_key)
        signer_id = publisher_id_from_public_key(public_key)
        current_policy.public_key_for(signer_id)
        if signer_id in current_policy.revoked_signers:
            raise NativeBundlePolicyRotationError(
                f"signer {signer_id} is revoked by the current threshold policy"
            )
        if signer_id in seen:
            raise NativeBundlePolicyRotationError("threshold policy transition has a duplicate signer")
        seen.add(signer_id)
        signatures.append({"signature": private.sign(message).hex(), "signer_id": signer_id})
    signatures.sort(key=lambda item: item["signer_id"])
    if len(signatures) < current_policy.threshold:
        raise NativeBundlePolicyRotationError(
            f"threshold policy transition requires at least {current_policy.threshold} "
            "distinct eligible signatures"
        )
    envelope = {
        "epoch": transition.epoch,
        "from_policy_id": transition.from_policy_id,
        "schema": _TRANSITION_SCHEMA,
        "signatures": signatures,
        "to_policy_id": transition.to_policy_id,
    }
    encoded = _canonical_json(envelope)
    if len(encoded) > _MAX_TRANSITION_BYTES:
        raise NativeBundlePolicyRotationError("threshold policy transition exceeds size limit")
    return encoded


def verify_threshold_policy_transition(
    encoded: bytes,
    current_policy: ThresholdReleasePolicy,
    next_policy: ThresholdReleasePolicy,
    *,
    expected_epoch: int | None = None,
) -> ThresholdPolicyTransition:
    """Verify canonical transition metadata and current-policy k-of-n signatures."""
    current_policy = _require_policy(current_policy)
    next_policy = _require_policy(next_policy)
    envelope = _decode_transition(encoded)
    transition = ThresholdPolicyTransition(
        envelope["from_policy_id"],
        envelope["to_policy_id"],
        envelope["epoch"],
    )
    if transition.from_policy_id != current_policy.policy_id:
        raise NativeBundlePolicyRotationError(
            "threshold policy transition predecessor does not match the pinned current policy"
        )
    if transition.to_policy_id != next_policy.policy_id:
        raise NativeBundlePolicyRotationError(
            "threshold policy transition target does not match the pinned next policy"
        )
    if expected_epoch is not None and transition.epoch != _validate_epoch(expected_epoch):
        raise NativeBundlePolicyRotationError(
            f"threshold policy transition epoch must be exactly {expected_epoch}"
        )

    message = _transition_message(transition)
    eligible = 0
    for signature in envelope["signatures"]:
        signer_id = normalize_publisher_id(signature["signer_id"])
        public_key = current_policy.public_key_for(signer_id)
        if signer_id in current_policy.revoked_signers:
            continue
        try:
            Ed25519PublicKey.from_public_bytes(public_key).verify(
                bytes.fromhex(signature["signature"]),
                message,
            )
        except InvalidSignature as exc:
            raise NativeBundlePolicyRotationError(
                f"threshold policy transition signature verification failed for {signer_id}"
            ) from exc
        eligible += 1
    if eligible < current_policy.threshold:
        raise NativeBundlePolicyRotationError(
            f"threshold policy transition requires {current_policy.threshold} valid "
            f"non-revoked signatures, found {eligible}"
        )
    return transition


def _decode_transition(encoded: bytes) -> dict[str, Any]:
    if not isinstance(encoded, bytes):
        raise TypeError("threshold policy transition must be bytes")
    if not encoded or len(encoded) > _MAX_TRANSITION_BYTES:
        raise NativeBundlePolicyRotationError("threshold policy transition size is invalid")
    try:
        decoded: Any = json.loads(encoded.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeBundlePolicyRotationError(
            "threshold policy transition is not valid canonical JSON"
        ) from exc
    fields = {"epoch", "from_policy_id", "schema", "signatures", "to_policy_id"}
    if not isinstance(decoded, dict) or set(decoded) != fields:
        raise NativeBundlePolicyRotationError("threshold policy transition fields are invalid")
    if decoded.get("schema") != _TRANSITION_SCHEMA:
        raise NativeBundlePolicyRotationError("threshold policy transition schema is unsupported")
    if isinstance(decoded.get("epoch"), bool) or not isinstance(decoded.get("epoch"), int):
        raise NativeBundlePolicyRotationError("threshold policy transition epoch must be an integer")
    for name in ("from_policy_id", "schema", "to_policy_id"):
        if not isinstance(decoded.get(name), str):
            raise NativeBundlePolicyRotationError(
                f"threshold policy transition {name} must be a string"
            )
    signatures = decoded.get("signatures")
    if not isinstance(signatures, list) or not signatures or len(signatures) > _MAX_SIGNERS:
        raise NativeBundlePolicyRotationError("threshold policy transition signatures are invalid")
    previous: str | None = None
    for item in signatures:
        if not isinstance(item, dict) or set(item) != {"signature", "signer_id"}:
            raise NativeBundlePolicyRotationError("threshold policy transition signature entry is invalid")
        signer = item.get("signer_id")
        signature = item.get("signature")
        if not isinstance(signer, str) or not isinstance(signature, str):
            raise NativeBundlePolicyRotationError(
                "threshold policy transition signature fields must be strings"
            )
        try:
            normalized = normalize_publisher_id(signer)
        except (TypeError, ValueError) as exc:
            raise NativeBundlePolicyRotationError(
                "threshold policy transition signer id is invalid"
            ) from exc
        if signer != normalized:
            raise NativeBundlePolicyRotationError(
                "threshold policy transition signer id is not canonical"
            )
        if _SIGNATURE_RE.fullmatch(signature) is None:
            raise NativeBundlePolicyRotationError(
                "threshold policy transition signature is not canonical Ed25519 hex"
            )
        if previous is not None and signer <= previous:
            raise NativeBundlePolicyRotationError(
                "threshold policy transition signatures must be unique and sorted by signer id"
            )
        previous = signer
    try:
        ThresholdPolicyTransition(
            decoded["from_policy_id"],
            decoded["to_policy_id"],
            decoded["epoch"],
        )
    except (TypeError, ValueError) as exc:
        raise NativeBundlePolicyRotationError("threshold policy transition metadata is invalid") from exc
    if _canonical_json(decoded) != encoded:
        raise NativeBundlePolicyRotationError("threshold policy transition JSON is not canonical")
    return decoded


def _transition_message(transition: ThresholdPolicyTransition) -> bytes:
    payload = {
        "epoch": transition.epoch,
        "from_policy_id": transition.from_policy_id,
        "schema": _TRANSITION_SCHEMA,
        "to_policy_id": transition.to_policy_id,
    }
    return _DOMAIN + _canonical_json(payload)


def _normalize_policy_id(policy_id: str) -> str:
    if not isinstance(policy_id, str):
        raise TypeError("threshold policy id must be a string")
    match = _POLICY_RE.fullmatch(policy_id)
    if match is None:
        raise ValueError(
            "threshold policy id must use canonical threshold-ed25519:<64 lowercase hex> form"
        )
    return f"threshold-ed25519:{match.group(1)}"


def _validate_epoch(epoch: int) -> int:
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise TypeError("threshold policy rotation epoch must be an integer")
    if epoch < 1:
        raise ValueError("threshold policy rotation epoch must be at least 1")
    return epoch


def _private_key(private_key: bytes) -> Ed25519PrivateKey:
    if not isinstance(private_key, bytes):
        raise TypeError("Ed25519 private key must be raw bytes")
    if len(private_key) != 32:
        raise ValueError("Ed25519 private key must be exactly 32 raw bytes")
    return Ed25519PrivateKey.from_private_bytes(private_key)


def _require_policy(policy: ThresholdReleasePolicy) -> ThresholdReleasePolicy:
    if not isinstance(policy, ThresholdReleasePolicy):
        raise TypeError("policy must be a ThresholdReleasePolicy")
    return policy


def _read_state(path: Path, bootstrap_policy_id: str) -> tuple[int, str]:
    bootstrap_policy_id = _normalize_policy_id(bootstrap_policy_id)
    if not path.exists():
        return 0, bootstrap_policy_id
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise NativeBundlePolicyRotationError("failed to read threshold policy rotation state") from exc
    if not raw or len(raw) > _MAX_STATE_BYTES:
        raise NativeBundlePolicyRotationError("threshold policy rotation state size is invalid")
    try:
        decoded: Any = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeBundlePolicyRotationError(
            "threshold policy rotation state is not valid JSON"
        ) from exc
    fields = {"bootstrap_policy_id", "current_policy_id", "epoch", "schema"}
    if not isinstance(decoded, dict) or set(decoded) != fields:
        raise NativeBundlePolicyRotationError("threshold policy rotation state fields are invalid")
    if decoded.get("schema") != _STATE_SCHEMA:
        raise NativeBundlePolicyRotationError("threshold policy rotation state schema is invalid")
    if isinstance(decoded.get("epoch"), bool) or not isinstance(decoded.get("epoch"), int):
        raise NativeBundlePolicyRotationError("threshold policy rotation state epoch is invalid")
    try:
        stored_bootstrap = _normalize_policy_id(decoded["bootstrap_policy_id"])
        current_policy_id = _normalize_policy_id(decoded["current_policy_id"])
    except (TypeError, ValueError) as exc:
        raise NativeBundlePolicyRotationError(
            "threshold policy rotation state policy identity is invalid"
        ) from exc
    if stored_bootstrap != bootstrap_policy_id:
        raise NativeBundlePolicyRotationError(
            "threshold policy rotation state bootstrap does not match the caller-pinned policy"
        )
    if decoded["epoch"] < 1:
        raise NativeBundlePolicyRotationError("persisted threshold policy rotation epoch is invalid")
    if _canonical_json(decoded) != raw:
        raise NativeBundlePolicyRotationError("threshold policy rotation state JSON is not canonical")
    return decoded["epoch"], current_policy_id


def _write_state(path: Path, bootstrap_policy_id: str, epoch: int, policy_id: str) -> None:
    payload = {
        "bootstrap_policy_id": _normalize_policy_id(bootstrap_policy_id),
        "current_policy_id": _normalize_policy_id(policy_id),
        "epoch": _validate_epoch(epoch),
        "schema": _STATE_SCHEMA,
    }
    encoded = _canonical_json(payload)
    if len(encoded) > _MAX_STATE_BYTES:
        raise NativeBundlePolicyRotationError("threshold policy rotation state exceeds size limit")
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
            directory = -1
            try:
                directory = os.open(path.parent, os.O_RDONLY)
            except OSError:
                pass
            if directory >= 0:
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise NativeBundlePolicyRotationError(
            "failed to persist threshold policy rotation state"
        ) from exc
