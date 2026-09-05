from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .native_bundle_attestation import (
    PublisherTrustPolicy,
    normalize_publisher_id,
    publisher_id_from_public_key,
    publisher_public_key_from_private_key,
)
from .native_bundle_release import _canonical_json
from .native_bundle_transparency import (
    NativeBundleTransparencyError,
    TransparencyCheckpoint,
    TransparencyStateStore,
    accept_release_transparency,
)

_POLICY_SCHEMA = "ttc-release-transparency-witness-policy-v1"
_QUORUM_SCHEMA = "ttc-release-transparency-witness-quorum-v1"
_DOMAIN = b"tiny-tensor-compiler\x00release-transparency-witness-quorum-v1\x00"
_POLICY_RE = re.compile(r"transparency-witness-ed25519:([0-9a-f]{64})\Z")
_DIGEST_RE = re.compile(r"sha256:([0-9a-f]{64})\Z")
_SIGNATURE_RE = re.compile(r"[0-9a-f]{128}\Z")
_MAX_WITNESSES = 16
_MAX_CHECKPOINT_BYTES = 16 * 1024
_MAX_QUORUM_BYTES = 32 * 1024


class NativeBundleTransparencyWitnessError(NativeBundleTransparencyError):
    """Raised when caller-pinned transparency witness evidence is invalid."""


@dataclass(frozen=True)
class TransparencyWitnessPolicy:
    """Caller-pinned k-of-n Ed25519 witnesses for one exact log checkpoint."""

    public_keys: tuple[bytes, ...]
    threshold: int
    revoked_witnesses: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, int):
            raise TypeError("witness threshold must be an integer")
        base = PublisherTrustPolicy(tuple(self.public_keys))
        ordered = tuple(sorted(base.public_keys, key=publisher_id_from_public_key))
        if len(ordered) > _MAX_WITNESSES:
            raise ValueError(f"witness policy supports at most {_MAX_WITNESSES} keys")
        if self.threshold < 1 or self.threshold > len(ordered):
            raise ValueError("witness threshold must be between 1 and the number of witness keys")
        revoked = frozenset(normalize_publisher_id(value) for value in self.revoked_witnesses)
        witness_ids = tuple(publisher_id_from_public_key(key) for key in ordered)
        if len([witness for witness in witness_ids if witness not in revoked]) < self.threshold:
            raise ValueError("witness revocations leave fewer eligible keys than the threshold")
        object.__setattr__(self, "public_keys", ordered)
        object.__setattr__(self, "revoked_witnesses", revoked)

    @property
    def witness_ids(self) -> tuple[str, ...]:
        return tuple(publisher_id_from_public_key(key) for key in self.public_keys)

    @property
    def policy_id(self) -> str:
        descriptor = {
            "schema": _POLICY_SCHEMA,
            "threshold": self.threshold,
            "witnesses": list(self.witness_ids),
        }
        return f"transparency-witness-ed25519:{hashlib.sha256(_canonical_json(descriptor)).hexdigest()}"

    @property
    def eligible_witnesses(self) -> tuple[str, ...]:
        return tuple(
            witness for witness in self.witness_ids if witness not in self.revoked_witnesses
        )

    def public_key_for(self, witness_id: str) -> bytes:
        normalized = normalize_publisher_id(witness_id)
        if normalized in self.revoked_witnesses:
            raise NativeBundleTransparencyWitnessError(
                f"witness {normalized} is revoked by the witness policy"
            )
        return self._public_key_for_member(normalized)

    def _public_key_for_member(self, witness_id: str) -> bytes:
        normalized = normalize_publisher_id(witness_id)
        for public_key in self.public_keys:
            if publisher_id_from_public_key(public_key) == normalized:
                return public_key
        raise NativeBundleTransparencyWitnessError(
            f"witness {normalized} is not in the witness policy"
        )


@dataclass(frozen=True)
class TransparencyWitnessQuorum:
    """Verified witness endorsement metadata for one exact checkpoint digest."""

    policy_id: str
    checkpoint_digest: str
    witness_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _normalize_policy_id(self.policy_id))
        object.__setattr__(self, "checkpoint_digest", _normalize_digest(self.checkpoint_digest))
        normalized = tuple(normalize_publisher_id(value) for value in self.witness_ids)
        if normalized != tuple(sorted(normalized)) or len(set(normalized)) != len(normalized):
            raise ValueError("verified witness ids must be unique and sorted")
        object.__setattr__(self, "witness_ids", normalized)


def create_transparency_witness_quorum(
    private_keys: Sequence[bytes],
    policy: TransparencyWitnessPolicy,
    encoded_checkpoint: bytes,
) -> bytes:
    """Create canonical k-of-n witness endorsements for exact checkpoint bytes."""
    policy = _require_policy(policy)
    digest = _checkpoint_digest(encoded_checkpoint)
    message = _witness_message(policy.policy_id, digest)
    signatures: list[dict[str, str]] = []
    seen: set[str] = set()

    for private_key in private_keys:
        private = _private_key(private_key)
        public_key = publisher_public_key_from_private_key(private_key)
        witness_id = publisher_id_from_public_key(public_key)
        policy._public_key_for_member(witness_id)
        if witness_id in policy.revoked_witnesses:
            raise NativeBundleTransparencyWitnessError(
                f"witness {witness_id} is revoked by the witness policy"
            )
        if witness_id in seen:
            raise NativeBundleTransparencyWitnessError(
                "transparency witness quorum contains a duplicate witness"
            )
        seen.add(witness_id)
        signatures.append(
            {
                "signature": private.sign(message).hex(),
                "witness_id": witness_id,
            }
        )

    signatures.sort(key=lambda item: item["witness_id"])
    if len(signatures) < policy.threshold:
        raise NativeBundleTransparencyWitnessError(
            f"transparency witness quorum requires at least {policy.threshold} distinct eligible signatures"
        )
    encoded = _canonical_json(
        {
            "checkpoint_digest": digest,
            "policy_id": policy.policy_id,
            "schema": _QUORUM_SCHEMA,
            "signatures": signatures,
        }
    )
    if len(encoded) > _MAX_QUORUM_BYTES:
        raise NativeBundleTransparencyWitnessError("transparency witness quorum exceeds size limit")
    return encoded


def verify_transparency_witness_quorum(
    encoded_quorum: bytes,
    encoded_checkpoint: bytes,
    policy: TransparencyWitnessPolicy,
) -> TransparencyWitnessQuorum:
    """Verify k distinct pinned witnesses over one exact transparency checkpoint."""
    policy = _require_policy(policy)
    expected_digest = _checkpoint_digest(encoded_checkpoint)
    envelope = _decode_quorum(encoded_quorum)
    policy_id = _normalize_policy_id(envelope["policy_id"])
    if policy_id != policy.policy_id:
        raise NativeBundleTransparencyWitnessError(
            f"transparency witness policy identity mismatch: expected {policy.policy_id}, found {policy_id}"
        )
    digest = _normalize_digest(envelope["checkpoint_digest"])
    if digest != expected_digest:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness checkpoint digest does not match exact checkpoint bytes"
        )

    message = _witness_message(policy_id, digest)
    eligible: list[str] = []
    for item in envelope["signatures"]:
        witness_id = normalize_publisher_id(item["witness_id"])
        public_key = policy._public_key_for_member(witness_id)
        try:
            Ed25519PublicKey.from_public_bytes(public_key).verify(
                bytes.fromhex(item["signature"]),
                message,
            )
        except InvalidSignature as exc:
            raise NativeBundleTransparencyWitnessError(
                f"transparency witness signature verification failed for {witness_id}"
            ) from exc
        if witness_id not in policy.revoked_witnesses:
            eligible.append(witness_id)

    if len(eligible) < policy.threshold:
        raise NativeBundleTransparencyWitnessError(
            f"transparency witness quorum requires {policy.threshold} valid non-revoked signatures, found {len(eligible)}"
        )
    return TransparencyWitnessQuorum(policy_id, digest, tuple(eligible))


def accept_witnessed_release_transparency(
    release_checkpoint_bytes: bytes,
    *,
    leaf_index: int,
    encoded_checkpoint: bytes,
    log_public_key: bytes,
    inclusion_proof: Sequence[bytes],
    consistency_proof: Sequence[bytes],
    state_store: TransparencyStateStore,
    encoded_witness_quorum: bytes,
    witness_policy: TransparencyWitnessPolicy,
) -> TransparencyCheckpoint:
    """Require witness quorum evidence before advancing local append-only log state.

    Witness endorsements prove only that the configured pinned witness keys signed the
    exact log-checkpoint bytes under this policy. They do not prove that witnesses ran
    independent consistency checks or exchanged gossip with other clients.
    """
    verify_transparency_witness_quorum(
        encoded_witness_quorum,
        encoded_checkpoint,
        witness_policy,
    )
    return accept_release_transparency(
        release_checkpoint_bytes,
        leaf_index=leaf_index,
        encoded_checkpoint=encoded_checkpoint,
        log_public_key=log_public_key,
        inclusion_proof=inclusion_proof,
        consistency_proof=consistency_proof,
        state_store=state_store,
    )


def _witness_message(policy_id: str, checkpoint_digest: str) -> bytes:
    return _DOMAIN + _canonical_json(
        {
            "checkpoint_digest": _normalize_digest(checkpoint_digest),
            "policy_id": _normalize_policy_id(policy_id),
            "schema": _QUORUM_SCHEMA,
        }
    )


def _checkpoint_digest(encoded_checkpoint: bytes) -> str:
    if not isinstance(encoded_checkpoint, bytes):
        raise TypeError("transparency checkpoint must be bytes")
    if not encoded_checkpoint or len(encoded_checkpoint) > _MAX_CHECKPOINT_BYTES:
        raise NativeBundleTransparencyWitnessError(
            "transparency checkpoint size is invalid for witness endorsement"
        )
    return f"sha256:{hashlib.sha256(encoded_checkpoint).hexdigest()}"


def _decode_quorum(encoded: bytes) -> dict[str, Any]:
    if not isinstance(encoded, bytes):
        raise TypeError("transparency witness quorum must be bytes")
    if not encoded or len(encoded) > _MAX_QUORUM_BYTES:
        raise NativeBundleTransparencyWitnessError("transparency witness quorum size is invalid")
    try:
        decoded: Any = json.loads(encoded.decode("ascii"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness quorum is not valid canonical JSON"
        ) from exc
    fields = {"checkpoint_digest", "policy_id", "schema", "signatures"}
    if not isinstance(decoded, dict) or set(decoded) != fields:
        raise NativeBundleTransparencyWitnessError("transparency witness quorum fields are invalid")
    if decoded.get("schema") != _QUORUM_SCHEMA:
        raise NativeBundleTransparencyWitnessError("transparency witness quorum schema is unsupported")
    try:
        _normalize_policy_id(decoded.get("policy_id"))
        _normalize_digest(decoded.get("checkpoint_digest"))
    except (TypeError, ValueError) as exc:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness quorum identity fields are invalid"
        ) from exc

    signatures = decoded.get("signatures")
    if not isinstance(signatures, list) or not signatures or len(signatures) > _MAX_WITNESSES:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness quorum signature list is invalid"
        )
    witness_ids: list[str] = []
    for item in signatures:
        if not isinstance(item, dict) or set(item) != {"signature", "witness_id"}:
            raise NativeBundleTransparencyWitnessError(
                "transparency witness quorum signature fields are invalid"
            )
        witness_id = item.get("witness_id")
        signature = item.get("signature")
        try:
            normalized = normalize_publisher_id(witness_id)
        except (TypeError, ValueError) as exc:
            raise NativeBundleTransparencyWitnessError(
                "transparency witness id is invalid"
            ) from exc
        if not isinstance(signature, str) or _SIGNATURE_RE.fullmatch(signature) is None:
            raise NativeBundleTransparencyWitnessError(
                "transparency witness signature is not canonical Ed25519 hex"
            )
        witness_ids.append(normalized)
    if witness_ids != sorted(witness_ids) or len(set(witness_ids)) != len(witness_ids):
        raise NativeBundleTransparencyWitnessError(
            "transparency witness signatures must use unique sorted witness ids"
        )
    if _canonical_json(decoded) != encoded:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness quorum JSON is not canonical"
        )
    return decoded


def _normalize_policy_id(policy_id: str) -> str:
    if not isinstance(policy_id, str):
        raise TypeError("transparency witness policy id must be a string")
    match = _POLICY_RE.fullmatch(policy_id)
    if match is None:
        raise ValueError(
            "transparency witness policy id must use canonical transparency-witness-ed25519:<64 lowercase hex> form"
        )
    return f"transparency-witness-ed25519:{match.group(1)}"


def _normalize_digest(digest: str) -> str:
    if not isinstance(digest, str):
        raise TypeError("transparency witness checkpoint digest must be a string")
    match = _DIGEST_RE.fullmatch(digest)
    if match is None:
        raise ValueError(
            "transparency witness checkpoint digest must use canonical sha256:<64 lowercase hex> form"
        )
    return f"sha256:{match.group(1)}"


def _private_key(private_key: bytes) -> Ed25519PrivateKey:
    if not isinstance(private_key, bytes):
        raise TypeError("Ed25519 private key must be raw bytes")
    if len(private_key) != 32:
        raise ValueError("Ed25519 private key must be exactly 32 raw bytes")
    return Ed25519PrivateKey.from_private_bytes(private_key)


def _require_policy(policy: TransparencyWitnessPolicy) -> TransparencyWitnessPolicy:
    if not isinstance(policy, TransparencyWitnessPolicy):
        raise TypeError("witness_policy must be a TransparencyWitnessPolicy")
    return policy


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NativeBundleTransparencyWitnessError(
                "transparency witness JSON contains duplicate object keys"
            )
        result[key] = value
    return result
