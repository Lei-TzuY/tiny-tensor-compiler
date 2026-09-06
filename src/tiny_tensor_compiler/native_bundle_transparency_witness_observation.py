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
    normalize_publisher_id,
    publisher_id_from_public_key,
    publisher_public_key_from_private_key,
)
from .native_bundle_release import _canonical_json
from .native_bundle_transparency import (
    TransparencyCheckpoint,
    TransparencyStateStore,
    verify_transparency_checkpoint,
    verify_transparency_consistency,
)
from .native_bundle_transparency_witness import (
    NativeBundleTransparencyWitnessError,
    TransparencyWitnessPolicy,
)

_OBSERVATION_SCHEMA = "ttc-release-transparency-witness-observation-v1"
_OBSERVATION_DOMAIN = b"tiny-tensor-compiler\x00release-transparency-witness-observation-v1\x00"
_DIGEST_RE = re.compile(r"sha256:([0-9a-f]{64})\Z")
_SIGNATURE_RE = re.compile(r"[0-9a-f]{128}\Z")
_MAX_OBSERVATION_BYTES = 64 * 1024


@dataclass(frozen=True)
class TransparencyWitnessObservation:
    """One verified witness-signed observation of a signed log checkpoint."""

    encoded_observation: bytes
    policy_id: str
    witness_id: str
    checkpoint_digest: str
    encoded_checkpoint: bytes
    checkpoint: TransparencyCheckpoint


@dataclass(frozen=True)
class TransparencyWitnessComparison:
    """Deterministic relationship between two independently signed observations."""

    relation: str
    policy_id: str
    log_id: str
    first: TransparencyWitnessObservation
    second: TransparencyWitnessObservation

    def __post_init__(self) -> None:
        if self.relation not in {"same_checkpoint", "same_size_fork", "consistent_growth"}:
            raise ValueError("unsupported transparency witness comparison relation")
        if self.first.witness_id == self.second.witness_id:
            raise ValueError("cross-witness comparison requires distinct witnesses")


def create_transparency_witness_observation(
    private_key: bytes,
    policy: TransparencyWitnessPolicy,
    encoded_checkpoint: bytes,
    *,
    log_public_key: bytes,
    state_store: TransparencyStateStore,
) -> bytes:
    """Sign the exact checkpoint bytes only after this witness has persisted them."""
    policy = _require_policy(policy)
    if not isinstance(state_store, TransparencyStateStore):
        raise TypeError("state_store must be a TransparencyStateStore")

    checkpoint = verify_transparency_checkpoint(encoded_checkpoint, log_public_key)
    if state_store.log_id != checkpoint.log_id:
        raise NativeBundleTransparencyWitnessError(
            "witness state store uses a different pinned log operator"
        )

    current = state_store.current()
    if current is None:
        raise NativeBundleTransparencyWitnessError(
            "witness state has no accepted checkpoint to observe"
        )
    if current != checkpoint:
        raise NativeBundleTransparencyWitnessError(
            "witness current checkpoint does not match observation checkpoint"
        )
    # Recheck under the store's own lock immediately before signing. If another actor
    # advanced the store since ``current()``, this historical checkpoint is rejected.
    state_store.precheck(checkpoint)

    private = _private_key(private_key)
    public_key = publisher_public_key_from_private_key(private_key)
    witness_id = publisher_id_from_public_key(public_key)
    policy.public_key_for(witness_id)
    digest = _checkpoint_digest(encoded_checkpoint)
    signature = private.sign(_observation_message(policy.policy_id, witness_id, digest))
    encoded = _canonical_json(
        {
            "checkpoint": encoded_checkpoint.decode("ascii"),
            "checkpoint_digest": digest,
            "policy_id": policy.policy_id,
            "schema": _OBSERVATION_SCHEMA,
            "signature": signature.hex(),
            "witness_id": witness_id,
        }
    )
    if len(encoded) > _MAX_OBSERVATION_BYTES:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness observation exceeds size limit"
        )
    return encoded


def verify_transparency_witness_observation(
    encoded_observation: bytes,
    *,
    log_public_key: bytes,
    policy: TransparencyWitnessPolicy,
) -> TransparencyWitnessObservation:
    """Verify one portable signed witness observation and its embedded checkpoint."""
    policy = _require_policy(policy)
    envelope = _decode_observation(encoded_observation)
    if envelope["policy_id"] != policy.policy_id:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness observation policy identity mismatch"
        )

    witness_id = normalize_publisher_id(envelope["witness_id"])
    public_key = policy.public_key_for(witness_id)
    try:
        encoded_checkpoint = envelope["checkpoint"].encode("ascii")
    except UnicodeEncodeError as exc:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness observation checkpoint is not ASCII"
        ) from exc

    digest = _checkpoint_digest(encoded_checkpoint)
    if envelope["checkpoint_digest"] != digest:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness observation checkpoint digest mismatch"
        )
    checkpoint = verify_transparency_checkpoint(encoded_checkpoint, log_public_key)
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            bytes.fromhex(envelope["signature"]),
            _observation_message(policy.policy_id, witness_id, digest),
        )
    except InvalidSignature as exc:
        raise NativeBundleTransparencyWitnessError(
            f"transparency witness observation signature verification failed for {witness_id}"
        ) from exc

    return TransparencyWitnessObservation(
        encoded_observation=encoded_observation,
        policy_id=policy.policy_id,
        witness_id=witness_id,
        checkpoint_digest=digest,
        encoded_checkpoint=encoded_checkpoint,
        checkpoint=checkpoint,
    )


def compare_transparency_witness_observations(
    first_encoded: bytes,
    second_encoded: bytes,
    *,
    log_public_key: bytes,
    policy: TransparencyWitnessPolicy,
    consistency_proof: Sequence[bytes] = (),
) -> TransparencyWitnessComparison:
    """Compare two verified witness observations without inventing gossip freshness.

    Same-size divergent roots are explicit signed equivocation evidence. Different
    sizes are called consistent growth only after the existing RFC 6962 consistency
    verifier succeeds; a missing or invalid proof remains an ordinary verification
    failure rather than being mislabeled as a fork.
    """
    first = verify_transparency_witness_observation(
        first_encoded,
        log_public_key=log_public_key,
        policy=policy,
    )
    second = verify_transparency_witness_observation(
        second_encoded,
        log_public_key=log_public_key,
        policy=policy,
    )
    if first.witness_id == second.witness_id:
        raise NativeBundleTransparencyWitnessError(
            "cross-witness comparison requires distinct witnesses"
        )
    if first.checkpoint.log_id != second.checkpoint.log_id:
        raise NativeBundleTransparencyWitnessError(
            "cross-witness observations use different log operators"
        )

    if first.checkpoint.tree_size == second.checkpoint.tree_size:
        if consistency_proof:
            raise NativeBundleTransparencyWitnessError(
                "same-size cross-witness comparison must not provide a consistency proof"
            )
        ordered = tuple(sorted((first, second), key=lambda item: item.witness_id))
        if first.checkpoint.root_hash != second.checkpoint.root_hash:
            relation = "same_size_fork"
        else:
            if first.checkpoint_digest != second.checkpoint_digest:
                raise NativeBundleTransparencyWitnessError(
                    "same-size same-root observations bind different checkpoint bytes"
                )
            relation = "same_checkpoint"
        return TransparencyWitnessComparison(
            relation=relation,
            policy_id=policy.policy_id,
            log_id=first.checkpoint.log_id,
            first=ordered[0],
            second=ordered[1],
        )

    older, newer = sorted(
        (first, second),
        key=lambda item: item.checkpoint.tree_size,
    )
    verify_transparency_consistency(
        older.checkpoint,
        newer.checkpoint,
        consistency_proof,
    )
    return TransparencyWitnessComparison(
        relation="consistent_growth",
        policy_id=policy.policy_id,
        log_id=older.checkpoint.log_id,
        first=older,
        second=newer,
    )


def _observation_message(policy_id: str, witness_id: str, checkpoint_digest: str) -> bytes:
    return _OBSERVATION_DOMAIN + _canonical_json(
        {
            "checkpoint_digest": checkpoint_digest,
            "policy_id": policy_id,
            "schema": _OBSERVATION_SCHEMA,
            "witness_id": normalize_publisher_id(witness_id),
        }
    )


def _checkpoint_digest(encoded_checkpoint: bytes) -> str:
    if not isinstance(encoded_checkpoint, bytes):
        raise TypeError("transparency checkpoint must be bytes")
    if not encoded_checkpoint:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness observation checkpoint must not be empty"
        )
    return f"sha256:{hashlib.sha256(encoded_checkpoint).hexdigest()}"


def _decode_observation(encoded: bytes) -> dict[str, Any]:
    if not isinstance(encoded, bytes):
        raise TypeError("transparency witness observation must be bytes")
    if not encoded or len(encoded) > _MAX_OBSERVATION_BYTES:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness observation size is invalid"
        )
    try:
        decoded: Any = json.loads(
            encoded.decode("ascii"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness observation is not valid canonical JSON"
        ) from exc

    fields = {
        "checkpoint",
        "checkpoint_digest",
        "policy_id",
        "schema",
        "signature",
        "witness_id",
    }
    if not isinstance(decoded, dict) or set(decoded) != fields:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness observation fields are invalid"
        )
    if decoded.get("schema") != _OBSERVATION_SCHEMA:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness observation schema is unsupported"
        )
    if not isinstance(decoded.get("checkpoint"), str) or not decoded["checkpoint"]:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness observation checkpoint is invalid"
        )
    digest = decoded.get("checkpoint_digest")
    if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness observation checkpoint digest is invalid"
        )
    if not isinstance(decoded.get("policy_id"), str):
        raise NativeBundleTransparencyWitnessError(
            "transparency witness observation policy id is invalid"
        )
    try:
        normalize_publisher_id(decoded.get("witness_id"))
    except (TypeError, ValueError) as exc:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness observation witness id is invalid"
        ) from exc
    signature = decoded.get("signature")
    if not isinstance(signature, str) or _SIGNATURE_RE.fullmatch(signature) is None:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness observation signature is invalid"
        )
    if _canonical_json(decoded) != encoded:
        raise NativeBundleTransparencyWitnessError(
            "transparency witness observation JSON is not canonical"
        )
    return decoded


def _private_key(private_key: bytes) -> Ed25519PrivateKey:
    if not isinstance(private_key, bytes):
        raise TypeError("Ed25519 private key must be raw bytes")
    if len(private_key) != 32:
        raise ValueError("Ed25519 private key must be exactly 32 raw bytes")
    return Ed25519PrivateKey.from_private_bytes(private_key)


def _require_policy(policy: TransparencyWitnessPolicy) -> TransparencyWitnessPolicy:
    if not isinstance(policy, TransparencyWitnessPolicy):
        raise TypeError("policy must be a TransparencyWitnessPolicy")
    return policy


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NativeBundleTransparencyWitnessError(
                "transparency witness observation contains duplicate object keys"
            )
        result[key] = value
    return result
