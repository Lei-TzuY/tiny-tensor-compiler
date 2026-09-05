from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

_SCHEMA = "ttc-ed25519-archive-attestation-v1"
_DOMAIN = b"tiny-tensor-compiler\x00native-bundle-archive-attestation-v1\x00"
_DIGEST_RE = re.compile(r"sha256:([0-9a-f]{64})\Z")
_PUBLISHER_RE = re.compile(r"ed25519:([0-9a-f]{64})\Z")
_SIGNATURE_RE = re.compile(r"[0-9a-f]{128}\Z")
_MAX_ATTESTATION_BYTES = 16 * 1024


class NativeBundleTrustError(RuntimeError):
    """Raised when a publisher attestation or caller trust policy is not satisfied."""


@dataclass(frozen=True)
class PublisherTrustPolicy:
    """Caller-pinned Ed25519 public keys plus explicit local revocations."""

    public_keys: tuple[bytes, ...]
    revoked_publishers: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        normalized_keys = tuple(_validate_public_key_bytes(key) for key in self.public_keys)
        if not normalized_keys:
            raise ValueError("publisher trust policy requires at least one public key")
        publisher_ids = tuple(publisher_id_from_public_key(key) for key in normalized_keys)
        if len(set(publisher_ids)) != len(publisher_ids):
            raise ValueError("publisher trust policy contains a duplicate public key")
        normalized_revoked = frozenset(
            normalize_publisher_id(publisher_id) for publisher_id in self.revoked_publishers
        )
        object.__setattr__(self, "public_keys", normalized_keys)
        object.__setattr__(self, "revoked_publishers", normalized_revoked)

    @property
    def trusted_publishers(self) -> tuple[str, ...]:
        return tuple(publisher_id_from_public_key(key) for key in self.public_keys)

    def public_key_for(self, publisher_id: str) -> bytes:
        normalized = normalize_publisher_id(publisher_id)
        if normalized in self.revoked_publishers:
            raise NativeBundleTrustError(f"publisher {normalized} is revoked by the trust policy")
        for public_key in self.public_keys:
            if publisher_id_from_public_key(public_key) == normalized:
                return public_key
        raise NativeBundleTrustError(f"publisher {normalized} is not trusted")


def publisher_public_key_from_private_key(private_key: bytes) -> bytes:
    """Derive the canonical raw 32-byte Ed25519 public key from one private seed."""
    key = Ed25519PrivateKey.from_private_bytes(_validate_private_key_bytes(private_key))
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def publisher_id_from_public_key(public_key: bytes) -> str:
    """Return the stable fingerprint used to address one pinned publisher key."""
    raw = _validate_public_key_bytes(public_key)
    return f"ed25519:{hashlib.sha256(raw).hexdigest()}"


def normalize_publisher_id(publisher_id: str) -> str:
    if not isinstance(publisher_id, str):
        raise TypeError("publisher id must be a string")
    match = _PUBLISHER_RE.fullmatch(publisher_id)
    if match is None:
        raise ValueError("publisher id must use canonical ed25519:<64 lowercase hex> form")
    return f"ed25519:{match.group(1)}"


def create_archive_attestation(private_key: bytes, digest: str) -> bytes:
    """Create one deterministic detached authorization for an exact archive digest."""
    normalized_digest = _normalize_digest(digest)
    private = Ed25519PrivateKey.from_private_bytes(_validate_private_key_bytes(private_key))
    public_key = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    publisher_id = publisher_id_from_public_key(public_key)
    signature = private.sign(_signature_message(normalized_digest, publisher_id))
    envelope = {
        "archive_digest": normalized_digest,
        "publisher_id": publisher_id,
        "schema": _SCHEMA,
        "signature": signature.hex(),
    }
    return (
        json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "ascii"
        )
        + b"\n"
    )


def verify_archive_attestation(
    attestation: bytes,
    digest: str,
    trust_policy: PublisherTrustPolicy,
    *,
    expected_publisher: str | None = None,
) -> str:
    """Verify schema, exact digest binding, pinned publisher trust, revocation, and signature."""
    if not isinstance(trust_policy, PublisherTrustPolicy):
        raise TypeError("trust_policy must be a PublisherTrustPolicy")
    normalized_digest = _normalize_digest(digest)
    envelope = _decode_attestation(attestation)
    if envelope["archive_digest"] != normalized_digest:
        raise NativeBundleTrustError("publisher attestation archive digest does not match")

    publisher_id = normalize_publisher_id(envelope["publisher_id"])
    if expected_publisher is not None:
        expected = normalize_publisher_id(expected_publisher)
        if publisher_id != expected:
            raise NativeBundleTrustError(
                f"publisher attestation identity mismatch: expected {expected}, found {publisher_id}"
            )

    public_key = trust_policy.public_key_for(publisher_id)
    signature_hex = envelope["signature"]
    if _SIGNATURE_RE.fullmatch(signature_hex) is None:
        raise NativeBundleTrustError("publisher attestation signature is not canonical Ed25519 hex")
    signature = bytes.fromhex(signature_hex)
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature,
            _signature_message(normalized_digest, publisher_id),
        )
    except InvalidSignature as exc:
        raise NativeBundleTrustError("publisher attestation signature verification failed") from exc
    return publisher_id


def _decode_attestation(attestation: bytes) -> dict[str, str]:
    if not isinstance(attestation, bytes):
        raise TypeError("publisher attestation must be bytes")
    if not attestation or len(attestation) > _MAX_ATTESTATION_BYTES:
        raise NativeBundleTrustError("publisher attestation size is invalid")
    try:
        decoded: Any = json.loads(attestation.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeBundleTrustError("publisher attestation is not valid canonical JSON") from exc
    if not isinstance(decoded, dict) or set(decoded) != {
        "archive_digest",
        "publisher_id",
        "schema",
        "signature",
    }:
        raise NativeBundleTrustError("publisher attestation fields are invalid")
    if any(not isinstance(value, str) for value in decoded.values()):
        raise NativeBundleTrustError("publisher attestation fields must be strings")
    if decoded["schema"] != _SCHEMA:
        raise NativeBundleTrustError("publisher attestation schema is unsupported")
    canonical = (
        json.dumps(decoded, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        + b"\n"
    )
    if canonical != attestation:
        raise NativeBundleTrustError("publisher attestation JSON is not canonical")
    _normalize_digest(decoded["archive_digest"])
    normalize_publisher_id(decoded["publisher_id"])
    return decoded


def _signature_message(digest: str, publisher_id: str) -> bytes:
    return _DOMAIN + digest.encode("ascii") + b"\x00" + publisher_id.encode("ascii")


def _normalize_digest(digest: str) -> str:
    if not isinstance(digest, str):
        raise TypeError("archive digest must be a string")
    match = _DIGEST_RE.fullmatch(digest)
    if match is None:
        raise ValueError("archive digest must use canonical sha256:<64 lowercase hex> form")
    return f"sha256:{match.group(1)}"


def _validate_private_key_bytes(private_key: bytes) -> bytes:
    if not isinstance(private_key, bytes):
        raise TypeError("Ed25519 private key must be raw bytes")
    if len(private_key) != 32:
        raise ValueError("Ed25519 private key must be exactly 32 raw bytes")
    return private_key


def _validate_public_key_bytes(public_key: bytes) -> bytes:
    if not isinstance(public_key, bytes):
        raise TypeError("Ed25519 public key must be raw bytes")
    if len(public_key) != 32:
        raise ValueError("Ed25519 public key must be exactly 32 raw bytes")
    # Construction rejects invalid encodings if the cryptography backend ever adds
    # stronger point validation; keeping it here also locks the primitive to Ed25519.
    Ed25519PublicKey.from_public_bytes(public_key)
    return public_key
