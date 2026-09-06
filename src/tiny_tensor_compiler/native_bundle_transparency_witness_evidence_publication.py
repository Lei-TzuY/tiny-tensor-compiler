from __future__ import annotations

import http.client
import json
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .native_bundle_attestation import (
    normalize_publisher_id,
    publisher_id_from_public_key,
    publisher_public_key_from_private_key,
)
from .native_bundle_release import _canonical_json
from .native_bundle_transparency_witness import TransparencyWitnessPolicy
from .native_bundle_transparency_witness_evidence import (
    TransparencyWitnessEvidenceSnapshot,
    TransparencyWitnessEvidenceStore,
)
from .native_bundle_transparency_witness_observation import (
    TransparencyWitnessObservation,
    verify_transparency_witness_observation,
)

_PUBLICATION_SCHEMA = "ttc-release-transparency-witness-evidence-publication-v1"
_PUBLICATION_DOMAIN = b"tiny-tensor-compiler\x00release-transparency-witness-evidence-publication-v1\x00"
_PUBLICATION_MEDIA_TYPE = (
    "application/vnd.tiny-tensor-compiler.transparency-witness-publication+json"
)
_DEFAULT_MAX_BYTES = 256 * 1024
_CHUNK_SIZE = 16 * 1024
_MAX_PROOFS = 16
_MAX_PROOF_NODES = 64
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_LOG_ID_RE = re.compile(r"ed25519:[0-9a-f]{64}\Z")
_NODE_RE = re.compile(r"[0-9a-f]{64}\Z")
_SIGNATURE_RE = re.compile(r"[0-9a-f]{128}\Z")


class NativeBundleTransparencyPublicationError(RuntimeError):
    """Raised when transparency evidence publication transport or verification fails."""


@dataclass(frozen=True)
class TransparencyWitnessEvidencePublication:
    """One verified publisher-signed transport envelope for witness evidence."""

    encoded_publication: bytes
    publisher_id: str
    log_id: str
    policy_id: str
    observation: TransparencyWitnessObservation
    consistency_proofs: tuple[tuple[str, tuple[bytes, ...]], ...]


def create_transparency_witness_evidence_publication(
    publisher_private_key: bytes,
    encoded_observation: bytes,
    *,
    log_public_key: bytes,
    policy: TransparencyWitnessPolicy,
    consistency_proofs: Mapping[str, Sequence[bytes]] | None = None,
) -> bytes:
    """Create one canonical signed evidence publication for external transport.

    The publication signature authenticates the transport envelope and its exact proof
    set. It does not replace verification of the embedded witness/log signatures and
    does not make any freshness claim.
    """
    policy = _require_policy(policy)
    observation = verify_transparency_witness_observation(
        encoded_observation,
        log_public_key=log_public_key,
        policy=policy,
    )
    proofs = _normalize_proofs(consistency_proofs)
    private_key = _private_key(publisher_private_key)
    publisher_public_key = publisher_public_key_from_private_key(publisher_private_key)
    publisher_id = publisher_id_from_public_key(publisher_public_key)
    body = _publication_body(
        publisher_id=publisher_id,
        log_id=observation.checkpoint.log_id,
        policy_id=policy.policy_id,
        encoded_observation=encoded_observation,
        consistency_proofs=proofs,
    )
    signature = private_key.sign(_publication_message(body))
    encoded = _canonical_json({**body, "signature": signature.hex()})
    if len(encoded) > _DEFAULT_MAX_BYTES:
        raise NativeBundleTransparencyPublicationError(
            "transparency witness evidence publication exceeds size limit"
        )
    return encoded


def verify_transparency_witness_evidence_publication(
    encoded_publication: bytes,
    *,
    publisher_public_key: bytes,
    log_public_key: bytes,
    policy: TransparencyWitnessPolicy,
) -> TransparencyWitnessEvidencePublication:
    """Verify publisher framing plus the embedded witness-signed log observation."""
    policy = _require_policy(policy)
    envelope = _decode_publication(encoded_publication)
    public_key = _public_key(publisher_public_key)
    expected_publisher_id = publisher_id_from_public_key(publisher_public_key)
    publisher_id = normalize_publisher_id(envelope["publisher_id"])
    if publisher_id != expected_publisher_id:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence publication publisher identity mismatch"
        )
    if envelope["policy_id"] != policy.policy_id:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence publication policy identity mismatch"
        )

    try:
        encoded_observation = envelope["observation"].encode("ascii")
    except UnicodeEncodeError as exc:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence publication observation is not ASCII"
        ) from exc
    observation = verify_transparency_witness_observation(
        encoded_observation,
        log_public_key=log_public_key,
        policy=policy,
    )
    if envelope["log_id"] != observation.checkpoint.log_id:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence publication log identity mismatch"
        )

    proofs = _decode_proofs(envelope["consistency_proofs"])
    body = _publication_body(
        publisher_id=publisher_id,
        log_id=envelope["log_id"],
        policy_id=policy.policy_id,
        encoded_observation=encoded_observation,
        consistency_proofs=proofs,
    )
    try:
        public_key.verify(
            bytes.fromhex(envelope["signature"]),
            _publication_message(body),
        )
    except InvalidSignature as exc:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence publication signature verification failed"
        ) from exc

    return TransparencyWitnessEvidencePublication(
        encoded_publication=encoded_publication,
        publisher_id=publisher_id,
        log_id=observation.checkpoint.log_id,
        policy_id=policy.policy_id,
        observation=observation,
        consistency_proofs=proofs,
    )


def fetch_transparency_witness_evidence_publication(
    publication_url: str,
    *,
    allow_insecure_http: bool = False,
    timeout: float = 10.0,
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> bytes:
    """Fetch bounded untrusted publication bytes without following redirects.

    Callers must still pass the returned bytes through
    ``verify_transparency_witness_evidence_publication`` or
    ``accept_transparency_witness_evidence_publication`` before treating them as
    authenticated evidence.
    """
    url = _normalize_publication_url(
        publication_url,
        allow_insecure_http=allow_insecure_http,
    )
    timeout = _validate_timeout(timeout)
    max_bytes = _validate_max_bytes(max_bytes)
    request = urllib.request.Request(
        url,
        headers={
            "Accept": _PUBLICATION_MEDIA_TYPE,
            "User-Agent": "tiny-tensor-compiler-transparency-evidence/1",
        },
        method="GET",
    )
    opener = urllib.request.build_opener(_NoRedirectHandler())
    received = bytearray()
    declared_size: int | None = None
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200:
                raise NativeBundleTransparencyPublicationError(
                    f"transparency evidence fetch returned unexpected HTTP status {response.status}"
                )
            media_type = response.headers.get_content_type()
            if media_type != _PUBLICATION_MEDIA_TYPE:
                raise NativeBundleTransparencyPublicationError(
                    "transparency evidence response media type is invalid"
                )
            declared = response.headers.get("Content-Length")
            if declared is not None:
                try:
                    declared_size = int(declared)
                except ValueError as exc:
                    raise NativeBundleTransparencyPublicationError(
                        "transparency evidence response Content-Length is malformed"
                    ) from exc
                if declared_size < 0 or declared_size > max_bytes:
                    raise NativeBundleTransparencyPublicationError(
                        f"transparency evidence publication exceeds transfer limit of {max_bytes} bytes"
                    )

            while True:
                chunk = response.read(_CHUNK_SIZE)
                if not chunk:
                    break
                received.extend(chunk)
                if len(received) > max_bytes:
                    raise NativeBundleTransparencyPublicationError(
                        f"transparency evidence publication exceeds transfer limit of {max_bytes} bytes"
                    )
    except urllib.error.HTTPError as exc:
        raise NativeBundleTransparencyPublicationError(
            f"transparency evidence fetch failed with HTTP status {exc.code}"
        ) from exc
    except urllib.error.URLError as exc:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence fetch transport failed"
        ) from exc
    except http.client.IncompleteRead as exc:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence response length does not match Content-Length"
        ) from exc

    if declared_size is not None and len(received) != declared_size:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence response length does not match Content-Length"
        )
    if not received:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence publication response is empty"
        )
    return bytes(received)


def accept_transparency_witness_evidence_publication(
    encoded_publication: bytes,
    *,
    publisher_public_key: bytes,
    log_public_key: bytes,
    policy: TransparencyWitnessPolicy,
    evidence_store: TransparencyWitnessEvidenceStore,
) -> TransparencyWitnessEvidenceSnapshot:
    """Verify one publication and atomically apply it through the durable store rules."""
    if not isinstance(evidence_store, TransparencyWitnessEvidenceStore):
        raise TypeError("evidence_store must be a TransparencyWitnessEvidenceStore")
    publication = verify_transparency_witness_evidence_publication(
        encoded_publication,
        publisher_public_key=publisher_public_key,
        log_public_key=log_public_key,
        policy=policy,
    )
    if evidence_store.log_id != publication.log_id:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence store uses a different pinned log operator"
        )
    if evidence_store.policy_id != publication.policy_id:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence store uses a different witness policy"
        )
    return evidence_store.record(
        publication.observation.encoded_observation,
        consistency_proofs=dict(publication.consistency_proofs),
    )


def _publication_body(
    *,
    publisher_id: str,
    log_id: str,
    policy_id: str,
    encoded_observation: bytes,
    consistency_proofs: tuple[tuple[str, tuple[bytes, ...]], ...],
) -> dict[str, Any]:
    try:
        observation_text = encoded_observation.decode("ascii")
    except UnicodeDecodeError as exc:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence publication observation is not ASCII"
        ) from exc
    return {
        "consistency_proofs": [
            {
                "checkpoint_digest": digest,
                "nodes": [node.hex() for node in nodes],
            }
            for digest, nodes in consistency_proofs
        ],
        "log_id": log_id,
        "observation": observation_text,
        "policy_id": policy_id,
        "publisher_id": normalize_publisher_id(publisher_id),
        "schema": _PUBLICATION_SCHEMA,
    }


def _publication_message(body: Mapping[str, Any]) -> bytes:
    return _PUBLICATION_DOMAIN + _canonical_json(dict(body))


def _decode_publication(encoded: bytes) -> dict[str, Any]:
    if not isinstance(encoded, bytes):
        raise TypeError("transparency evidence publication must be bytes")
    if not encoded or len(encoded) > _DEFAULT_MAX_BYTES:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence publication size is invalid"
        )
    try:
        decoded: Any = json.loads(
            encoded.decode("ascii"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence publication is not valid canonical JSON"
        ) from exc
    fields = {
        "consistency_proofs",
        "log_id",
        "observation",
        "policy_id",
        "publisher_id",
        "schema",
        "signature",
    }
    if not isinstance(decoded, dict) or set(decoded) != fields:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence publication fields are invalid"
        )
    if decoded.get("schema") != _PUBLICATION_SCHEMA:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence publication schema is unsupported"
        )
    if not isinstance(decoded.get("publisher_id"), str):
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence publication publisher identity is invalid"
        )
    if not isinstance(decoded.get("log_id"), str) or _LOG_ID_RE.fullmatch(decoded["log_id"]) is None:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence publication log identity is invalid"
        )
    if not isinstance(decoded.get("policy_id"), str) or not decoded["policy_id"]:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence publication policy identity is invalid"
        )
    if not isinstance(decoded.get("observation"), str) or not decoded["observation"]:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence publication observation is invalid"
        )
    if not isinstance(decoded.get("consistency_proofs"), list):
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence publication proof set is invalid"
        )
    if not isinstance(decoded.get("signature"), str) or _SIGNATURE_RE.fullmatch(decoded["signature"]) is None:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence publication signature is invalid"
        )
    if _canonical_json(decoded) != encoded:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence publication JSON is not canonical"
        )
    return decoded


def _decode_proofs(value: list[Any]) -> tuple[tuple[str, tuple[bytes, ...]], ...]:
    if len(value) > _MAX_PROOFS:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence publication exceeds proof-count limit"
        )
    decoded: list[tuple[str, tuple[bytes, ...]]] = []
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"checkpoint_digest", "nodes"}:
            raise NativeBundleTransparencyPublicationError(
                "transparency evidence publication proof entry is invalid"
            )
        digest = entry["checkpoint_digest"]
        nodes = entry["nodes"]
        if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
            raise NativeBundleTransparencyPublicationError(
                "transparency evidence publication proof digest is invalid"
            )
        if not isinstance(nodes, list) or len(nodes) > _MAX_PROOF_NODES:
            raise NativeBundleTransparencyPublicationError(
                "transparency evidence publication proof nodes are invalid"
            )
        proof: list[bytes] = []
        for node in nodes:
            if not isinstance(node, str) or _NODE_RE.fullmatch(node) is None:
                raise NativeBundleTransparencyPublicationError(
                    "transparency evidence publication proof node is invalid"
                )
            proof.append(bytes.fromhex(node))
        decoded.append((digest, tuple(proof)))
    if tuple(digest for digest, _nodes in decoded) != tuple(
        sorted(digest for digest, _nodes in decoded)
    ):
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence publication proofs are not deterministically ordered"
        )
    if len({digest for digest, _nodes in decoded}) != len(decoded):
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence publication proof digests are duplicated"
        )
    return tuple(decoded)


def _normalize_proofs(
    consistency_proofs: Mapping[str, Sequence[bytes]] | None,
) -> tuple[tuple[str, tuple[bytes, ...]], ...]:
    if consistency_proofs is None:
        return ()
    if not isinstance(consistency_proofs, Mapping):
        raise TypeError("consistency_proofs must be a mapping keyed by checkpoint digest")
    if len(consistency_proofs) > _MAX_PROOFS:
        raise NativeBundleTransparencyPublicationError(
            "transparency evidence publication exceeds proof-count limit"
        )
    normalized: list[tuple[str, tuple[bytes, ...]]] = []
    for digest, proof in consistency_proofs.items():
        if not isinstance(digest, str) or _DIGEST_RE.fullmatch(digest) is None:
            raise NativeBundleTransparencyPublicationError(
                "transparency evidence publication proof digest is invalid"
            )
        if isinstance(proof, (bytes, bytearray, str)) or not isinstance(proof, Sequence):
            raise TypeError("consistency proof must be a sequence of hash nodes")
        nodes = tuple(proof)
        if len(nodes) > _MAX_PROOF_NODES:
            raise NativeBundleTransparencyPublicationError(
                "transparency evidence publication proof exceeds node limit"
            )
        if any(not isinstance(node, bytes) or len(node) != 32 for node in nodes):
            raise TypeError("consistency proof nodes must be 32-byte values")
        normalized.append((digest, nodes))
    return tuple(sorted(normalized, key=lambda item: item[0]))


def _normalize_publication_url(publication_url: str, *, allow_insecure_http: bool) -> str:
    if not isinstance(publication_url, str) or not publication_url:
        raise TypeError("publication_url must be a non-empty string")
    parsed = urlsplit(publication_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("publication_url must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("publication_url must not embed credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("publication_url must not contain a query string or fragment")
    if parsed.scheme == "http" and not allow_insecure_http:
        raise ValueError(
            "insecure HTTP transparency evidence transport requires allow_insecure_http=True"
        )
    return publication_url


def _validate_timeout(timeout: float) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError("transparency evidence timeout must be a positive number")
    value = float(timeout)
    if value <= 0:
        raise ValueError("transparency evidence timeout must be positive")
    return value


def _validate_max_bytes(max_bytes: int) -> int:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise TypeError("transparency evidence max_bytes must be a positive integer")
    if max_bytes <= 0:
        raise ValueError("transparency evidence max_bytes must be positive")
    return max_bytes


def _private_key(value: bytes) -> Ed25519PrivateKey:
    if not isinstance(value, bytes):
        raise TypeError("transparency evidence publisher private key must be bytes")
    if len(value) != 32:
        raise ValueError("transparency evidence publisher private key must be 32 bytes")
    return Ed25519PrivateKey.from_private_bytes(value)


def _public_key(value: bytes) -> Ed25519PublicKey:
    if not isinstance(value, bytes):
        raise TypeError("transparency evidence publisher public key must be bytes")
    if len(value) != 32:
        raise ValueError("transparency evidence publisher public key must be 32 bytes")
    return Ed25519PublicKey.from_public_bytes(value)


def _require_policy(policy: TransparencyWitnessPolicy) -> TransparencyWitnessPolicy:
    if not isinstance(policy, TransparencyWitnessPolicy):
        raise TypeError("policy must be a TransparencyWitnessPolicy")
    return policy


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NativeBundleTransparencyPublicationError(
                "transparency evidence publication JSON contains duplicate keys"
            )
        result[key] = value
    return result


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None
