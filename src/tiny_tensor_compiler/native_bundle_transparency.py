from __future__ import annotations

import hashlib
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
    publisher_id_from_public_key,
    publisher_public_key_from_private_key,
)
from .native_bundle_release import NativeBundleReleaseError, _canonical_json, _state_lock

_CHECKPOINT_SCHEMA = "ttc-release-transparency-checkpoint-v1"
_CHECKPOINT_DOMAIN = b"tiny-tensor-compiler\x00release-transparency-checkpoint-v1\x00"
_STATE_SCHEMA = "ttc-release-transparency-state-v1"
_HASH_RE = re.compile(r"sha256:([0-9a-f]{64})\Z")
_SIGNATURE_RE = re.compile(r"[0-9a-f]{128}\Z")
_MAX_CHECKPOINT_BYTES = 16 * 1024
_MAX_STATE_BYTES = 16 * 1024
_MAX_TREE_SIZE = (1 << 63) - 1
_MAX_PROOF_NODES = 64
_HASH_SIZE = 32


class NativeBundleTransparencyError(NativeBundleReleaseError):
    """Raised when transparency metadata, proofs, or local state are invalid."""


class NativeBundleTransparencyRollbackError(NativeBundleTransparencyError):
    """Raised when a transparency head is older than locally accepted state."""


@dataclass(frozen=True, order=True)
class TransparencyCheckpoint:
    """One signed append-only log tree head pinned to an Ed25519 operator."""

    log_id: str
    tree_size: int
    root_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "log_id", _normalize_log_id(self.log_id))
        object.__setattr__(self, "tree_size", _validate_tree_size(self.tree_size))
        object.__setattr__(self, "root_hash", _normalize_root_hash(self.root_hash))

    @property
    def root_hash_bytes(self) -> bytes:
        return bytes.fromhex(self.root_hash.removeprefix("sha256:"))


class TransparencyStateStore:
    """Caller-owned persistent append-only floor for one pinned log operator.

    The state file is local policy state, not a cryptographic trust root. The caller
    must protect it from hostile deletion or modification. A missing file represents
    first contact and therefore provides no external freshness guarantee.
    """

    def __init__(self, path: str | os.PathLike[str], log_public_key: bytes) -> None:
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError("transparency state path must be path-like")
        self._path = Path(path).expanduser().resolve()
        if self._path.exists() and not self._path.is_file():
            raise ValueError("transparency state path must name a file")
        self._log_public_key = _public_key_bytes(log_public_key)
        self._log_id = log_id_from_public_key(self._log_public_key)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def log_id(self) -> str:
        return self._log_id

    def current(self) -> TransparencyCheckpoint | None:
        with _state_lock(self._path):
            return _read_state(self._path, self._log_id)

    def precheck(
        self,
        checkpoint: TransparencyCheckpoint,
        consistency_proof: Sequence[bytes] = (),
    ) -> None:
        checkpoint = _require_checkpoint(checkpoint)
        _require_log(checkpoint, self._log_id)
        with _state_lock(self._path):
            previous = _read_state(self._path, self._log_id)
            _check_progress(previous, checkpoint, consistency_proof)

    def record(
        self,
        checkpoint: TransparencyCheckpoint,
        consistency_proof: Sequence[bytes] = (),
    ) -> TransparencyCheckpoint:
        checkpoint = _require_checkpoint(checkpoint)
        _require_log(checkpoint, self._log_id)
        with _state_lock(self._path):
            previous = _read_state(self._path, self._log_id)
            _check_progress(previous, checkpoint, consistency_proof)
            if previous == checkpoint:
                return checkpoint
            _write_state(self._path, checkpoint)
            return checkpoint


def log_id_from_public_key(public_key: bytes) -> str:
    """Return the stable Ed25519 fingerprint used to pin one log operator."""
    return publisher_id_from_public_key(_public_key_bytes(public_key))


def transparency_leaf_hash(data: bytes) -> bytes:
    """Return the RFC 6962 Merkle leaf hash for exact release metadata bytes."""
    if not isinstance(data, bytes):
        raise TypeError("transparency leaf data must be bytes")
    return hashlib.sha256(b"\x00" + data).digest()


def transparency_node_hash(left: bytes, right: bytes) -> bytes:
    """Return the RFC 6962 Merkle interior-node hash."""
    return hashlib.sha256(b"\x01" + _hash_node(left) + _hash_node(right)).digest()


def create_transparency_checkpoint(
    private_key: bytes,
    tree_size: int,
    root_hash: bytes | str,
) -> bytes:
    """Create a canonical signed transparency tree head for one log operator."""
    private = _private_key(private_key)
    public_key = publisher_public_key_from_private_key(private_key)
    checkpoint = TransparencyCheckpoint(
        log_id_from_public_key(public_key),
        tree_size,
        _normalize_root_hash(root_hash),
    )
    signature = private.sign(_checkpoint_message(checkpoint))
    return _canonical_json(
        {
            "log_id": checkpoint.log_id,
            "root_hash": checkpoint.root_hash,
            "schema": _CHECKPOINT_SCHEMA,
            "signature": signature.hex(),
            "tree_size": checkpoint.tree_size,
        }
    )


def verify_transparency_checkpoint(
    encoded: bytes,
    log_public_key: bytes,
) -> TransparencyCheckpoint:
    """Verify canonical tree-head metadata against one caller-pinned log key."""
    public_key = _public_key_bytes(log_public_key)
    envelope = _decode_checkpoint(encoded)
    checkpoint = TransparencyCheckpoint(
        envelope["log_id"],
        envelope["tree_size"],
        envelope["root_hash"],
    )
    expected_log_id = log_id_from_public_key(public_key)
    if checkpoint.log_id != expected_log_id:
        raise NativeBundleTransparencyError(
            "transparency checkpoint log operator identity does not match the pinned key"
        )
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            bytes.fromhex(envelope["signature"]),
            _checkpoint_message(checkpoint),
        )
    except InvalidSignature as exc:
        raise NativeBundleTransparencyError(
            "transparency checkpoint log operator signature verification failed"
        ) from exc
    return checkpoint


def verify_transparency_inclusion(
    leaf_data: bytes,
    *,
    leaf_index: int,
    checkpoint: TransparencyCheckpoint,
    proof: Sequence[bytes],
) -> None:
    """Verify one RFC 6962 audit path against a signed checkpoint root."""
    checkpoint = _require_checkpoint(checkpoint)
    if isinstance(leaf_index, bool) or not isinstance(leaf_index, int):
        raise TypeError("transparency leaf index must be an integer")
    if leaf_index < 0 or leaf_index >= checkpoint.tree_size:
        raise ValueError("transparency leaf index is outside the checkpoint tree")
    nodes = _proof_nodes(proof)

    fn = leaf_index
    sn = checkpoint.tree_size - 1
    root = transparency_leaf_hash(leaf_data)
    for node in nodes:
        if sn == 0:
            raise NativeBundleTransparencyError("transparency inclusion proof has extra nodes")
        if fn & 1 or fn == sn:
            root = transparency_node_hash(node, root)
            if fn == sn:
                while fn and not (fn & 1):
                    fn >>= 1
                    sn >>= 1
        else:
            root = transparency_node_hash(root, node)
        fn >>= 1
        sn >>= 1

    if sn != 0 or root != checkpoint.root_hash_bytes:
        raise NativeBundleTransparencyError("transparency inclusion proof does not match tree root")


def verify_transparency_consistency(
    previous: TransparencyCheckpoint,
    current: TransparencyCheckpoint,
    proof: Sequence[bytes],
) -> None:
    """Verify that ``current`` is an RFC 6962 append-only extension of ``previous``."""
    previous = _require_checkpoint(previous)
    current = _require_checkpoint(current)
    if previous.log_id != current.log_id:
        raise NativeBundleTransparencyError(
            "transparency consistency checkpoints use different log operators"
        )
    nodes = _proof_nodes(proof)
    if previous.tree_size > current.tree_size:
        raise NativeBundleTransparencyRollbackError(
            "transparency tree-size rollback is not append-only"
        )
    if previous.tree_size == current.tree_size:
        if nodes or previous.root_hash != current.root_hash:
            raise NativeBundleTransparencyError(
                "same-size transparency checkpoints do not identify one tree"
            )
        return

    fn = previous.tree_size - 1
    sn = current.tree_size - 1
    while fn & 1:
        fn >>= 1
        sn >>= 1

    remaining = list(nodes)
    if fn == 0:
        first_root = previous.root_hash_bytes
        second_root = previous.root_hash_bytes
    else:
        if not remaining:
            raise NativeBundleTransparencyError("transparency consistency proof is incomplete")
        first_root = remaining[0]
        second_root = remaining[0]
        remaining = remaining[1:]

    for node in remaining:
        if sn == 0:
            raise NativeBundleTransparencyError("transparency consistency proof has extra nodes")
        if fn & 1 or fn == sn:
            first_root = transparency_node_hash(node, first_root)
            second_root = transparency_node_hash(node, second_root)
            while fn and not (fn & 1):
                fn >>= 1
                sn >>= 1
        else:
            second_root = transparency_node_hash(second_root, node)
        fn >>= 1
        sn >>= 1

    if (
        sn != 0
        or first_root != previous.root_hash_bytes
        or second_root != current.root_hash_bytes
    ):
        raise NativeBundleTransparencyError(
            "transparency consistency proof does not match checkpoint roots"
        )


def accept_release_transparency(
    release_checkpoint_bytes: bytes,
    *,
    leaf_index: int,
    encoded_checkpoint: bytes,
    log_public_key: bytes,
    inclusion_proof: Sequence[bytes],
    consistency_proof: Sequence[bytes],
    state_store: TransparencyStateStore,
) -> TransparencyCheckpoint:
    """Verify one already-authenticated release's log inclusion and append-only state.

    This function proves only transparency-log properties. The release bytes must be
    authenticated separately through the existing publisher or threshold release
    verifier before a caller treats them as an authorized release.
    """
    if not isinstance(state_store, TransparencyStateStore):
        raise TypeError("state_store must be a TransparencyStateStore")
    checkpoint = verify_transparency_checkpoint(encoded_checkpoint, log_public_key)
    if checkpoint.log_id != state_store.log_id:
        raise NativeBundleTransparencyError(
            "transparency checkpoint does not use the state store's pinned log operator"
        )
    verify_transparency_inclusion(
        release_checkpoint_bytes,
        leaf_index=leaf_index,
        checkpoint=checkpoint,
        proof=inclusion_proof,
    )
    return state_store.record(checkpoint, consistency_proof)


def _check_progress(
    previous: TransparencyCheckpoint | None,
    current: TransparencyCheckpoint,
    consistency_proof: Sequence[bytes],
) -> None:
    if previous is None:
        _proof_nodes(consistency_proof)
        if consistency_proof:
            raise NativeBundleTransparencyError(
                "first transparency checkpoint must not provide a consistency proof"
            )
        return
    if current.tree_size < previous.tree_size:
        raise NativeBundleTransparencyRollbackError(
            "transparency checkpoint rollback is below the locally accepted tree size"
        )
    if current.tree_size == previous.tree_size:
        if current.root_hash != previous.root_hash:
            raise NativeBundleTransparencyError(
                "same-size transparency fork conflicts with locally accepted tree root"
            )
        if consistency_proof:
            raise NativeBundleTransparencyError(
                "idempotent transparency checkpoint must not provide a consistency proof"
            )
        return
    verify_transparency_consistency(previous, current, consistency_proof)


def _checkpoint_message(checkpoint: TransparencyCheckpoint) -> bytes:
    return _CHECKPOINT_DOMAIN + _canonical_json(
        {
            "log_id": checkpoint.log_id,
            "root_hash": checkpoint.root_hash,
            "schema": _CHECKPOINT_SCHEMA,
            "tree_size": checkpoint.tree_size,
        }
    )


def _decode_checkpoint(encoded: bytes) -> dict[str, Any]:
    if not isinstance(encoded, bytes):
        raise TypeError("transparency checkpoint must be bytes")
    if not encoded or len(encoded) > _MAX_CHECKPOINT_BYTES:
        raise NativeBundleTransparencyError("transparency checkpoint size is invalid")
    try:
        decoded: Any = json.loads(
            encoded.decode("ascii"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeBundleTransparencyError(
            "transparency checkpoint is not valid canonical JSON"
        ) from exc
    fields = {"log_id", "root_hash", "schema", "signature", "tree_size"}
    if not isinstance(decoded, dict) or set(decoded) != fields:
        raise NativeBundleTransparencyError("transparency checkpoint fields are invalid")
    if decoded.get("schema") != _CHECKPOINT_SCHEMA:
        raise NativeBundleTransparencyError("transparency checkpoint schema is unsupported")
    if not isinstance(decoded.get("log_id"), str) or not isinstance(
        decoded.get("root_hash"), str
    ):
        raise NativeBundleTransparencyError("transparency checkpoint identity fields are invalid")
    if isinstance(decoded.get("tree_size"), bool) or not isinstance(
        decoded.get("tree_size"), int
    ):
        raise NativeBundleTransparencyError("transparency checkpoint tree size is invalid")
    signature = decoded.get("signature")
    if not isinstance(signature, str) or _SIGNATURE_RE.fullmatch(signature) is None:
        raise NativeBundleTransparencyError(
            "transparency checkpoint signature is not canonical Ed25519 hex"
        )
    try:
        TransparencyCheckpoint(
            decoded["log_id"],
            decoded["tree_size"],
            decoded["root_hash"],
        )
    except (TypeError, ValueError) as exc:
        raise NativeBundleTransparencyError(
            "transparency checkpoint metadata is invalid"
        ) from exc
    if _canonical_json(decoded) != encoded:
        raise NativeBundleTransparencyError("transparency checkpoint JSON is not canonical")
    return decoded


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NativeBundleTransparencyError(
                "transparency JSON contains duplicate object keys"
            )
        result[key] = value
    return result


def _read_state(path: Path, expected_log_id: str) -> TransparencyCheckpoint | None:
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise NativeBundleTransparencyError("failed to read transparency state") from exc
    if not raw or len(raw) > _MAX_STATE_BYTES:
        raise NativeBundleTransparencyError("transparency state size is invalid")
    try:
        decoded: Any = json.loads(raw.decode("ascii"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeBundleTransparencyError("transparency state is not valid JSON") from exc
    fields = {"log_id", "root_hash", "schema", "tree_size"}
    if not isinstance(decoded, dict) or set(decoded) != fields:
        raise NativeBundleTransparencyError("transparency state fields are invalid")
    if decoded.get("schema") != _STATE_SCHEMA:
        raise NativeBundleTransparencyError("transparency state schema is invalid")
    try:
        checkpoint = TransparencyCheckpoint(
            decoded["log_id"],
            decoded["tree_size"],
            decoded["root_hash"],
        )
    except (TypeError, ValueError) as exc:
        raise NativeBundleTransparencyError("transparency state checkpoint is invalid") from exc
    if checkpoint.log_id != expected_log_id:
        raise NativeBundleTransparencyError(
            "transparency state does not match the caller's pinned log operator"
        )
    if _canonical_json(decoded) != raw:
        raise NativeBundleTransparencyError("transparency state JSON is not canonical")
    return checkpoint


def _write_state(path: Path, checkpoint: TransparencyCheckpoint) -> None:
    encoded = _canonical_json(
        {
            "log_id": checkpoint.log_id,
            "root_hash": checkpoint.root_hash,
            "schema": _STATE_SCHEMA,
            "tree_size": checkpoint.tree_size,
        }
    )
    if len(encoded) > _MAX_STATE_BYTES:
        raise NativeBundleTransparencyError("transparency state exceeds size limit")
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
        raise NativeBundleTransparencyError("failed to persist transparency state") from exc


def _normalize_log_id(log_id: str) -> str:
    if not isinstance(log_id, str):
        raise TypeError("transparency log id must be a string")
    if not log_id.startswith("ed25519:"):
        raise ValueError("transparency log id must use canonical ed25519 fingerprint form")
    suffix = log_id.removeprefix("ed25519:")
    if len(suffix) != 64 or any(character not in "0123456789abcdef" for character in suffix):
        raise ValueError("transparency log id must use canonical ed25519 fingerprint form")
    return log_id


def _normalize_root_hash(root_hash: bytes | str) -> str:
    if isinstance(root_hash, bytes):
        if len(root_hash) != _HASH_SIZE:
            raise ValueError("transparency root hash must be exactly 32 bytes")
        return f"sha256:{root_hash.hex()}"
    if not isinstance(root_hash, str):
        raise TypeError("transparency root hash must be bytes or canonical sha256 text")
    match = _HASH_RE.fullmatch(root_hash)
    if match is None:
        raise ValueError("transparency root hash must use canonical sha256:<64 lowercase hex> form")
    return f"sha256:{match.group(1)}"


def _validate_tree_size(tree_size: int) -> int:
    if isinstance(tree_size, bool) or not isinstance(tree_size, int):
        raise TypeError("transparency tree size must be an integer")
    if tree_size < 1 or tree_size > _MAX_TREE_SIZE:
        raise ValueError("transparency tree size must be between 1 and 2^63-1")
    return tree_size


def _proof_nodes(proof: Sequence[bytes]) -> tuple[bytes, ...]:
    if isinstance(proof, (bytes, bytearray, str)) or not isinstance(proof, Sequence):
        raise TypeError("transparency proof must be a sequence of hash nodes")
    if len(proof) > _MAX_PROOF_NODES:
        raise NativeBundleTransparencyError("transparency proof exceeds bounded node count")
    return tuple(_hash_node(node) for node in proof)


def _hash_node(node: bytes) -> bytes:
    if not isinstance(node, bytes) or len(node) != _HASH_SIZE:
        raise NativeBundleTransparencyError(
            "transparency proof node must be exactly 32 raw SHA-256 bytes"
        )
    return node


def _private_key(private_key: bytes) -> Ed25519PrivateKey:
    if not isinstance(private_key, bytes):
        raise TypeError("Ed25519 private key must be raw bytes")
    if len(private_key) != 32:
        raise ValueError("Ed25519 private key must be exactly 32 raw bytes")
    return Ed25519PrivateKey.from_private_bytes(private_key)


def _public_key_bytes(public_key: bytes) -> bytes:
    if not isinstance(public_key, bytes):
        raise TypeError("Ed25519 public key must be raw bytes")
    if len(public_key) != 32:
        raise ValueError("Ed25519 public key must be exactly 32 raw bytes")
    Ed25519PublicKey.from_public_bytes(public_key)
    return public_key


def _require_checkpoint(checkpoint: TransparencyCheckpoint) -> TransparencyCheckpoint:
    if not isinstance(checkpoint, TransparencyCheckpoint):
        raise TypeError("checkpoint must be a TransparencyCheckpoint")
    return checkpoint


def _require_log(checkpoint: TransparencyCheckpoint, expected_log_id: str) -> None:
    if checkpoint.log_id != expected_log_id:
        raise NativeBundleTransparencyError(
            "transparency checkpoint does not match the pinned log operator"
        )
