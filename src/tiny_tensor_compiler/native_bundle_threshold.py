from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .ir import SymbolicDim
from .native_bundle import NativeBundleExecutable
from .native_bundle_archive import NativeBundleSetArchiveExecutable, load_dynamic_bundle_set_archive
from .native_bundle_attestation import (
    PublisherTrustPolicy,
    normalize_publisher_id,
    publisher_id_from_public_key,
    publisher_public_key_from_private_key,
)
from .native_bundle_registry import (
    _normalize_registry_url,
    _validate_timeout,
    _validate_token,
    digest_dynamic_bundle_set_archive,
    fetch_dynamic_bundle_set_archive,
    publish_dynamic_bundle_set_archive,
)
from .native_bundle_release import (
    NativeBundleReleaseError,
    _canonical_json,
    _close_release_executable,
    _download_release_head,
    _normalize_channel,
    _normalize_digest,
    _publish_release_head,
    _state_lock,
    _validate_sequence,
)

_POLICY_SCHEMA = "ttc-threshold-ed25519-policy-v1"
_CHECKPOINT_SCHEMA = "ttc-threshold-ed25519-release-channel-v1"
_STATE_SCHEMA = "ttc-threshold-release-state-v1"
_DOMAIN = b"tiny-tensor-compiler\x00threshold-ed25519-release-channel-v1\x00"
_CHANNEL_PATH = "v1/channels/threshold-ed25519"
_POLICY_RE = re.compile(r"threshold-ed25519:([0-9a-f]{64})\Z")
_SIGNATURE_RE = re.compile(r"[0-9a-f]{128}\Z")
_MAX_CHECKPOINT_BYTES = 16 * 1024
_MAX_STATE_BYTES = 1024 * 1024
_MAX_SIGNERS = 16


class NativeBundleThresholdError(NativeBundleReleaseError):
    """Raised when threshold release authorization or its local state is invalid."""


class NativeBundleThresholdRollbackError(NativeBundleThresholdError):
    """Raised when a threshold-authorized channel is older than locally accepted state."""


@dataclass(frozen=True)
class ThresholdReleasePolicy:
    """Caller-pinned k-of-n Ed25519 release authorization policy."""

    public_keys: tuple[bytes, ...]
    threshold: int
    revoked_signers: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, int):
            raise TypeError("threshold must be an integer")
        base = PublisherTrustPolicy(self.public_keys, self.revoked_signers)
        ordered = tuple(sorted(base.public_keys, key=publisher_id_from_public_key))
        if len(ordered) > _MAX_SIGNERS:
            raise ValueError(f"threshold policy supports at most {_MAX_SIGNERS} signer keys")
        if self.threshold < 2 or self.threshold > len(ordered):
            raise ValueError("threshold must be between 2 and the number of signer keys")
        revoked = frozenset(normalize_publisher_id(value) for value in self.revoked_signers)
        signer_ids = tuple(publisher_id_from_public_key(key) for key in ordered)
        if len([signer for signer in signer_ids if signer not in revoked]) < self.threshold:
            raise ValueError("revocations leave fewer eligible signers than the threshold")
        object.__setattr__(self, "public_keys", ordered)
        object.__setattr__(self, "revoked_signers", revoked)

    @property
    def signer_ids(self) -> tuple[str, ...]:
        return tuple(publisher_id_from_public_key(key) for key in self.public_keys)

    @property
    def policy_id(self) -> str:
        descriptor = {
            "schema": _POLICY_SCHEMA,
            "signers": list(self.signer_ids),
            "threshold": self.threshold,
        }
        return f"threshold-ed25519:{hashlib.sha256(_canonical_json(descriptor)).hexdigest()}"

    @property
    def eligible_signers(self) -> tuple[str, ...]:
        return tuple(signer for signer in self.signer_ids if signer not in self.revoked_signers)

    def public_key_for(self, signer_id: str) -> bytes:
        normalized = normalize_publisher_id(signer_id)
        for public_key in self.public_keys:
            if publisher_id_from_public_key(public_key) == normalized:
                return public_key
        raise NativeBundleThresholdError(f"signer {normalized} is not in the threshold policy")


@dataclass(frozen=True, order=True)
class ThresholdReleaseCheckpoint:
    """One k-of-n authorized release sequence bound to an exact archive digest."""

    policy_id: str
    channel: str
    sequence: int
    archive_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _normalize_policy_id(self.policy_id))
        object.__setattr__(self, "channel", _normalize_channel(self.channel))
        object.__setattr__(self, "sequence", _validate_sequence(self.sequence))
        object.__setattr__(self, "archive_digest", _normalize_digest(self.archive_digest))


class ThresholdReleaseStateStore:
    """Caller-owned persistent rollback floor keyed by threshold policy and channel."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError("threshold release state path must be path-like")
        self._path = Path(path).expanduser().resolve()
        if self._path.exists() and not self._path.is_file():
            raise ValueError("threshold release state path must name a file")

    @property
    def path(self) -> Path:
        return self._path

    def floor(self, policy_id: str, channel: str) -> ThresholdReleaseCheckpoint | None:
        key = (_normalize_policy_id(policy_id), _normalize_channel(channel))
        with _state_lock(self._path):
            return _read_state(self._path).get(key)

    def precheck(self, checkpoint: ThresholdReleaseCheckpoint) -> None:
        checkpoint = _require_checkpoint(checkpoint)
        key = (checkpoint.policy_id, checkpoint.channel)
        with _state_lock(self._path):
            _check_checkpoint(_read_state(self._path).get(key), checkpoint)

    def record(self, checkpoint: ThresholdReleaseCheckpoint) -> ThresholdReleaseCheckpoint:
        checkpoint = _require_checkpoint(checkpoint)
        key = (checkpoint.policy_id, checkpoint.channel)
        with _state_lock(self._path):
            entries = _read_state(self._path)
            previous = entries.get(key)
            _check_checkpoint(previous, checkpoint)
            if previous == checkpoint:
                return checkpoint
            entries[key] = checkpoint
            _write_state(self._path, entries)
            return checkpoint


class ThresholdReleaseChannelRegistryExecutable:
    """Compiler-free executable loaded from one threshold-authorized channel head."""

    def __init__(
        self,
        executable: NativeBundleSetArchiveExecutable,
        download_root: Path,
        checkpoint: ThresholdReleaseCheckpoint,
    ) -> None:
        self._executable = executable
        self._download_root = download_root
        self._checkpoint = checkpoint
        self._finalizer = weakref.finalize(
            self,
            _close_release_executable,
            executable,
            download_root,
        )

    @property
    def checkpoint(self) -> ThresholdReleaseCheckpoint:
        return self._checkpoint

    @property
    def policy_id(self) -> str:
        return self._checkpoint.policy_id

    @property
    def digest(self) -> str:
        return self._checkpoint.archive_digest

    @property
    def channel(self) -> str:
        return self._checkpoint.channel

    @property
    def sequence(self) -> int:
        return self._checkpoint.sequence

    @property
    def symbolic_dims(self) -> tuple[str, ...]:
        return self._executable.symbolic_dims

    @property
    def available_bindings(self) -> tuple[tuple[tuple[str, int], ...], ...]:
        return self._executable.available_bindings

    @property
    def loaded_bindings(self) -> tuple[tuple[tuple[str, int], ...], ...]:
        return self._executable.loaded_bindings

    @property
    def closed(self) -> bool:
        return not self._finalizer.alive

    def close(self) -> None:
        if self._finalizer.alive:
            self._finalizer()

    def specialize(
        self,
        bindings: Mapping[SymbolicDim | str, int],
    ) -> NativeBundleExecutable:
        if self.closed:
            raise RuntimeError("threshold release channel registry executable is closed")
        return self._executable.specialize(bindings)

    def execute(self, inputs: Sequence[Any] = (), out: Any = None):
        if self.closed:
            raise RuntimeError("threshold release channel registry executable is closed")
        return self._executable.execute(inputs=inputs, out=out)

    def __call__(self, inputs: Sequence[Any] = (), out: Any = None):
        return self.execute(inputs=inputs, out=out)


def create_threshold_release_checkpoint(
    private_keys: Sequence[bytes],
    policy: ThresholdReleasePolicy,
    channel: str,
    sequence: int,
    archive_digest: str,
) -> bytes:
    """Create canonical k-of-n signatures over one exact release checkpoint."""
    policy = _require_policy(policy)
    checkpoint = ThresholdReleaseCheckpoint(
        policy.policy_id,
        channel,
        sequence,
        archive_digest,
    )
    signatures: list[dict[str, str]] = []
    seen: set[str] = set()
    for private_key in private_keys:
        private = _private_key(private_key)
        public_key = publisher_public_key_from_private_key(private_key)
        signer_id = publisher_id_from_public_key(public_key)
        policy.public_key_for(signer_id)
        if signer_id in policy.revoked_signers:
            raise NativeBundleThresholdError(f"signer {signer_id} is revoked by the threshold policy")
        if signer_id in seen:
            raise NativeBundleThresholdError("threshold checkpoint contains a duplicate signer")
        seen.add(signer_id)
        signatures.append(
            {
                "signature": private.sign(_threshold_message(checkpoint)).hex(),
                "signer_id": signer_id,
            }
        )
    signatures.sort(key=lambda item: item["signer_id"])
    if len(signatures) < policy.threshold:
        raise NativeBundleThresholdError(
            f"threshold checkpoint requires at least {policy.threshold} distinct eligible signatures"
        )
    envelope = {
        "archive_digest": checkpoint.archive_digest,
        "channel": checkpoint.channel,
        "policy_id": checkpoint.policy_id,
        "schema": _CHECKPOINT_SCHEMA,
        "sequence": checkpoint.sequence,
        "signatures": signatures,
    }
    encoded = _canonical_json(envelope)
    if len(encoded) > _MAX_CHECKPOINT_BYTES:
        raise NativeBundleThresholdError("threshold release checkpoint exceeds size limit")
    return encoded


def verify_threshold_release_checkpoint(
    encoded: bytes,
    policy: ThresholdReleasePolicy,
    *,
    expected_channel: str | None = None,
) -> ThresholdReleaseCheckpoint:
    """Verify canonical metadata and at least k distinct non-revoked Ed25519 signatures."""
    policy = _require_policy(policy)
    envelope = _decode_checkpoint(encoded)
    checkpoint = ThresholdReleaseCheckpoint(
        envelope["policy_id"],
        envelope["channel"],
        envelope["sequence"],
        envelope["archive_digest"],
    )
    if checkpoint.policy_id != policy.policy_id:
        raise NativeBundleThresholdError(
            f"threshold policy identity mismatch: expected {policy.policy_id}, "
            f"found {checkpoint.policy_id}"
        )
    if expected_channel is not None:
        expected = _normalize_channel(expected_channel)
        if checkpoint.channel != expected:
            raise NativeBundleThresholdError(
                f"threshold release channel mismatch: expected {expected}, found {checkpoint.channel}"
            )

    eligible = 0
    for signature in envelope["signatures"]:
        signer_id = normalize_publisher_id(signature["signer_id"])
        public_key = policy.public_key_for(signer_id)
        if signer_id in policy.revoked_signers:
            continue
        try:
            Ed25519PublicKey.from_public_bytes(public_key).verify(
                bytes.fromhex(signature["signature"]),
                _threshold_message(checkpoint),
            )
        except InvalidSignature as exc:
            raise NativeBundleThresholdError(
                f"threshold release signature verification failed for {signer_id}"
            ) from exc
        eligible += 1
    if eligible < policy.threshold:
        raise NativeBundleThresholdError(
            f"threshold release requires {policy.threshold} valid non-revoked signatures, found {eligible}"
        )
    return checkpoint


def publish_threshold_release_channel(
    archive: str | os.PathLike[str],
    registry_url: str,
    policy: ThresholdReleasePolicy,
    private_keys: Sequence[bytes],
    channel: str,
    sequence: int,
    *,
    token: str | None = None,
    allow_insecure_http: bool = False,
    timeout: float = 30.0,
    max_bytes: int = 512 * 1024 * 1024,
) -> ThresholdReleaseCheckpoint:
    """Publish an immutable archive and advance one k-of-n signed channel head."""
    policy = _require_policy(policy)
    normalized_channel = _normalize_channel(channel)
    normalized_sequence = _validate_sequence(sequence)
    digest = digest_dynamic_bundle_set_archive(archive)
    base_url = _normalize_registry_url(
        registry_url,
        allow_insecure_http=allow_insecure_http,
    )
    normalized_token = _validate_token(token)
    normalized_timeout = _validate_timeout(timeout)
    head_url = _threshold_channel_url(base_url, policy.policy_id, normalized_channel)
    current = _download_release_head(
        head_url,
        token=normalized_token,
        timeout=normalized_timeout,
        missing_ok=True,
    )
    current_checkpoint: ThresholdReleaseCheckpoint | None = None
    current_etag: str | None = None
    if current is not None:
        current_bytes, current_etag = current
        current_checkpoint = verify_threshold_release_checkpoint(
            current_bytes,
            policy,
            expected_channel=normalized_channel,
        )
        if normalized_sequence < current_checkpoint.sequence:
            raise NativeBundleThresholdError(
                "threshold release channel sequence must not move backward"
            )
        if (
            normalized_sequence == current_checkpoint.sequence
            and digest != current_checkpoint.archive_digest
        ):
            raise NativeBundleThresholdError(
                "threshold release channel cannot bind one sequence to a different digest"
            )

    published_digest = publish_dynamic_bundle_set_archive(
        archive,
        base_url,
        token=normalized_token,
        allow_insecure_http=allow_insecure_http,
        timeout=normalized_timeout,
        max_bytes=max_bytes,
    )
    if published_digest != digest:
        raise NativeBundleThresholdError("published archive digest changed unexpectedly")

    checkpoint = ThresholdReleaseCheckpoint(
        policy.policy_id,
        normalized_channel,
        normalized_sequence,
        digest,
    )
    if current_checkpoint == checkpoint:
        return checkpoint
    encoded = create_threshold_release_checkpoint(
        private_keys,
        policy,
        normalized_channel,
        normalized_sequence,
        digest,
    )
    _publish_release_head(
        head_url,
        encoded,
        token=normalized_token,
        timeout=normalized_timeout,
        previous_etag=current_etag,
    )
    remote = _download_release_head(
        head_url,
        token=normalized_token,
        timeout=normalized_timeout,
        missing_ok=False,
    )
    if remote is None:  # pragma: no cover - fail-closed helper contract
        raise NativeBundleThresholdError("threshold release head disappeared after publication")
    verified = verify_threshold_release_checkpoint(
        remote[0],
        policy,
        expected_channel=normalized_channel,
    )
    if verified != checkpoint:
        raise NativeBundleThresholdError(
            "threshold release channel failed post-publication verification"
        )
    return checkpoint


def fetch_threshold_release_channel_archive(
    registry_url: str,
    policy: ThresholdReleasePolicy,
    channel: str,
    destination: str | os.PathLike[str],
    state_store: ThresholdReleaseStateStore,
    *,
    token: str | None = None,
    allow_insecure_http: bool = False,
    timeout: float = 30.0,
    max_bytes: int = 512 * 1024 * 1024,
) -> ThresholdReleaseCheckpoint:
    """Fetch one k-of-n channel only if local rollback state permits the signed head."""
    policy = _require_policy(policy)
    if not isinstance(state_store, ThresholdReleaseStateStore):
        raise TypeError("state_store must be a ThresholdReleaseStateStore")
    normalized_channel = _normalize_channel(channel)
    base_url = _normalize_registry_url(
        registry_url,
        allow_insecure_http=allow_insecure_http,
    )
    normalized_token = _validate_token(token)
    normalized_timeout = _validate_timeout(timeout)
    head = _download_release_head(
        _threshold_channel_url(base_url, policy.policy_id, normalized_channel),
        token=normalized_token,
        timeout=normalized_timeout,
        missing_ok=False,
    )
    if head is None:  # pragma: no cover - fail-closed helper contract
        raise NativeBundleThresholdError("threshold release channel head is missing")
    checkpoint = verify_threshold_release_checkpoint(
        head[0],
        policy,
        expected_channel=normalized_channel,
    )

    # Reject a locally known rollback before any archive network request.
    state_store.precheck(checkpoint)
    destination_path = Path(destination).expanduser().resolve()
    if destination_path.exists():
        raise FileExistsError(
            f"threshold release channel destination already exists: {destination_path}"
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{destination_path.name}.threshold-release-", dir=destination_path.parent)
    )
    staged_archive = staging_root / "payload.ttca"
    try:
        fetch_dynamic_bundle_set_archive(
            base_url,
            checkpoint.archive_digest,
            staged_archive,
            token=normalized_token,
            allow_insecure_http=allow_insecure_http,
            timeout=normalized_timeout,
            max_bytes=max_bytes,
        )
        # Recheck atomically after remote verification in case another process advanced
        # the rollback floor while this download was in flight.
        state_store.record(checkpoint)
        if destination_path.exists():
            raise FileExistsError(
                f"threshold release channel destination already exists: {destination_path}"
            )
        os.replace(staged_archive, destination_path)
        return checkpoint
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def load_threshold_release_channel_registry(
    registry_url: str,
    policy: ThresholdReleasePolicy,
    channel: str,
    state_store: ThresholdReleaseStateStore,
    *,
    token: str | None = None,
    allow_insecure_http: bool = False,
    timeout: float = 30.0,
    max_bytes: int = 512 * 1024 * 1024,
) -> ThresholdReleaseChannelRegistryExecutable:
    """Load the current threshold-authorized channel without invoking a compiler."""
    root = Path(tempfile.mkdtemp(prefix="ttc-threshold-release-channel-"))
    archive = root / "payload.ttca"
    try:
        checkpoint = fetch_threshold_release_channel_archive(
            registry_url,
            policy,
            channel,
            archive,
            state_store,
            token=token,
            allow_insecure_http=allow_insecure_http,
            timeout=timeout,
            max_bytes=max_bytes,
        )
        executable = load_dynamic_bundle_set_archive(archive)
        return ThresholdReleaseChannelRegistryExecutable(executable, root, checkpoint)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def _decode_checkpoint(encoded: bytes) -> dict[str, Any]:
    if not isinstance(encoded, bytes):
        raise TypeError("threshold release checkpoint must be bytes")
    if not encoded or len(encoded) > _MAX_CHECKPOINT_BYTES:
        raise NativeBundleThresholdError("threshold release checkpoint size is invalid")
    try:
        decoded: Any = json.loads(encoded.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeBundleThresholdError(
            "threshold release checkpoint is not valid canonical JSON"
        ) from exc
    fields = {
        "archive_digest",
        "channel",
        "policy_id",
        "schema",
        "sequence",
        "signatures",
    }
    if not isinstance(decoded, dict) or set(decoded) != fields:
        raise NativeBundleThresholdError("threshold release checkpoint fields are invalid")
    if decoded.get("schema") != _CHECKPOINT_SCHEMA:
        raise NativeBundleThresholdError("threshold release checkpoint schema is unsupported")
    if isinstance(decoded.get("sequence"), bool) or not isinstance(decoded.get("sequence"), int):
        raise NativeBundleThresholdError("threshold release sequence must be an integer")
    for name in ("archive_digest", "channel", "policy_id", "schema"):
        if not isinstance(decoded.get(name), str):
            raise NativeBundleThresholdError(
                f"threshold release checkpoint {name} must be a string"
            )
    signatures = decoded.get("signatures")
    if not isinstance(signatures, list) or not signatures or len(signatures) > _MAX_SIGNERS:
        raise NativeBundleThresholdError("threshold release signatures are invalid")
    previous: str | None = None
    for item in signatures:
        if not isinstance(item, dict) or set(item) != {"signature", "signer_id"}:
            raise NativeBundleThresholdError("threshold release signature entry is invalid")
        signer = item.get("signer_id")
        signature = item.get("signature")
        if not isinstance(signer, str) or not isinstance(signature, str):
            raise NativeBundleThresholdError("threshold release signature fields must be strings")
        try:
            normalized_signer = normalize_publisher_id(signer)
        except (TypeError, ValueError) as exc:
            raise NativeBundleThresholdError("threshold release signer id is invalid") from exc
        if signer != normalized_signer:
            raise NativeBundleThresholdError("threshold release signer id is not canonical")
        if _SIGNATURE_RE.fullmatch(signature) is None:
            raise NativeBundleThresholdError("threshold release signature is not canonical Ed25519 hex")
        if previous is not None and signer <= previous:
            raise NativeBundleThresholdError(
                "threshold release signatures must be unique and sorted by signer id"
            )
        previous = signer
    if _canonical_json(decoded) != encoded:
        raise NativeBundleThresholdError("threshold release checkpoint JSON is not canonical")
    return decoded


def _threshold_message(checkpoint: ThresholdReleaseCheckpoint) -> bytes:
    return b"\x00".join(
        (
            _DOMAIN.rstrip(b"\x00"),
            checkpoint.policy_id.encode("ascii"),
            checkpoint.channel.encode("ascii"),
            str(checkpoint.sequence).encode("ascii"),
            checkpoint.archive_digest.encode("ascii"),
        )
    )


def _threshold_channel_url(base_url: str, policy_id: str, channel: str) -> str:
    policy_hash = _normalize_policy_id(policy_id).removeprefix("threshold-ed25519:")
    return f"{base_url}/{_CHANNEL_PATH}/{policy_hash}/{_normalize_channel(channel)}"


def _normalize_policy_id(policy_id: str) -> str:
    if not isinstance(policy_id, str):
        raise TypeError("threshold policy id must be a string")
    match = _POLICY_RE.fullmatch(policy_id)
    if match is None:
        raise ValueError(
            "threshold policy id must use canonical threshold-ed25519:<64 lowercase hex> form"
        )
    return f"threshold-ed25519:{match.group(1)}"


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


def _require_checkpoint(checkpoint: ThresholdReleaseCheckpoint) -> ThresholdReleaseCheckpoint:
    if not isinstance(checkpoint, ThresholdReleaseCheckpoint):
        raise TypeError("checkpoint must be a ThresholdReleaseCheckpoint")
    return checkpoint


def _check_checkpoint(
    previous: ThresholdReleaseCheckpoint | None,
    checkpoint: ThresholdReleaseCheckpoint,
) -> None:
    if previous is None:
        return
    if checkpoint.sequence < previous.sequence:
        raise NativeBundleThresholdRollbackError(
            f"threshold release rollback rejected: local sequence {previous.sequence}, "
            f"remote sequence {checkpoint.sequence}"
        )
    if (
        checkpoint.sequence == previous.sequence
        and checkpoint.archive_digest != previous.archive_digest
    ):
        raise NativeBundleThresholdError(
            "threshold release channel same sequence is bound to a different archive digest"
        )


def _read_state(path: Path) -> dict[tuple[str, str], ThresholdReleaseCheckpoint]:
    if not path.exists():
        return {}
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise NativeBundleThresholdError("failed to read threshold release rollback state") from exc
    if not raw or len(raw) > _MAX_STATE_BYTES:
        raise NativeBundleThresholdError("threshold release rollback state size is invalid")
    try:
        decoded: Any = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeBundleThresholdError(
            "threshold release rollback state is not valid JSON"
        ) from exc
    if not isinstance(decoded, dict) or set(decoded) != {"entries", "schema"}:
        raise NativeBundleThresholdError("threshold release rollback state fields are invalid")
    if decoded.get("schema") != _STATE_SCHEMA or not isinstance(decoded.get("entries"), list):
        raise NativeBundleThresholdError("threshold release rollback state schema is invalid")
    if _canonical_json(decoded) != raw:
        raise NativeBundleThresholdError("threshold release rollback state JSON is not canonical")

    entries: dict[tuple[str, str], ThresholdReleaseCheckpoint] = {}
    previous_key: tuple[str, str] | None = None
    for item in decoded["entries"]:
        if not isinstance(item, dict) or set(item) != {
            "archive_digest",
            "channel",
            "policy_id",
            "sequence",
        }:
            raise NativeBundleThresholdError("threshold release rollback state entry is invalid")
        try:
            checkpoint = ThresholdReleaseCheckpoint(
                item["policy_id"],
                item["channel"],
                item["sequence"],
                item["archive_digest"],
            )
        except (TypeError, ValueError) as exc:
            raise NativeBundleThresholdError(
                "threshold release rollback state entry is invalid"
            ) from exc
        key = (checkpoint.policy_id, checkpoint.channel)
        if key in entries or (previous_key is not None and key <= previous_key):
            raise NativeBundleThresholdError(
                "threshold release rollback state entries are not unique and sorted"
            )
        entries[key] = checkpoint
        previous_key = key
    return entries


def _write_state(
    path: Path,
    entries: dict[tuple[str, str], ThresholdReleaseCheckpoint],
) -> None:
    payload = {
        "entries": [
            {
                "archive_digest": checkpoint.archive_digest,
                "channel": checkpoint.channel,
                "policy_id": checkpoint.policy_id,
                "sequence": checkpoint.sequence,
            }
            for _key, checkpoint in sorted(entries.items())
        ],
        "schema": _STATE_SCHEMA,
    }
    encoded = _canonical_json(payload)
    if len(encoded) > _MAX_STATE_BYTES:
        raise NativeBundleThresholdError("threshold release rollback state exceeds size limit")
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
        raise NativeBundleThresholdError(
            "failed to persist threshold release rollback state"
        ) from exc
