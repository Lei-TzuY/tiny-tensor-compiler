from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.request
import weakref
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .ir import SymbolicDim
from .native_bundle import NativeBundleExecutable
from .native_bundle_archive import NativeBundleSetArchiveExecutable, load_dynamic_bundle_set_archive
from .native_bundle_attestation import (
    NativeBundleTrustError,
    PublisherTrustPolicy,
    normalize_publisher_id,
    publisher_id_from_public_key,
    publisher_public_key_from_private_key,
)
from .native_bundle_registry import (
    NativeBundleRegistryError,
    _normalize_registry_url,
    _registry_opener,
    _request_headers,
    _validate_timeout,
    _validate_token,
    digest_dynamic_bundle_set_archive,
)
from .native_bundle_trust import (
    fetch_attested_dynamic_bundle_set_archive,
    publish_attested_dynamic_bundle_set_archive,
)
from .native_cache_lock import _lock_stream, _unlock_stream

_RELEASE_SCHEMA = "ttc-ed25519-release-channel-v1"
_RELEASE_DOMAIN = b"tiny-tensor-compiler\x00native-bundle-release-channel-v1\x00"
_STATE_SCHEMA = "ttc-release-state-v1"
_CHANNEL_PATH = "v1/channels/ed25519"
_RELEASE_MEDIA_TYPE = "application/vnd.tiny-tensor-compiler.release-channel+json"
_MAX_RELEASE_BYTES = 16 * 1024
_MAX_STATE_BYTES = 1024 * 1024
_CHUNK_SIZE = 16 * 1024
_MAX_SEQUENCE = (1 << 63) - 1
_DIGEST_RE = re.compile(r"sha256:([0-9a-f]{64})\Z")
_SIGNATURE_RE = re.compile(r"[0-9a-f]{128}\Z")
_CHANNEL_RE = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?\Z")


class NativeBundleReleaseError(NativeBundleTrustError):
    """Raised when signed release metadata or rollback state is invalid."""


class NativeBundleRollbackError(NativeBundleReleaseError):
    """Raised when a signed channel head is older than locally accepted state."""


@dataclass(frozen=True, order=True)
class ReleaseCheckpoint:
    """One verified publisher/channel sequence bound to an exact archive digest."""

    publisher_id: str
    channel: str
    sequence: int
    archive_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "publisher_id", normalize_publisher_id(self.publisher_id))
        object.__setattr__(self, "channel", _normalize_channel(self.channel))
        object.__setattr__(self, "sequence", _validate_sequence(self.sequence))
        object.__setattr__(self, "archive_digest", _normalize_digest(self.archive_digest))


class ReleaseStateStore:
    """Caller-owned persistent rollback floor for publisher release channels.

    The file is local policy state, not a cryptographic trust root. Callers that need
    rollback protection must protect it from deletion or hostile local modification.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError("release state path must be path-like")
        self._path = Path(path).expanduser().resolve()
        if self._path.exists() and not self._path.is_file():
            raise ValueError("release state path must name a file")

    @property
    def path(self) -> Path:
        return self._path

    def floor(self, publisher_id: str, channel: str) -> ReleaseCheckpoint | None:
        key = (normalize_publisher_id(publisher_id), _normalize_channel(channel))
        with _state_lock(self._path):
            return _read_state(self._path).get(key)

    def precheck(self, checkpoint: ReleaseCheckpoint) -> None:
        checkpoint = _require_checkpoint(checkpoint)
        key = (checkpoint.publisher_id, checkpoint.channel)
        with _state_lock(self._path):
            _check_checkpoint(_read_state(self._path).get(key), checkpoint)

    def record(self, checkpoint: ReleaseCheckpoint) -> ReleaseCheckpoint:
        checkpoint = _require_checkpoint(checkpoint)
        key = (checkpoint.publisher_id, checkpoint.channel)
        with _state_lock(self._path):
            entries = _read_state(self._path)
            previous = entries.get(key)
            _check_checkpoint(previous, checkpoint)
            if previous == checkpoint:
                return checkpoint
            entries[key] = checkpoint
            _write_state(self._path, entries)
            return checkpoint


class ReleaseChannelRegistryExecutable:
    """Compiler-free executable loaded from one rollback-checked signed channel head."""

    def __init__(
        self,
        executable: NativeBundleSetArchiveExecutable,
        download_root: Path,
        checkpoint: ReleaseCheckpoint,
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
    def checkpoint(self) -> ReleaseCheckpoint:
        return self._checkpoint

    @property
    def digest(self) -> str:
        return self._checkpoint.archive_digest

    @property
    def publisher_id(self) -> str:
        return self._checkpoint.publisher_id

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

    def specialize(self, bindings: dict[SymbolicDim | str, int]) -> NativeBundleExecutable:
        if self.closed:
            raise RuntimeError("release channel registry executable is closed")
        return self._executable.specialize(bindings)

    def execute(self, inputs=(), out: Any = None):
        if self.closed:
            raise RuntimeError("release channel registry executable is closed")
        return self._executable.execute(inputs=inputs, out=out)

    def __call__(self, inputs=(), out: Any = None):
        return self.execute(inputs=inputs, out=out)


def create_release_checkpoint(
    private_key: bytes,
    channel: str,
    sequence: int,
    archive_digest: str,
) -> bytes:
    """Create deterministic canonical signed metadata for one release-channel head."""
    private = _private_key(private_key)
    public = publisher_public_key_from_private_key(private_key)
    checkpoint = ReleaseCheckpoint(
        publisher_id_from_public_key(public),
        channel,
        sequence,
        archive_digest,
    )
    signature = private.sign(_release_message(checkpoint))
    envelope = {
        "archive_digest": checkpoint.archive_digest,
        "channel": checkpoint.channel,
        "publisher_id": checkpoint.publisher_id,
        "schema": _RELEASE_SCHEMA,
        "sequence": checkpoint.sequence,
        "signature": signature.hex(),
    }
    return _canonical_json(envelope)


def verify_release_checkpoint(
    encoded: bytes,
    trust_policy: PublisherTrustPolicy,
    *,
    expected_publisher: str | None = None,
    expected_channel: str | None = None,
) -> ReleaseCheckpoint:
    """Verify canonical release metadata, pinned publisher identity, and Ed25519 signature."""
    if not isinstance(trust_policy, PublisherTrustPolicy):
        raise TypeError("trust_policy must be a PublisherTrustPolicy")
    envelope = _decode_release(encoded)
    checkpoint = ReleaseCheckpoint(
        envelope["publisher_id"],
        envelope["channel"],
        envelope["sequence"],
        envelope["archive_digest"],
    )
    if expected_publisher is not None:
        expected = normalize_publisher_id(expected_publisher)
        if checkpoint.publisher_id != expected:
            raise NativeBundleReleaseError(
                f"release publisher identity mismatch: expected {expected}, "
                f"found {checkpoint.publisher_id}"
            )
    if expected_channel is not None:
        expected = _normalize_channel(expected_channel)
        if checkpoint.channel != expected:
            raise NativeBundleReleaseError(
                f"release channel mismatch: expected {expected}, found {checkpoint.channel}"
            )

    public_key = trust_policy.public_key_for(checkpoint.publisher_id)
    signature_hex = envelope["signature"]
    if not isinstance(signature_hex, str) or _SIGNATURE_RE.fullmatch(signature_hex) is None:
        raise NativeBundleReleaseError("release signature is not canonical Ed25519 hex")
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            bytes.fromhex(signature_hex),
            _release_message(checkpoint),
        )
    except InvalidSignature as exc:
        raise NativeBundleReleaseError("release signature verification failed") from exc
    return checkpoint


def publish_release_channel(
    archive: str | os.PathLike[str],
    registry_url: str,
    private_key: bytes,
    channel: str,
    sequence: int,
    *,
    token: str | None = None,
    allow_insecure_http: bool = False,
    timeout: float = 30.0,
    max_bytes: int = 512 * 1024 * 1024,
) -> ReleaseCheckpoint:
    """Advance a publisher channel to a higher signed sequence and verified archive."""
    normalized_channel = _normalize_channel(channel)
    normalized_sequence = _validate_sequence(sequence)
    digest = digest_dynamic_bundle_set_archive(archive)
    public_key = publisher_public_key_from_private_key(private_key)
    publisher_id = publisher_id_from_public_key(public_key)
    policy = PublisherTrustPolicy((public_key,))
    base_url = _normalize_registry_url(
        registry_url,
        allow_insecure_http=allow_insecure_http,
    )
    normalized_token = _validate_token(token)
    normalized_timeout = _validate_timeout(timeout)
    head_url = _channel_url(base_url, publisher_id, normalized_channel)
    current = _download_release_head(
        head_url,
        token=normalized_token,
        timeout=normalized_timeout,
        missing_ok=True,
    )
    current_checkpoint: ReleaseCheckpoint | None = None
    current_etag: str | None = None
    if current is not None:
        current_bytes, current_etag = current
        current_checkpoint = verify_release_checkpoint(
            current_bytes,
            policy,
            expected_publisher=publisher_id,
            expected_channel=normalized_channel,
        )
        if normalized_sequence < current_checkpoint.sequence:
            raise NativeBundleReleaseError(
                "release channel sequence must advance beyond the current head"
            )
        if normalized_sequence == current_checkpoint.sequence:
            if digest != current_checkpoint.archive_digest:
                raise NativeBundleReleaseError(
                    "release channel cannot bind the same sequence to a different digest"
                )

    published_digest, published_id = publish_attested_dynamic_bundle_set_archive(
        archive,
        base_url,
        private_key,
        token=normalized_token,
        allow_insecure_http=allow_insecure_http,
        timeout=normalized_timeout,
        max_bytes=max_bytes,
    )
    if published_digest != digest or published_id != publisher_id:
        raise NativeBundleReleaseError("attested archive publication identity changed unexpectedly")

    checkpoint = ReleaseCheckpoint(
        publisher_id,
        normalized_channel,
        normalized_sequence,
        digest,
    )
    if current_checkpoint == checkpoint:
        return checkpoint

    encoded = create_release_checkpoint(
        private_key,
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
    if remote is None:  # pragma: no cover - missing_ok=False is fail-closed
        raise NativeBundleReleaseError("release channel head disappeared after publication")
    verified = verify_release_checkpoint(
        remote[0],
        policy,
        expected_publisher=publisher_id,
        expected_channel=normalized_channel,
    )
    if verified != checkpoint:
        raise NativeBundleReleaseError("release channel failed post-publication verification")
    return checkpoint


def fetch_release_channel_archive(
    registry_url: str,
    publisher_id: str,
    channel: str,
    destination: str | os.PathLike[str],
    trust_policy: PublisherTrustPolicy,
    state_store: ReleaseStateStore,
    *,
    token: str | None = None,
    allow_insecure_http: bool = False,
    timeout: float = 30.0,
    max_bytes: int = 512 * 1024 * 1024,
) -> ReleaseCheckpoint:
    """Fetch a signed channel head and archive without accepting a local rollback."""
    if not isinstance(trust_policy, PublisherTrustPolicy):
        raise TypeError("trust_policy must be a PublisherTrustPolicy")
    if not isinstance(state_store, ReleaseStateStore):
        raise TypeError("state_store must be a ReleaseStateStore")
    normalized_publisher = normalize_publisher_id(publisher_id)
    normalized_channel = _normalize_channel(channel)
    trust_policy.public_key_for(normalized_publisher)
    base_url = _normalize_registry_url(
        registry_url,
        allow_insecure_http=allow_insecure_http,
    )
    normalized_token = _validate_token(token)
    normalized_timeout = _validate_timeout(timeout)
    head = _download_release_head(
        _channel_url(base_url, normalized_publisher, normalized_channel),
        token=normalized_token,
        timeout=normalized_timeout,
        missing_ok=False,
    )
    if head is None:  # pragma: no cover - missing_ok=False is fail-closed
        raise NativeBundleReleaseError("release channel head is missing")
    checkpoint = verify_release_checkpoint(
        head[0],
        trust_policy,
        expected_publisher=normalized_publisher,
        expected_channel=normalized_channel,
    )

    # This check intentionally happens before archive/attestation network requests.
    state_store.precheck(checkpoint)
    destination_path = Path(destination).expanduser().resolve()
    if destination_path.exists():
        raise FileExistsError(f"release channel destination already exists: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{destination_path.name}.release-", dir=destination_path.parent)
    )
    staged_archive = staging_root / "payload.ttca"
    try:
        fetch_attested_dynamic_bundle_set_archive(
            base_url,
            checkpoint.archive_digest,
            checkpoint.publisher_id,
            staged_archive,
            trust_policy,
            token=normalized_token,
            allow_insecure_http=allow_insecure_http,
            timeout=normalized_timeout,
            max_bytes=max_bytes,
        )
        # Recheck under the cross-process state lock after all remote verification.
        # Another process may have accepted a newer release while this one downloaded.
        state_store.record(checkpoint)
        if destination_path.exists():
            raise FileExistsError(
                f"release channel destination already exists: {destination_path}"
            )
        os.replace(staged_archive, destination_path)
        return checkpoint
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def load_release_channel_registry(
    registry_url: str,
    publisher_id: str,
    channel: str,
    trust_policy: PublisherTrustPolicy,
    state_store: ReleaseStateStore,
    *,
    token: str | None = None,
    allow_insecure_http: bool = False,
    timeout: float = 30.0,
    max_bytes: int = 512 * 1024 * 1024,
) -> ReleaseChannelRegistryExecutable:
    """Load the current signed channel head after persistent rollback protection."""
    root = Path(tempfile.mkdtemp(prefix="ttc-release-channel-"))
    archive = root / "payload.ttca"
    try:
        checkpoint = fetch_release_channel_archive(
            registry_url,
            publisher_id,
            channel,
            archive,
            trust_policy,
            state_store,
            token=token,
            allow_insecure_http=allow_insecure_http,
            timeout=timeout,
            max_bytes=max_bytes,
        )
        executable = load_dynamic_bundle_set_archive(archive)
        return ReleaseChannelRegistryExecutable(executable, root, checkpoint)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def _decode_release(encoded: bytes) -> dict[str, Any]:
    if not isinstance(encoded, bytes):
        raise TypeError("release checkpoint must be bytes")
    if not encoded or len(encoded) > _MAX_RELEASE_BYTES:
        raise NativeBundleReleaseError("release checkpoint size is invalid")
    try:
        decoded: Any = json.loads(encoded.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeBundleReleaseError("release checkpoint is not valid canonical JSON") from exc
    fields = {
        "archive_digest",
        "channel",
        "publisher_id",
        "schema",
        "sequence",
        "signature",
    }
    if not isinstance(decoded, dict) or set(decoded) != fields:
        raise NativeBundleReleaseError("release checkpoint fields are invalid")
    if decoded.get("schema") != _RELEASE_SCHEMA:
        raise NativeBundleReleaseError("release checkpoint schema is unsupported")
    for name in ("archive_digest", "channel", "publisher_id", "schema", "signature"):
        if not isinstance(decoded.get(name), str):
            raise NativeBundleReleaseError(f"release checkpoint {name} must be a string")
    if isinstance(decoded.get("sequence"), bool) or not isinstance(decoded.get("sequence"), int):
        raise NativeBundleReleaseError("release checkpoint sequence must be an integer")
    if _canonical_json(decoded) != encoded:
        raise NativeBundleReleaseError("release checkpoint JSON is not canonical")
    return decoded


def _release_message(checkpoint: ReleaseCheckpoint) -> bytes:
    return b"\x00".join(
        (
            _RELEASE_DOMAIN.rstrip(b"\x00"),
            checkpoint.publisher_id.encode("ascii"),
            checkpoint.channel.encode("ascii"),
            str(checkpoint.sequence).encode("ascii"),
            checkpoint.archive_digest.encode("ascii"),
        )
    )


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        + b"\n"
    )


def _private_key(private_key: bytes) -> Ed25519PrivateKey:
    if not isinstance(private_key, bytes):
        raise TypeError("Ed25519 private key must be raw bytes")
    if len(private_key) != 32:
        raise ValueError("Ed25519 private key must be exactly 32 raw bytes")
    return Ed25519PrivateKey.from_private_bytes(private_key)


def _normalize_digest(digest: str) -> str:
    if not isinstance(digest, str):
        raise TypeError("release archive digest must be a string")
    match = _DIGEST_RE.fullmatch(digest)
    if match is None:
        raise ValueError("release archive digest must use canonical sha256:<64 lowercase hex> form")
    return f"sha256:{match.group(1)}"


def _normalize_channel(channel: str) -> str:
    if not isinstance(channel, str):
        raise TypeError("release channel must be a string")
    if _CHANNEL_RE.fullmatch(channel) is None:
        raise ValueError(
            "release channel must be 1-64 lowercase alphanumeric/dot/underscore/hyphen characters"
        )
    return channel


def _validate_sequence(sequence: int) -> int:
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise TypeError("release sequence must be an integer")
    if sequence < 0 or sequence > _MAX_SEQUENCE:
        raise ValueError(f"release sequence must be between 0 and {_MAX_SEQUENCE}")
    return sequence


def _require_checkpoint(checkpoint: ReleaseCheckpoint) -> ReleaseCheckpoint:
    if not isinstance(checkpoint, ReleaseCheckpoint):
        raise TypeError("checkpoint must be a ReleaseCheckpoint")
    return checkpoint


def _check_checkpoint(
    previous: ReleaseCheckpoint | None,
    checkpoint: ReleaseCheckpoint,
) -> None:
    if previous is None:
        return
    if checkpoint.sequence < previous.sequence:
        raise NativeBundleRollbackError(
            f"release rollback rejected: local sequence {previous.sequence}, "
            f"remote sequence {checkpoint.sequence}"
        )
    if (
        checkpoint.sequence == previous.sequence
        and checkpoint.archive_digest != previous.archive_digest
    ):
        raise NativeBundleReleaseError(
            "release channel same sequence is bound to a different archive digest"
        )


def _read_state(path: Path) -> dict[tuple[str, str], ReleaseCheckpoint]:
    if not path.exists():
        return {}
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise NativeBundleReleaseError("failed to read release rollback state") from exc
    if not raw or len(raw) > _MAX_STATE_BYTES:
        raise NativeBundleReleaseError("release rollback state size is invalid")
    try:
        decoded: Any = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeBundleReleaseError("release rollback state is not valid JSON") from exc
    if not isinstance(decoded, dict) or set(decoded) != {"entries", "schema"}:
        raise NativeBundleReleaseError("release rollback state fields are invalid")
    if decoded.get("schema") != _STATE_SCHEMA or not isinstance(decoded.get("entries"), list):
        raise NativeBundleReleaseError("release rollback state schema is invalid")
    if _canonical_json(decoded) != raw:
        raise NativeBundleReleaseError("release rollback state JSON is not canonical")

    entries: dict[tuple[str, str], ReleaseCheckpoint] = {}
    previous_sort_key: tuple[str, str] | None = None
    for item in decoded["entries"]:
        if not isinstance(item, dict) or set(item) != {
            "archive_digest",
            "channel",
            "publisher_id",
            "sequence",
        }:
            raise NativeBundleReleaseError("release rollback state entry is invalid")
        try:
            checkpoint = ReleaseCheckpoint(
                item["publisher_id"],
                item["channel"],
                item["sequence"],
                item["archive_digest"],
            )
        except (TypeError, ValueError) as exc:
            raise NativeBundleReleaseError("release rollback state entry is invalid") from exc
        key = (checkpoint.publisher_id, checkpoint.channel)
        if key in entries or (previous_sort_key is not None and key <= previous_sort_key):
            raise NativeBundleReleaseError("release rollback state entries are not unique and sorted")
        entries[key] = checkpoint
        previous_sort_key = key
    return entries


def _write_state(
    path: Path,
    entries: dict[tuple[str, str], ReleaseCheckpoint],
) -> None:
    payload = {
        "entries": [
            {
                "archive_digest": checkpoint.archive_digest,
                "channel": checkpoint.channel,
                "publisher_id": checkpoint.publisher_id,
                "sequence": checkpoint.sequence,
            }
            for _key, checkpoint in sorted(entries.items())
        ],
        "schema": _STATE_SCHEMA,
    }
    encoded = _canonical_json(payload)
    if len(encoded) > _MAX_STATE_BYTES:
        raise NativeBundleReleaseError("release rollback state exceeds size limit")
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
        raise NativeBundleReleaseError("failed to persist release rollback state") from exc


@contextmanager
def _state_lock(path: Path):
    lock_path = path.with_name(f".{path.name}.lock")
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        stream = lock_path.open("a+b")
    except OSError as exc:
        raise NativeBundleReleaseError("failed to open release rollback state lock") from exc
    locked = False
    try:
        _lock_stream(stream)
        locked = True
        yield
    except NativeBundleReleaseError:
        raise
    except Exception as exc:
        raise NativeBundleReleaseError("failed to lock release rollback state") from exc
    finally:
        try:
            if locked:
                _unlock_stream(stream)
        finally:
            stream.close()


def _channel_url(base_url: str, publisher_id: str, channel: str) -> str:
    publisher = normalize_publisher_id(publisher_id).removeprefix("ed25519:")
    return f"{base_url}/{_CHANNEL_PATH}/{publisher}/{_normalize_channel(channel)}"


def _download_release_head(
    object_url: str,
    *,
    token: str | None,
    timeout: float,
    missing_ok: bool,
) -> tuple[bytes, str | None] | None:
    request = urllib.request.Request(
        object_url,
        headers={**_request_headers(token), "Accept": _RELEASE_MEDIA_TYPE},
        method="GET",
    )
    opener = _registry_opener()
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200:
                raise NativeBundleRegistryError(
                    f"release channel fetch returned unexpected HTTP status {response.status}"
                )
            declared = response.headers.get("Content-Length")
            if declared is not None:
                try:
                    declared_size = int(declared)
                except ValueError as exc:
                    raise NativeBundleReleaseError(
                        "release channel Content-Length is malformed"
                    ) from exc
                if declared_size < 0 or declared_size > _MAX_RELEASE_BYTES:
                    raise NativeBundleReleaseError("release channel metadata exceeds transfer limit")
            chunks: list[bytes] = []
            received = 0
            while True:
                chunk = response.read(_CHUNK_SIZE)
                if not chunk:
                    break
                received += len(chunk)
                if received > _MAX_RELEASE_BYTES:
                    raise NativeBundleReleaseError("release channel metadata exceeds transfer limit")
                chunks.append(chunk)
            if declared is not None and received != declared_size:
                raise NativeBundleReleaseError(
                    "release channel response length does not match Content-Length"
                )
            etag = response.headers.get("ETag")
            if etag is not None and ("\r" in etag or "\n" in etag or not etag):
                raise NativeBundleReleaseError("release channel ETag is invalid")
            return b"".join(chunks), etag
    except urllib.error.HTTPError as exc:
        if missing_ok and exc.code == 404:
            return None
        raise NativeBundleRegistryError(
            f"release channel fetch failed with HTTP status {exc.code}"
        ) from exc
    except urllib.error.URLError as exc:
        raise NativeBundleRegistryError("release channel fetch transport failed") from exc


def _publish_release_head(
    object_url: str,
    encoded: bytes,
    *,
    token: str | None,
    timeout: float,
    previous_etag: str | None,
) -> None:
    if previous_etag is not None and not previous_etag:
        raise NativeBundleReleaseError("release channel ETag is invalid")
    headers = _request_headers(token)
    headers.update(
        {
            "Content-Length": str(len(encoded)),
            "Content-Type": _RELEASE_MEDIA_TYPE,
        }
    )
    if previous_etag is None:
        headers["If-None-Match"] = "*"
    else:
        headers["If-Match"] = previous_etag
    request = urllib.request.Request(object_url, data=encoded, headers=headers, method="PUT")
    opener = _registry_opener()
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status not in {200, 201, 204}:
                raise NativeBundleRegistryError(
                    f"release channel publish returned unexpected HTTP status {response.status}"
                )
    except urllib.error.HTTPError as exc:
        if exc.code in {409, 412}:
            raise NativeBundleReleaseError(
                "release channel head changed concurrently; retry publication"
            ) from exc
        raise NativeBundleRegistryError(
            f"release channel publish failed with HTTP status {exc.code}"
        ) from exc
    except urllib.error.URLError as exc:
        raise NativeBundleRegistryError("release channel publish transport failed") from exc


def _close_release_executable(
    executable: NativeBundleSetArchiveExecutable,
    download_root: Path,
) -> None:
    try:
        executable.close()
    finally:
        shutil.rmtree(download_root, ignore_errors=True)
