from __future__ import annotations

import os
import shutil
import tempfile
import urllib.error
import urllib.request
import weakref
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .ir import SymbolicDim
from .native_bundle import NativeBundleExecutable
from .native_bundle_archive import (
    NativeBundleSetArchiveExecutable,
    load_dynamic_bundle_set_archive,
)
from .native_bundle_attestation import (
    NativeBundleTrustError,
    PublisherTrustPolicy,
    create_archive_attestation,
    normalize_publisher_id,
    publisher_id_from_public_key,
    publisher_public_key_from_private_key,
    verify_archive_attestation,
)
from .native_bundle_registry import (
    NativeBundleRegistryError,
    _normalize_registry_url,
    _registry_opener,
    _request_headers,
    _validate_timeout,
    _validate_token,
    fetch_dynamic_bundle_set_archive,
    publish_dynamic_bundle_set_archive,
)

_ATTESTATION_PATH = "v1/attestations/ed25519"
_ATTESTATION_MEDIA_TYPE = "application/vnd.tiny-tensor-compiler.bundle-attestation+json"
_MAX_ATTESTATION_BYTES = 16 * 1024
_CHUNK_SIZE = 16 * 1024


class AttestedNativeBundleRegistryExecutable:
    """Compiler-free registry executable whose archive has a verified publisher attestation."""

    def __init__(
        self,
        executable: NativeBundleSetArchiveExecutable,
        download_root: Path,
        digest: str,
        publisher_id: str,
    ) -> None:
        self._executable = executable
        self._download_root = download_root
        self._digest = digest
        self._publisher_id = publisher_id
        self._finalizer = weakref.finalize(
            self,
            _close_attested_registry_executable,
            executable,
            download_root,
        )

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def publisher_id(self) -> str:
        return self._publisher_id

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
            raise RuntimeError("attested native bundle registry executable is closed")
        return self._executable.specialize(bindings)

    def execute(
        self,
        inputs: Sequence[Any] = (),
        out: Any = None,
    ):
        if self.closed:
            raise RuntimeError("attested native bundle registry executable is closed")
        return self._executable.execute(inputs=inputs, out=out)

    def __call__(
        self,
        inputs: Sequence[Any] = (),
        out: Any = None,
    ):
        return self.execute(inputs=inputs, out=out)


def publish_attested_dynamic_bundle_set_archive(
    archive: str | os.PathLike[str],
    registry_url: str,
    private_key: bytes,
    *,
    token: str | None = None,
    allow_insecure_http: bool = False,
    timeout: float = 30.0,
    max_bytes: int = 512 * 1024 * 1024,
) -> tuple[str, str]:
    """Publish a verified archive plus immutable Ed25519 authorization for its digest."""
    digest = publish_dynamic_bundle_set_archive(
        archive,
        registry_url,
        token=token,
        allow_insecure_http=allow_insecure_http,
        timeout=timeout,
        max_bytes=max_bytes,
    )
    public_key = publisher_public_key_from_private_key(private_key)
    publisher_id = publisher_id_from_public_key(public_key)
    attestation = create_archive_attestation(private_key, digest)
    base_url = _normalize_registry_url(
        registry_url,
        allow_insecure_http=allow_insecure_http,
    )
    token = _validate_token(token)
    timeout = _validate_timeout(timeout)
    object_url = _attestation_url(base_url, publisher_id, digest)
    _publish_immutable_attestation(
        object_url,
        attestation,
        token=token,
        timeout=timeout,
    )
    remote = _download_attestation(object_url, token=token, timeout=timeout)
    policy = PublisherTrustPolicy((public_key,))
    try:
        verify_archive_attestation(
            remote,
            digest,
            policy,
            expected_publisher=publisher_id,
        )
    except Exception as exc:
        raise NativeBundleRegistryError(
            "registry publisher attestation failed post-publication verification"
        ) from exc
    return digest, publisher_id


def fetch_attested_dynamic_bundle_set_archive(
    registry_url: str,
    digest: str,
    publisher_id: str,
    destination: str | os.PathLike[str],
    trust_policy: PublisherTrustPolicy,
    *,
    token: str | None = None,
    allow_insecure_http: bool = False,
    timeout: float = 30.0,
    max_bytes: int = 512 * 1024 * 1024,
) -> Path:
    """Fetch an archive and publish it locally only after pinned publisher verification."""
    normalized_publisher = normalize_publisher_id(publisher_id)
    if not isinstance(trust_policy, PublisherTrustPolicy):
        raise TypeError("trust_policy must be a PublisherTrustPolicy")
    # Fail before network access when the requested publisher is unknown or revoked.
    trust_policy.public_key_for(normalized_publisher)
    base_url = _normalize_registry_url(
        registry_url,
        allow_insecure_http=allow_insecure_http,
    )
    token = _validate_token(token)
    timeout = _validate_timeout(timeout)
    destination_path = Path(destination).expanduser().resolve()
    if destination_path.exists():
        raise FileExistsError(
            f"attested bundle registry destination already exists: {destination_path}"
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_path.name}.attested-",
            dir=destination_path.parent,
        )
    )
    staged_archive = staging_root / "payload.ttca"
    published = False
    try:
        fetch_dynamic_bundle_set_archive(
            base_url,
            digest,
            staged_archive,
            token=token,
            allow_insecure_http=allow_insecure_http,
            timeout=timeout,
            max_bytes=max_bytes,
        )
        attestation = _download_attestation(
            _attestation_url(base_url, normalized_publisher, digest),
            token=token,
            timeout=timeout,
        )
        verify_archive_attestation(
            attestation,
            digest,
            trust_policy,
            expected_publisher=normalized_publisher,
        )
        if destination_path.exists():
            raise FileExistsError(
                f"attested bundle registry destination already exists: {destination_path}"
            )
        os.replace(staged_archive, destination_path)
        published = True
        return destination_path
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        if not published:
            destination_path.unlink(missing_ok=True)


def load_attested_dynamic_bundle_set_registry(
    registry_url: str,
    digest: str,
    publisher_id: str,
    trust_policy: PublisherTrustPolicy,
    *,
    token: str | None = None,
    allow_insecure_http: bool = False,
    timeout: float = 30.0,
    max_bytes: int = 512 * 1024 * 1024,
) -> AttestedNativeBundleRegistryExecutable:
    """Load one compiler-free archive only after digest and publisher authorization checks."""
    download_root = Path(tempfile.mkdtemp(prefix="ttc-attested-bundle-registry-"))
    archive_path = download_root / "payload.ttca"
    try:
        fetch_attested_dynamic_bundle_set_archive(
            registry_url,
            digest,
            publisher_id,
            archive_path,
            trust_policy,
            token=token,
            allow_insecure_http=allow_insecure_http,
            timeout=timeout,
            max_bytes=max_bytes,
        )
        executable = load_dynamic_bundle_set_archive(archive_path)
        return AttestedNativeBundleRegistryExecutable(
            executable,
            download_root,
            digest,
            normalize_publisher_id(publisher_id),
        )
    except Exception:
        shutil.rmtree(download_root, ignore_errors=True)
        raise


def _publish_immutable_attestation(
    object_url: str,
    attestation: bytes,
    *,
    token: str | None,
    timeout: float,
) -> None:
    headers = _request_headers(token)
    headers.update(
        {
            "Content-Length": str(len(attestation)),
            "Content-Type": _ATTESTATION_MEDIA_TYPE,
            "If-None-Match": "*",
        }
    )
    request = urllib.request.Request(object_url, data=attestation, headers=headers, method="PUT")
    opener = _registry_opener()
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status not in {200, 201, 204}:
                raise NativeBundleRegistryError(
                    f"publisher attestation publish returned unexpected HTTP status {response.status}"
                )
    except urllib.error.HTTPError as exc:
        if exc.code not in {409, 412}:
            raise NativeBundleRegistryError(
                f"publisher attestation publish failed with HTTP status {exc.code}"
            ) from exc
    except urllib.error.URLError as exc:
        raise NativeBundleRegistryError("publisher attestation publish transport failed") from exc


def _download_attestation(
    object_url: str,
    *,
    token: str | None,
    timeout: float,
) -> bytes:
    request = urllib.request.Request(
        object_url,
        headers={**_request_headers(token), "Accept": _ATTESTATION_MEDIA_TYPE},
        method="GET",
    )
    opener = _registry_opener()
    chunks: list[bytes] = []
    received = 0
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200:
                raise NativeBundleRegistryError(
                    f"publisher attestation fetch returned unexpected HTTP status {response.status}"
                )
            declared = response.headers.get("Content-Length")
            if declared is not None:
                try:
                    declared_size = int(declared)
                except ValueError as exc:
                    raise NativeBundleRegistryError(
                        "publisher attestation Content-Length is malformed"
                    ) from exc
                if declared_size < 0 or declared_size > _MAX_ATTESTATION_BYTES:
                    raise NativeBundleRegistryError("publisher attestation exceeds transfer limit")
            while True:
                chunk = response.read(_CHUNK_SIZE)
                if not chunk:
                    break
                received += len(chunk)
                if received > _MAX_ATTESTATION_BYTES:
                    raise NativeBundleRegistryError("publisher attestation exceeds transfer limit")
                chunks.append(chunk)
            if declared is not None and received != declared_size:
                raise NativeBundleRegistryError(
                    "publisher attestation response length does not match Content-Length"
                )
    except urllib.error.HTTPError as exc:
        raise NativeBundleRegistryError(
            f"publisher attestation fetch failed with HTTP status {exc.code}"
        ) from exc
    except urllib.error.URLError as exc:
        raise NativeBundleRegistryError("publisher attestation fetch transport failed") from exc
    return b"".join(chunks)


def _attestation_url(base_url: str, publisher_id: str, digest: str) -> str:
    normalized_publisher = normalize_publisher_id(publisher_id)
    if not isinstance(digest, str) or not digest.startswith("sha256:") or len(digest) != 71:
        raise ValueError("archive digest must use canonical sha256:<64 lowercase hex> form")
    digest_hex = digest.removeprefix("sha256:")
    if any(char not in "0123456789abcdef" for char in digest_hex):
        raise ValueError("archive digest must use canonical sha256:<64 lowercase hex> form")
    return (
        f"{base_url}/{_ATTESTATION_PATH}/"
        f"{normalized_publisher.removeprefix('ed25519:')}/{digest_hex}"
    )


def _close_attested_registry_executable(
    executable: NativeBundleSetArchiveExecutable,
    download_root: Path,
) -> None:
    try:
        executable.close()
    finally:
        shutil.rmtree(download_root, ignore_errors=True)
