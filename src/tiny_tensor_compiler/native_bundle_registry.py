from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.request
import weakref
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .ir import SymbolicDim
from .native_bundle import NativeBundleExecutable
from .native_bundle_archive import (
    NativeBundleSetArchiveExecutable,
    load_dynamic_bundle_set_archive,
)

_REGISTRY_PATH = "v1/archives/sha256"
_ARCHIVE_MEDIA_TYPE = "application/vnd.tiny-tensor-compiler.bundle-archive"
_DEFAULT_MAX_BYTES = 512 * 1024 * 1024
_CHUNK_SIZE = 64 * 1024
_DIGEST_RE = re.compile(r"sha256:([0-9a-f]{64})\Z")


class NativeBundleRegistryError(RuntimeError):
    """Raised when remote bundle-registry transport or verification fails."""


class NativeBundleRegistryExecutable:
    """Compiler-free executable backed by one verified registry download."""

    def __init__(
        self,
        executable: NativeBundleSetArchiveExecutable,
        download_root: Path,
        digest: str,
    ) -> None:
        self._executable = executable
        self._download_root = download_root
        self._digest = digest
        self._finalizer = weakref.finalize(
            self,
            _close_registry_executable,
            executable,
            download_root,
        )

    @property
    def digest(self) -> str:
        return self._digest

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
        """Close child bundles and remove the private downloaded archive."""
        if self._finalizer.alive:
            self._finalizer()

    def specialize(
        self,
        bindings: Mapping[SymbolicDim | str, int],
    ) -> NativeBundleExecutable:
        if self.closed:
            raise RuntimeError("native bundle registry executable is closed")
        return self._executable.specialize(bindings)

    def execute(
        self,
        inputs: Sequence[Any] = (),
        out: Any = None,
    ):
        if self.closed:
            raise RuntimeError("native bundle registry executable is closed")
        return self._executable.execute(inputs=inputs, out=out)

    def __call__(
        self,
        inputs: Sequence[Any] = (),
        out: Any = None,
    ):
        return self.execute(inputs=inputs, out=out)


def digest_dynamic_bundle_set_archive(archive: str | os.PathLike[str]) -> str:
    """Return the canonical SHA-256 content address for one archive file."""
    archive_path = Path(archive).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"native bundle archive does not exist: {archive_path}")
    return f"sha256:{_sha256_file(archive_path)}"


def publish_dynamic_bundle_set_archive(
    archive: str | os.PathLike[str],
    registry_url: str,
    *,
    token: str | None = None,
    allow_insecure_http: bool = False,
    timeout: float = 30.0,
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> str:
    """Publish one verified archive to its immutable content-addressed registry URL."""
    archive_path = Path(archive).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"native bundle archive does not exist: {archive_path}")
    max_bytes = _validate_max_bytes(max_bytes)
    timeout = _validate_timeout(timeout)
    base_url = _normalize_registry_url(registry_url, allow_insecure_http=allow_insecure_http)
    token = _validate_token(token)

    size = archive_path.stat().st_size
    if size > max_bytes:
        raise NativeBundleRegistryError(
            f"native bundle archive exceeds registry transfer limit of {max_bytes} bytes"
        )

    verified = load_dynamic_bundle_set_archive(archive_path)
    verified.close()
    digest = digest_dynamic_bundle_set_archive(archive_path)
    object_url = _object_url(base_url, digest)
    payload = archive_path.read_bytes()

    headers = _request_headers(token)
    headers.update(
        {
            "Content-Length": str(len(payload)),
            "Content-Type": _ARCHIVE_MEDIA_TYPE,
            "If-None-Match": "*",
            "X-TTC-Content-SHA256": digest.removeprefix("sha256:"),
        }
    )
    request = urllib.request.Request(object_url, data=payload, headers=headers, method="PUT")
    opener = _registry_opener()
    already_exists = False
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status not in {200, 201, 204}:
                raise NativeBundleRegistryError(
                    f"registry publish returned unexpected HTTP status {response.status}"
                )
    except urllib.error.HTTPError as exc:
        if exc.code in {409, 412}:
            already_exists = True
        else:
            raise NativeBundleRegistryError(
                f"registry publish failed with HTTP status {exc.code}"
            ) from exc
    except urllib.error.URLError as exc:
        raise NativeBundleRegistryError("registry publish transport failed") from exc

    # Never trust the upload response alone. Read the immutable object back, verify the
    # caller-derived digest, and run the existing archive/child-ABI verifier over the
    # exact remote bytes before declaring publication successful. This also makes an
    # idempotent 409/412 safe only when the pre-existing object is coherent.
    try:
        _verify_remote_object(
            object_url,
            digest,
            token=token,
            timeout=timeout,
            max_bytes=max_bytes,
        )
    except Exception as exc:
        state = "pre-existing" if already_exists else "published"
        raise NativeBundleRegistryError(
            f"registry {state} object failed post-publication verification"
        ) from exc
    return digest


def fetch_dynamic_bundle_set_archive(
    registry_url: str,
    digest: str,
    destination: str | os.PathLike[str],
    *,
    token: str | None = None,
    allow_insecure_http: bool = False,
    timeout: float = 30.0,
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> Path:
    """Fetch, digest-check, fully verify, and atomically publish one archive locally."""
    normalized_digest = _normalize_digest(digest)
    base_url = _normalize_registry_url(registry_url, allow_insecure_http=allow_insecure_http)
    token = _validate_token(token)
    timeout = _validate_timeout(timeout)
    max_bytes = _validate_max_bytes(max_bytes)
    destination_path = Path(destination).expanduser().resolve()
    if destination_path.exists():
        raise FileExistsError(
            f"native bundle registry destination already exists: {destination_path}"
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.download-",
        suffix=".tmp",
        dir=destination_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    published = False
    try:
        _download_registry_object(
            _object_url(base_url, normalized_digest),
            normalized_digest,
            temporary_path,
            token=token,
            timeout=timeout,
            max_bytes=max_bytes,
        )
        verified = load_dynamic_bundle_set_archive(temporary_path)
        verified.close()
        if destination_path.exists():
            raise FileExistsError(
                f"native bundle registry destination already exists: {destination_path}"
            )
        os.replace(temporary_path, destination_path)
        published = True
        return destination_path
    finally:
        if not published:
            temporary_path.unlink(missing_ok=True)


def load_dynamic_bundle_set_registry(
    registry_url: str,
    digest: str,
    *,
    token: str | None = None,
    allow_insecure_http: bool = False,
    timeout: float = 30.0,
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> NativeBundleRegistryExecutable:
    """Download and load one compiler-free content-addressed bundle archive."""
    normalized_digest = _normalize_digest(digest)
    base_url = _normalize_registry_url(registry_url, allow_insecure_http=allow_insecure_http)
    token = _validate_token(token)
    timeout = _validate_timeout(timeout)
    max_bytes = _validate_max_bytes(max_bytes)
    download_root = Path(tempfile.mkdtemp(prefix="ttc-bundle-registry-"))
    archive_path = download_root / "payload.ttca"
    try:
        _download_registry_object(
            _object_url(base_url, normalized_digest),
            normalized_digest,
            archive_path,
            token=token,
            timeout=timeout,
            max_bytes=max_bytes,
        )
        executable = load_dynamic_bundle_set_archive(archive_path)
        return NativeBundleRegistryExecutable(executable, download_root, normalized_digest)
    except Exception:
        shutil.rmtree(download_root, ignore_errors=True)
        raise


def _verify_remote_object(
    object_url: str,
    digest: str,
    *,
    token: str | None,
    timeout: float,
    max_bytes: int,
) -> None:
    root = Path(tempfile.mkdtemp(prefix="ttc-bundle-registry-verify-"))
    archive_path = root / "payload.ttca"
    try:
        _download_registry_object(
            object_url,
            digest,
            archive_path,
            token=token,
            timeout=timeout,
            max_bytes=max_bytes,
        )
        executable = load_dynamic_bundle_set_archive(archive_path)
        executable.close()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _download_registry_object(
    object_url: str,
    digest: str,
    destination: Path,
    *,
    token: str | None,
    timeout: float,
    max_bytes: int,
) -> None:
    request = urllib.request.Request(
        object_url,
        headers={**_request_headers(token), "Accept": _ARCHIVE_MEDIA_TYPE},
        method="GET",
    )
    opener = _registry_opener()
    hasher = hashlib.sha256()
    received = 0
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200:
                raise NativeBundleRegistryError(
                    f"registry fetch returned unexpected HTTP status {response.status}"
                )
            declared = response.headers.get("Content-Length")
            if declared is not None:
                try:
                    declared_size = int(declared)
                except ValueError as exc:
                    raise NativeBundleRegistryError(
                        "registry response Content-Length is malformed"
                    ) from exc
                if declared_size < 0 or declared_size > max_bytes:
                    raise NativeBundleRegistryError(
                        f"registry object exceeds transfer limit of {max_bytes} bytes"
                    )

            with destination.open("wb") as target:
                while True:
                    chunk = response.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > max_bytes:
                        raise NativeBundleRegistryError(
                            f"registry object exceeds transfer limit of {max_bytes} bytes"
                        )
                    hasher.update(chunk)
                    target.write(chunk)

            if declared is not None and received != declared_size:
                raise NativeBundleRegistryError(
                    "registry response length does not match Content-Length"
                )
    except urllib.error.HTTPError as exc:
        raise NativeBundleRegistryError(
            f"registry fetch failed with HTTP status {exc.code}"
        ) from exc
    except urllib.error.URLError as exc:
        raise NativeBundleRegistryError("registry fetch transport failed") from exc

    actual = f"sha256:{hasher.hexdigest()}"
    if actual != digest:
        raise NativeBundleRegistryError(
            f"registry object digest mismatch: expected {digest}, found {actual}"
        )


def _normalize_registry_url(registry_url: str, *, allow_insecure_http: bool) -> str:
    if not isinstance(registry_url, str) or not registry_url:
        raise TypeError("registry_url must be a non-empty string")
    parsed = urlsplit(registry_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("registry_url must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("registry_url must not embed credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("registry_url must not contain a query string or fragment")
    if parsed.scheme == "http" and not allow_insecure_http:
        raise ValueError("insecure HTTP registry transport requires allow_insecure_http=True")
    return registry_url.rstrip("/")


def _normalize_digest(digest: str) -> str:
    if not isinstance(digest, str):
        raise TypeError("registry digest must be a string")
    match = _DIGEST_RE.fullmatch(digest)
    if match is None:
        raise ValueError("registry digest must use canonical sha256:<64 lowercase hex> form")
    return f"sha256:{match.group(1)}"


def _object_url(base_url: str, digest: str) -> str:
    return f"{base_url}/{_REGISTRY_PATH}/{digest.removeprefix('sha256:')}"


def _request_headers(token: str | None) -> dict[str, str]:
    headers = {"User-Agent": "tiny-tensor-compiler-registry/1"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _validate_token(token: str | None) -> str | None:
    if token is None:
        return None
    if not isinstance(token, str) or not token:
        raise TypeError("registry token must be a non-empty string or None")
    if "\r" in token or "\n" in token:
        raise ValueError("registry token must not contain line breaks")
    return token


def _validate_timeout(timeout: float) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError("registry timeout must be a positive number")
    value = float(timeout)
    if value <= 0:
        raise ValueError("registry timeout must be positive")
    return value


def _validate_max_bytes(max_bytes: int) -> int:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
        raise TypeError("registry max_bytes must be a positive integer")
    if max_bytes <= 0:
        raise ValueError("registry max_bytes must be positive")
    return max_bytes


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _registry_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_NoRedirectHandler())


def _close_registry_executable(
    executable: NativeBundleSetArchiveExecutable,
    download_root: Path,
) -> None:
    try:
        executable.close()
    finally:
        shutil.rmtree(download_root, ignore_errors=True)
