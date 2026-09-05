from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import weakref
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .ir import SymbolicDim
from .native_bundle import NativeBundleError, NativeBundleExecutable
from .native_bundle_set import (
    NativeBundleSetError,
    NativeBundleSetExecutable,
    load_dynamic_bundle_set,
)

_ARCHIVE_SCHEMA = "native-bundle-archive-v1"
_ARCHIVE_MANIFEST = "archive.json"
_PAYLOAD_ROOT = "bundle"
_FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_REGULAR_FILE_MODE = (stat.S_IFREG | 0o644) << 16


class NativeBundleArchiveError(RuntimeError):
    """Raised when a native bundle archive is malformed or unsafe to extract."""


class NativeBundleSetArchiveExecutable:
    """Compiler-free finite bundle-set executable backed by one extracted archive."""

    def __init__(self, executable: NativeBundleSetExecutable, extraction_root: Path) -> None:
        self._executable = executable
        self._extraction_root = extraction_root
        self._finalizer = weakref.finalize(
            self,
            _close_archive_executable,
            executable,
            extraction_root,
        )

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
        """Close loaded child bundles and remove the private extracted payload."""
        if self._finalizer.alive:
            self._finalizer()

    def specialize(
        self,
        bindings: Mapping[SymbolicDim | str, int],
    ) -> NativeBundleExecutable:
        if self.closed:
            raise RuntimeError("native bundle archive executable is closed")
        return self._executable.specialize(bindings)

    def execute(
        self,
        inputs: Sequence[Any] = (),
        out: Any = None,
    ):
        if self.closed:
            raise RuntimeError("native bundle archive executable is closed")
        return self._executable.execute(inputs=inputs, out=out)

    def __call__(
        self,
        inputs: Sequence[Any] = (),
        out: Any = None,
    ):
        return self.execute(inputs=inputs, out=out)


def pack_dynamic_bundle_set_archive(
    bundle: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> Path:
    """Validate and deterministically pack one finite native bundle set."""
    bundle_path = Path(bundle).expanduser().resolve()
    if not bundle_path.is_dir():
        raise FileNotFoundError(f"dynamic bundle set does not exist: {bundle_path}")

    archive_path = Path(destination).expanduser().resolve()
    if archive_path.exists():
        raise FileExistsError(f"native bundle archive destination already exists: {archive_path}")
    if archive_path.is_relative_to(bundle_path):
        raise ValueError("native bundle archive destination must be outside the source bundle")

    _fully_validate_bundle_set_tree(
        bundle_path,
        error_message="source bundle set failed verification",
    )

    files = _collect_payload_files(bundle_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{archive_path.name}.build-",
        suffix=".tmp",
        dir=archive_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    published = False
    try:
        with zipfile.ZipFile(temporary_path, mode="w", compression=zipfile.ZIP_STORED) as archive:
            _write_entry(archive, _ARCHIVE_MANIFEST, _archive_manifest_bytes())
            for relative_path, source in files:
                _write_entry(
                    archive,
                    f"{_PAYLOAD_ROOT}/{relative_path}",
                    source.read_bytes(),
                )

        # Validate the exact bytes that are about to be published. Besides reusing the
        # full bundle/ABI checks, this closes the source-validation/read TOCTOU window:
        # any mutation while the archive is being assembled produces a rejected temp
        # artifact rather than a published archive that only fails later at load time.
        verified = load_dynamic_bundle_set_archive(temporary_path)
        verified.close()

        if archive_path.exists():
            raise FileExistsError(
                f"native bundle archive destination already exists: {archive_path}"
            )
        os.replace(temporary_path, archive_path)
        published = True
        return archive_path
    finally:
        if not published:
            temporary_path.unlink(missing_ok=True)


def load_dynamic_bundle_set_archive(
    archive: str | os.PathLike[str],
) -> NativeBundleSetArchiveExecutable:
    """Safely extract and load one compiler-free finite native bundle-set archive."""
    archive_path = Path(archive).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"native bundle archive does not exist: {archive_path}")

    extraction_root = Path(tempfile.mkdtemp(prefix="ttc-bundle-archive-"))
    try:
        with zipfile.ZipFile(archive_path, mode="r") as packed:
            entries = _validate_archive_entries(packed)
            manifest = _decode_archive_manifest(packed.read(_ARCHIVE_MANIFEST))
            payload_root = manifest["root"]
            for entry in entries:
                if entry.filename == _ARCHIVE_MANIFEST:
                    continue
                relative = PurePosixPath(entry.filename).relative_to(payload_root)
                destination = extraction_root.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with packed.open(entry, mode="r") as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)

        # The ordinary bundle-set loader intentionally validates child libraries lazily.
        # A transport boundary must be fail-closed before it is handed to the caller, so
        # force-load every packaged child once, close those validation handles, then
        # construct a fresh dispatcher whose normal execution remains lazy.
        _fully_validate_bundle_set_tree(
            extraction_root,
            error_message="archive payload failed bundle verification",
        )
        try:
            executable = load_dynamic_bundle_set(extraction_root)
        except (NativeBundleError, NativeBundleSetError) as exc:
            raise NativeBundleArchiveError("archive payload failed bundle verification") from exc
        return NativeBundleSetArchiveExecutable(executable, extraction_root)
    except Exception:
        shutil.rmtree(extraction_root, ignore_errors=True)
        raise


def _fully_validate_bundle_set_tree(
    bundle_path: Path,
    *,
    error_message: str,
) -> None:
    executable: NativeBundleSetExecutable | None = None
    try:
        executable = load_dynamic_bundle_set(bundle_path)
        for binding in executable.available_bindings:
            executable.specialize(dict(binding))
    except (NativeBundleError, NativeBundleSetError, OSError) as exc:
        raise NativeBundleArchiveError(error_message) from exc
    finally:
        if executable is not None:
            executable.close()


def _collect_payload_files(bundle_path: Path) -> tuple[tuple[str, Path], ...]:
    files: list[tuple[str, Path]] = []
    for path in bundle_path.rglob("*"):
        if path.is_symlink():
            raise NativeBundleArchiveError("source bundle contains a symbolic link")
        if path.is_dir():
            continue
        if not path.is_file():
            raise NativeBundleArchiveError("source bundle contains a non-regular entry")
        relative = path.relative_to(bundle_path).as_posix()
        _validate_relative_name(relative)
        files.append((relative, path))
    if not files:
        raise NativeBundleArchiveError("source bundle contains no files")
    return tuple(sorted(files, key=lambda item: item[0]))


def _validate_archive_entries(archive: zipfile.ZipFile) -> tuple[zipfile.ZipInfo, ...]:
    entries = tuple(archive.infolist())
    names = [entry.filename for entry in entries]
    if len(names) != len(set(names)):
        raise NativeBundleArchiveError("native bundle archive contains duplicate entry names")
    if _ARCHIVE_MANIFEST not in names:
        raise NativeBundleArchiveError("native bundle archive is missing archive.json")

    payload_prefix = f"{_PAYLOAD_ROOT}/"
    payload_entries = 0
    for entry in entries:
        _validate_relative_name(entry.filename)
        if entry.is_dir():
            raise NativeBundleArchiveError("native bundle archive must not contain directory entries")
        if entry.flag_bits & 0x1:
            raise NativeBundleArchiveError("encrypted native bundle archive entries are unsupported")
        if entry.compress_type != zipfile.ZIP_STORED:
            raise NativeBundleArchiveError("native bundle archive entries must use stored compression")
        mode = entry.external_attr >> 16
        if stat.S_IFMT(mode) == stat.S_IFLNK:
            raise NativeBundleArchiveError("native bundle archive must not contain symbolic links")
        if entry.filename == _ARCHIVE_MANIFEST:
            continue
        if not entry.filename.startswith(payload_prefix):
            raise NativeBundleArchiveError("native bundle archive entry is outside the payload root")
        if entry.filename == payload_prefix:
            raise NativeBundleArchiveError("native bundle archive payload entry is empty")
        payload_entries += 1

    if payload_entries == 0:
        raise NativeBundleArchiveError("native bundle archive payload is empty")
    return entries


def _validate_relative_name(name: str) -> None:
    if not name or "\\" in name:
        raise NativeBundleArchiveError("native bundle archive entry name is not canonical")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise NativeBundleArchiveError("native bundle archive entry escapes its payload root")


def _archive_manifest_bytes() -> bytes:
    manifest = {
        "kind": "dynamic-bundle-set",
        "root": _PAYLOAD_ROOT,
        "schema": _ARCHIVE_SCHEMA,
    }
    return (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _decode_archive_manifest(data: bytes) -> dict[str, str]:
    try:
        decoded = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeBundleArchiveError("native bundle archive manifest is malformed") from exc
    if not isinstance(decoded, dict) or set(decoded) != {"kind", "root", "schema"}:
        raise NativeBundleArchiveError("native bundle archive manifest fields are invalid")
    if decoded.get("schema") != _ARCHIVE_SCHEMA:
        raise NativeBundleArchiveError("unsupported native bundle archive schema")
    if decoded.get("kind") != "dynamic-bundle-set":
        raise NativeBundleArchiveError("unsupported native bundle archive payload kind")
    if decoded.get("root") != _PAYLOAD_ROOT:
        raise NativeBundleArchiveError("native bundle archive payload root is invalid")
    return decoded


def _write_entry(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(filename=name, date_time=_FIXED_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = _REGULAR_FILE_MODE
    info.flag_bits = 0
    archive.writestr(info, data)


def _close_archive_executable(
    executable: NativeBundleSetExecutable,
    extraction_root: Path,
) -> None:
    try:
        executable.close()
    finally:
        shutil.rmtree(extraction_root, ignore_errors=True)
