from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


_PERSISTENT_CACHE_LEASE_POLL_SECONDS = 0.05
_PERSISTENT_CACHE_LEASE_TIMEOUT_SECONDS = 300.0


@contextmanager
def persistent_cache_lease(library_path: Path) -> Iterator[None]:
    """Hold one cross-process lease for a persistent-cache digest.

    The lease is backed by an operating-system file lock, so it is released when
    the file descriptor closes or the owning process exits unexpectedly.
    """
    lock_path = _lock_path(library_path)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        stream = lock_path.open("a+b")
    except OSError as error:
        raise RuntimeError(f"failed to open persistent native cache lease: {error}") from error

    try:
        _lock_stream(stream)
        yield
    finally:
        try:
            _unlock_stream(stream)
        finally:
            stream.close()


def _lock_path(library_path: Path) -> Path:
    digest = library_path.parent.name
    return library_path.parent.parent / f".{digest}.lock"


def _lock_stream(stream) -> None:
    if os.name == "nt":
        _lock_stream_windows(stream)
        return
    if os.name == "posix":
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        return
    raise RuntimeError(f"persistent native cache leases are unsupported on platform: {os.name}")


def _unlock_stream(stream) -> None:
    if os.name == "nt":
        import msvcrt

        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return
    if os.name == "posix":
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        return
    raise RuntimeError(f"persistent native cache leases are unsupported on platform: {os.name}")


def _lock_stream_windows(stream) -> None:
    import msvcrt

    stream.seek(0, os.SEEK_END)
    if stream.tell() == 0:
        stream.write(b"\0")
        stream.flush()
    stream.seek(0)

    deadline = time.monotonic() + _PERSISTENT_CACHE_LEASE_TIMEOUT_SECONDS
    while True:
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError as error:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "timed out waiting for persistent native cache lease"
                ) from error
            time.sleep(_PERSISTENT_CACHE_LEASE_POLL_SECONDS)
