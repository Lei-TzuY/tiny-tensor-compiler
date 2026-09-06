from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_PERSISTENT_CACHE_LEASE_POLL_SECONDS = 0.05
_PERSISTENT_CACHE_LEASE_TIMEOUT_SECONDS = 300.0


class PersistentCacheLeaseError(RuntimeError):
    """Raised when a persistent-cache cross-process lease cannot be acquired."""


class PersistentCacheLeaseTimeout(PersistentCacheLeaseError):
    """Raised when an explicitly bounded persistent-cache lease wait expires."""

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        super().__init__(
            f"timed out after {timeout:g}s waiting for persistent native cache lease"
        )


@contextmanager
def persistent_cache_lease(
    library_path: Path,
    *,
    timeout: float | None = None,
) -> Iterator[None]:
    """Hold one cross-process lease for a persistent-cache digest.

    The lease is backed by an operating-system file lock, so it is released when
    the file descriptor closes or the owning process exits unexpectedly. An
    explicit timeout uses non-blocking polling on every supported platform.
    """
    lock_path = _lock_path(library_path)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        stream = lock_path.open("a+b")
    except OSError as error:
        raise PersistentCacheLeaseError(
            f"failed to open persistent native cache lease: {error}"
        ) from error

    locked = False
    try:
        _lock_stream(stream, timeout=timeout)
        locked = True
        yield
    finally:
        try:
            if locked:
                _unlock_stream(stream)
        finally:
            stream.close()


def _lock_path(library_path: Path) -> Path:
    digest = library_path.parent.name
    return library_path.parent.parent / f".{digest}.lock"


def _lock_stream(stream, *, timeout: float | None = None) -> None:
    if os.name == "nt":
        _lock_stream_windows(stream, timeout=timeout)
        return
    if os.name == "posix":
        _lock_stream_posix(stream, timeout=timeout)
        return
    raise PersistentCacheLeaseError(
        f"persistent native cache leases are unsupported on platform: {os.name}"
    )


def _lock_stream_posix(stream, *, timeout: float | None) -> None:
    import fcntl

    if timeout is None:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        return

    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError as error:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise PersistentCacheLeaseTimeout(timeout) from error
            time.sleep(min(_PERSISTENT_CACHE_LEASE_POLL_SECONDS, remaining))


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
    raise PersistentCacheLeaseError(
        f"persistent native cache leases are unsupported on platform: {os.name}"
    )


def _lock_stream_windows(stream, *, timeout: float | None) -> None:
    import msvcrt

    # Windows byte-range locks may extend past EOF. Do not initialize the lock
    # file before acquiring the byte-range lock: concurrent first-time lockers
    # can otherwise race in write/flush before either process owns the lock.
    stream.seek(0)

    effective_timeout = (
        _PERSISTENT_CACHE_LEASE_TIMEOUT_SECONDS if timeout is None else timeout
    )
    deadline = time.monotonic() + effective_timeout
    while True:
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError as error:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                if timeout is None:
                    raise PersistentCacheLeaseError(
                        "timed out waiting for persistent native cache lease"
                    ) from error
                raise PersistentCacheLeaseTimeout(timeout) from error
            time.sleep(min(_PERSISTENT_CACHE_LEASE_POLL_SECONDS, remaining))
