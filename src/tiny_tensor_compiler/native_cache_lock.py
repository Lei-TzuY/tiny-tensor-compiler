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


class PersistentCacheLeaseDeadlineExceeded(PersistentCacheLeaseError):
    """Raised when the caller's total compile deadline expires while waiting."""


@contextmanager
def persistent_cache_lease(
    library_path: Path,
    *,
    deadline_at: float | None = None,
) -> Iterator[None]:
    """Hold one cross-process lease for a persistent-cache digest.

    The lease is backed by an operating-system file lock, so it is released when
    the file descriptor closes or the owning process exits unexpectedly. When an
    absolute monotonic ``deadline_at`` is supplied, lease acquisition is bounded
    by that same deadline instead of starting an independent timeout budget.
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
        _lock_stream(stream, deadline_at=deadline_at)
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


def _lock_stream(stream, *, deadline_at: float | None = None) -> None:
    if os.name == "nt":
        _lock_stream_windows(stream, deadline_at=deadline_at)
        return
    if os.name == "posix":
        _lock_stream_posix(stream, deadline_at=deadline_at)
        return
    raise PersistentCacheLeaseError(
        f"persistent native cache leases are unsupported on platform: {os.name}"
    )


def _lock_stream_posix(stream, *, deadline_at: float | None) -> None:
    import fcntl

    if deadline_at is None:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        return

    while True:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError as error:
            remaining = deadline_at - time.monotonic()
            if remaining <= 0.0:
                raise PersistentCacheLeaseDeadlineExceeded(
                    "total native compile deadline expired while waiting for persistent cache lease"
                ) from error
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


def _lock_stream_windows(stream, *, deadline_at: float | None) -> None:
    import msvcrt

    # Windows byte-range locks may extend past EOF. Do not initialize the lock
    # file before acquiring the byte-range lock: concurrent first-time lockers
    # can otherwise race in write/flush before either process owns the lock.
    stream.seek(0)

    lease_deadline = time.monotonic() + _PERSISTENT_CACHE_LEASE_TIMEOUT_SECONDS
    while True:
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError as error:
            now = time.monotonic()
            if deadline_at is not None and now >= deadline_at:
                raise PersistentCacheLeaseDeadlineExceeded(
                    "total native compile deadline expired while waiting for persistent cache lease"
                ) from error
            if now >= lease_deadline:
                raise PersistentCacheLeaseError(
                    "timed out waiting for persistent native cache lease"
                ) from error
            remaining = lease_deadline - now
            if deadline_at is not None:
                remaining = min(remaining, deadline_at - now)
            time.sleep(min(_PERSISTENT_CACHE_LEASE_POLL_SECONDS, max(remaining, 0.0)))
