import contextlib
import multiprocessing
import os
import time
from pathlib import Path

import pytest

import tiny_tensor_compiler.native as native_module


def _wait_for(path: Path, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.02)


def _hold_persistent_lease(
    library: str,
    ready: str,
    release: str,
) -> None:
    library_path = Path(library)
    ready_path = Path(ready)
    release_path = Path(release)
    with native_module._persistent_cache_lease(library_path):
        ready_path.write_text("ready", encoding="utf-8")
        _wait_for(release_path)


def _acquire_persistent_lease(
    library: str,
    started: str,
    acquired: str,
) -> None:
    library_path = Path(library)
    Path(started).write_text("started", encoding="utf-8")
    with native_module._persistent_cache_lease(library_path):
        Path(acquired).write_text("acquired", encoding="utf-8")


def _acquire_persistent_lease_and_exit(library: str, acquired: str) -> None:
    lease = native_module._persistent_cache_lease(Path(library))
    lease.__enter__()
    Path(acquired).write_text("acquired", encoding="utf-8")
    os._exit(0)


def test_persistent_cache_lease_is_exclusive_across_processes(tmp_path):
    library = tmp_path / native_module._PERSISTENT_CACHE_SCHEMA / "digest" / native_module._library_name()
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    started = tmp_path / "started"
    acquired = tmp_path / "acquired"
    context = multiprocessing.get_context("spawn")

    holder = context.Process(
        target=_hold_persistent_lease,
        args=(str(library), str(ready), str(release)),
    )
    waiter = context.Process(
        target=_acquire_persistent_lease,
        args=(str(library), str(started), str(acquired)),
    )
    holder.start()
    try:
        _wait_for(ready)
        waiter.start()
        _wait_for(started)
        time.sleep(0.2)
        assert not acquired.exists()

        release.write_text("release", encoding="utf-8")
        holder.join(timeout=20)
        waiter.join(timeout=20)
        assert holder.exitcode == 0
        assert waiter.exitcode == 0
        assert acquired.read_text(encoding="utf-8") == "acquired"
    finally:
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)
        if waiter.pid is not None and waiter.is_alive():
            waiter.terminate()
            waiter.join(timeout=5)


def test_persistent_cache_lease_does_not_mutate_lock_file(tmp_path):
    library = tmp_path / native_module._PERSISTENT_CACHE_SCHEMA / "digest" / native_module._library_name()
    lock_path = library.parent.parent / ".digest.lock"

    with native_module._persistent_cache_lease(library):
        assert lock_path.exists()
        assert lock_path.read_bytes() == b""

    assert lock_path.read_bytes() == b""


def test_persistent_cache_leases_are_independent_per_digest(tmp_path):
    schema = tmp_path / native_module._PERSISTENT_CACHE_SCHEMA
    held_library = schema / "digest-a" / native_module._library_name()
    other_library = schema / "digest-b" / native_module._library_name()
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    started = tmp_path / "started"
    acquired = tmp_path / "acquired"
    context = multiprocessing.get_context("spawn")

    holder = context.Process(
        target=_hold_persistent_lease,
        args=(str(held_library), str(ready), str(release)),
    )
    waiter = context.Process(
        target=_acquire_persistent_lease,
        args=(str(other_library), str(started), str(acquired)),
    )
    holder.start()
    try:
        _wait_for(ready)
        waiter.start()
        _wait_for(started)
        _wait_for(acquired)
        assert holder.is_alive()

        release.write_text("release", encoding="utf-8")
        holder.join(timeout=20)
        waiter.join(timeout=20)
        assert holder.exitcode == 0
        assert waiter.exitcode == 0
    finally:
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)
        if waiter.pid is not None and waiter.is_alive():
            waiter.terminate()
            waiter.join(timeout=5)


def test_persistent_cache_lease_releases_when_owner_process_exits(tmp_path):
    library = tmp_path / native_module._PERSISTENT_CACHE_SCHEMA / "digest" / native_module._library_name()
    child_acquired = tmp_path / "child-acquired"
    parent_acquired = tmp_path / "parent-acquired"
    started = tmp_path / "parent-started"
    context = multiprocessing.get_context("spawn")

    owner = context.Process(
        target=_acquire_persistent_lease_and_exit,
        args=(str(library), str(child_acquired)),
    )
    owner.start()
    _wait_for(child_acquired)
    owner.join(timeout=20)
    assert owner.exitcode == 0

    follower = context.Process(
        target=_acquire_persistent_lease,
        args=(str(library), str(started), str(parent_acquired)),
    )
    follower.start()
    try:
        follower.join(timeout=20)
        assert follower.exitcode == 0
        assert parent_acquired.read_text(encoding="utf-8") == "acquired"
    finally:
        if follower.is_alive():
            follower.terminate()
            follower.join(timeout=5)


def test_persistent_cache_rechecks_verified_entry_after_lease(monkeypatch, tmp_path):
    library = tmp_path / native_module._PERSISTENT_CACHE_SCHEMA / "digest" / native_module._library_name()
    artifact = object()
    lease_active = False

    @contextlib.contextmanager
    def synthetic_lease(path):
        nonlocal lease_active
        assert path == library
        lease_active = True
        try:
            yield
        finally:
            lease_active = False

    def staged(path):
        assert lease_active
        assert path == library
        return artifact

    def must_not_compile(*args, **kwargs):
        pytest.fail("persistent cache follower recompiled after the lease was acquired")

    monkeypatch.setattr(native_module, "_persistent_cache_lease", synthetic_lease)
    monkeypatch.setattr(native_module, "_stage_existing_persistent_artifact", staged)
    monkeypatch.setattr(native_module, "_compile_source", must_not_compile)

    result = native_module._get_or_compile_persistent_artifact(
        "source",
        ["cc"],
        library,
    )

    assert result is artifact
