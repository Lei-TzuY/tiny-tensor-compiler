from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tiny_tensor_compiler import GraphBuilder, SymbolicDim, compile_module
from tiny_tensor_compiler import native as native_module
from tiny_tensor_compiler.managed_native import (
    compile_resource_managed_module,
    manage_native_executable,
)
from tiny_tensor_compiler.parallel_native import ParallelNativeExecutable
from tiny_tensor_compiler.specialization_cache import compile_resource_managed_dynamic_module


def _relu_module(shape=(2, 4)):
    builder = GraphBuilder()
    value = builder.input(shape, dtype="float32")
    return builder.finish(value.relu())


def _dynamic_relu_module():
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    value = builder.input((batch, 4), dtype="float32")
    return batch, builder.finish(value.relu())


def _artifact_directories() -> set[Path]:
    with native_module._NATIVE_CACHE_LOCK:
        return {artifact.directory for artifact in native_module._NATIVE_CACHE.values()}


def _inputs(rows: int = 2) -> np.ndarray:
    return np.arange(rows * 4, dtype=np.float32).reshape(rows, 4) - 3


def test_resource_managed_concrete_close_releases_final_serial_artifact():
    native_module.clear_native_cache()
    managed = compile_resource_managed_module(_relu_module())
    directories = _artifact_directories()
    assert len(directories) == 1
    assert all(path.exists() for path in directories)

    values = _inputs()
    np.testing.assert_array_equal(managed(inputs=[values]), np.maximum(values, 0))

    assert managed.close() is True
    assert managed.closed is True
    assert all(not path.exists() for path in directories)
    with pytest.raises(RuntimeError, match="closed"):
        managed(inputs=[values])
    assert managed.close() is False


def test_resource_managed_concrete_owners_release_only_after_final_close():
    native_module.clear_native_cache()
    module = _relu_module()
    left = compile_resource_managed_module(module)
    shared_directories = _artifact_directories()
    right = compile_resource_managed_module(module)
    assert _artifact_directories() == shared_directories

    assert left.close() is False
    assert all(path.exists() for path in shared_directories)
    np.testing.assert_array_equal(right(inputs=[_inputs()]), np.maximum(_inputs(), 0))

    assert right.close() is True
    assert all(not path.exists() for path in shared_directories)


def test_resource_managed_concrete_shares_ownership_with_dynamic_handle():
    native_module.clear_native_cache()
    batch, module = _dynamic_relu_module()
    dynamic = compile_resource_managed_dynamic_module(
        module,
        max_cached_specializations=1,
    )

    first = dynamic.specialize({batch: 2})
    lease = manage_native_executable(first)
    first_directories = _artifact_directories()
    dynamic.specialize({batch: 3})

    assert dynamic.eviction_count == 1
    assert dynamic.released_native_artifact_count == 0
    assert first_directories <= _artifact_directories()
    assert all(path.exists() for path in first_directories)

    assert lease.close() is True
    assert all(not path.exists() for path in first_directories)


def test_managed_lease_adopts_ordinary_handle_without_changing_ordinary_reacquire_semantics():
    native_module.clear_native_cache()
    ordinary = compile_module(_relu_module())
    lease = manage_native_executable(ordinary)
    original_directories = _artifact_directories()

    assert lease.close() is True
    assert all(not path.exists() for path in original_directories)

    values = _inputs()
    np.testing.assert_array_equal(ordinary(inputs=[values]), np.maximum(values, 0))
    reacquired_directories = _artifact_directories()
    assert len(reacquired_directories) == 1
    assert all(path.exists() for path in reacquired_directories)


def test_resource_managed_concrete_tolerates_global_clear_and_releases_reacquired_artifact():
    native_module.clear_native_cache()
    managed = compile_resource_managed_module(_relu_module())
    native_module.clear_native_cache()
    assert not _artifact_directories()

    values = _inputs()
    np.testing.assert_array_equal(managed(inputs=[values]), np.maximum(values, 0))
    reacquired_directories = _artifact_directories()
    assert len(reacquired_directories) == 1

    assert managed.close() is True
    assert all(not path.exists() for path in reacquired_directories)


def test_resource_managed_concrete_preserves_persistent_artifact_on_close(
    tmp_path,
    monkeypatch,
):
    native_module.clear_native_cache()
    cache_dir = tmp_path / "cache"
    managed = compile_resource_managed_module(_relu_module(), cache_dir=cache_dir)
    persistent_library = managed.executable._persistent_library
    assert persistent_library is not None
    assert persistent_library.exists()

    assert managed.close() is True
    assert persistent_library.exists()

    def fail_compile(*args, **kwargs):
        raise AssertionError("managed close must not delete the persistent native artifact")

    monkeypatch.setattr(native_module, "_compile_source", fail_compile)
    ordinary = compile_module(_relu_module(), cache_dir=cache_dir)
    values = _inputs()
    np.testing.assert_array_equal(ordinary(inputs=[values]), np.maximum(values, 0))


def test_resource_managed_concrete_context_manager_closes_handle():
    native_module.clear_native_cache()
    values = _inputs()
    with compile_resource_managed_module(_relu_module()) as managed:
        np.testing.assert_array_equal(managed(inputs=[values]), np.maximum(values, 0))
        assert managed.closed is False
    assert managed.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        managed(inputs=[values])


def test_resource_managed_concrete_rejects_parallel_and_invalid_adoption():
    with pytest.raises(ValueError, match="process-pinned"):
        compile_resource_managed_module(_relu_module(), parallel=True)

    with pytest.raises(TypeError, match="NativeExecutable"):
        manage_native_executable(object())

    parallel = object.__new__(ParallelNativeExecutable)
    with pytest.raises(ValueError, match="process-pinned"):
        manage_native_executable(parallel)
