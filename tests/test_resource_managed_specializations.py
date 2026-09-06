from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tiny_tensor_compiler import CompileBudget, GraphBuilder, SymbolicDim
from tiny_tensor_compiler import native as native_module
from tiny_tensor_compiler.specialization_cache import (
    compile_resource_managed_adaptive_dynamic_module,
    compile_resource_managed_dynamic_module,
)


def _dynamic_relu_module():
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    value = builder.input((batch, 4), dtype="float32")
    return batch, builder.finish(value.relu())


def _artifact_directories() -> set[Path]:
    with native_module._NATIVE_CACHE_LOCK:
        return {artifact.directory for artifact in native_module._NATIVE_CACHE.values()}


def test_resource_managed_dynamic_evicts_lru_and_releases_serial_artifact():
    native_module.clear_native_cache()
    batch, module = _dynamic_relu_module()
    executable = compile_resource_managed_dynamic_module(
        module,
        max_cached_specializations=1,
    )

    first = executable.specialize({batch: 2})
    first_directories = _artifact_directories()
    assert len(first_directories) == 1
    assert all(path.exists() for path in first_directories)

    second = executable.specialize({batch: 3})
    assert second is not first
    assert executable.cached_bindings == ((("B", 3),),)
    assert executable.retained_bindings_lru == ((("B", 3),),)
    assert executable.eviction_count == 1
    assert executable.released_native_artifact_count == 1
    assert all(not path.exists() for path in first_directories)

    result = first(
        inputs=[np.arange(8, dtype=np.float32).reshape(2, 4) - 3],
    )
    np.testing.assert_array_equal(
        result,
        np.maximum(np.arange(8, dtype=np.float32).reshape(2, 4) - 3, 0),
    )


def test_resource_managed_dynamic_refreshes_lru_on_cache_hit():
    native_module.clear_native_cache()
    batch, module = _dynamic_relu_module()
    executable = compile_resource_managed_dynamic_module(
        module,
        max_cached_specializations=2,
    )

    first = executable.specialize({batch: 2})
    executable.specialize({batch: 3})
    assert executable.retained_bindings_lru == (
        (("B", 2),),
        (("B", 3),),
    )

    assert executable.specialize({batch: 2}) is first
    assert executable.retained_bindings_lru == (
        (("B", 3),),
        (("B", 2),),
    )

    executable.specialize({batch: 4})
    assert executable.cached_bindings == ((("B", 2),), (("B", 4),))
    assert executable.retained_bindings_lru == (
        (("B", 2),),
        (("B", 4),),
    )
    assert executable.eviction_count == 1


def test_resource_managed_dynamic_zero_retention_returns_reacquirable_handle():
    native_module.clear_native_cache()
    batch, module = _dynamic_relu_module()
    executable = compile_resource_managed_dynamic_module(
        module,
        max_cached_specializations=0,
    )

    compiled = executable.specialize({batch: 2})
    assert executable.cached_bindings == ()
    assert executable.eviction_count == 1
    assert executable.released_native_artifact_count == 1
    assert not _artifact_directories()

    inputs = np.array([[-1.0, 2.0, -3.0, 4.0], [5.0, -6.0, 7.0, -8.0]], dtype=np.float32)
    np.testing.assert_array_equal(compiled(inputs=[inputs]), np.maximum(inputs, 0))


def test_resource_managed_dynamic_reloads_evicted_persistent_artifact_without_compiler(
    tmp_path,
    monkeypatch,
):
    native_module.clear_native_cache()
    batch, module = _dynamic_relu_module()
    executable = compile_resource_managed_dynamic_module(
        module,
        cache_dir=tmp_path / "cache",
        max_cached_specializations=1,
    )

    first = executable.specialize({batch: 2})
    executable.specialize({batch: 3})
    assert executable.released_native_artifact_count == 1

    def fail_compile(*args, **kwargs):
        raise AssertionError("evicted persistent specialization should not invoke the compiler")

    monkeypatch.setattr(native_module, "_compile_source", fail_compile)
    inputs = np.arange(8, dtype=np.float32).reshape(2, 4) - 4
    np.testing.assert_array_equal(first(inputs=[inputs]), np.maximum(inputs, 0))


def test_resource_managed_adaptive_releases_native_specialization_artifact():
    native_module.clear_native_cache()
    batch, module = _dynamic_relu_module()
    executable = compile_resource_managed_adaptive_dynamic_module(
        module,
        budget=CompileBudget(),
        max_cached_specializations=1,
    )

    first = executable.specialize({batch: 2})
    first_directories = _artifact_directories()
    second = executable.specialize({batch: 3})
    assert first.backend == "native"
    assert second.backend == "native"
    assert executable.cached_bindings == ((("B", 3),),)
    assert executable.eviction_count == 1
    assert executable.released_native_artifact_count == 1
    assert all(not path.exists() for path in first_directories)

    inputs = np.arange(8, dtype=np.float32).reshape(2, 4) - 3
    np.testing.assert_array_equal(first(inputs=[inputs]), np.maximum(inputs, 0))


def test_resource_managed_adaptive_evicts_loop_decisions_without_native_release():
    batch, module = _dynamic_relu_module()
    executable = compile_resource_managed_adaptive_dynamic_module(
        module,
        budget=CompileBudget(max_planned_storage_bytes=0),
        max_cached_specializations=1,
    )

    first = executable.specialize({batch: 2})
    second = executable.specialize({batch: 3})
    assert first.backend == "loop"
    assert second.backend == "loop"
    assert executable.cached_bindings == ((("B", 3),),)
    assert executable.eviction_count == 1
    assert executable.released_native_artifact_count == 0


def test_resource_managed_retention_rejects_unsupported_or_ambiguous_policies():
    _, module = _dynamic_relu_module()

    with pytest.raises(TypeError, match="max_cached_specializations"):
        compile_resource_managed_dynamic_module(module, max_cached_specializations=True)
    with pytest.raises(ValueError, match="max_cached_specializations"):
        compile_resource_managed_dynamic_module(module, max_cached_specializations=-1)
    with pytest.raises(ValueError, match="process-pinned"):
        compile_resource_managed_dynamic_module(
            module,
            max_cached_specializations=1,
            parallel=True,
        )
    with pytest.raises(ValueError, match="max_dynamic_specializations"):
        compile_resource_managed_dynamic_module(
            module,
            max_cached_specializations=1,
            budget=CompileBudget(max_dynamic_specializations=2),
        )
