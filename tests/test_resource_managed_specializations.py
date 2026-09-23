from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tiny_tensor_compiler import (
    CompileBudget,
    GraphBuilder,
    SymbolicDim,
    differentiate_module,
    jacobian_vector_product_module,
    specialize_module,
    vector_jacobian_product_module,
)
from tiny_tensor_compiler import native as native_module
from tiny_tensor_compiler.analysis import analyze_module
from tiny_tensor_compiler.compiler import _shared_linearization_modules
from tiny_tensor_compiler.specialization_cache import (
    compile_resource_managed_adaptive_dynamic_gradient_module,
    compile_resource_managed_adaptive_dynamic_linearization,
    compile_resource_managed_adaptive_dynamic_hvp_module,
    compile_resource_managed_adaptive_dynamic_jvp_module,
    compile_resource_managed_adaptive_dynamic_module,
    compile_resource_managed_adaptive_dynamic_vjp_module,
    compile_resource_managed_dynamic_gradient_module,
    compile_resource_managed_dynamic_hvp_module,
    compile_resource_managed_dynamic_jvp_module,
    compile_resource_managed_dynamic_linearization,
    compile_resource_managed_dynamic_module,
    compile_resource_managed_dynamic_vjp_module,
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


def test_resource_managed_handles_release_shared_artifact_only_after_last_owner_evicts():
    native_module.clear_native_cache()
    batch, module = _dynamic_relu_module()
    left = compile_resource_managed_dynamic_module(
        module,
        max_cached_specializations=1,
    )
    right = compile_resource_managed_dynamic_module(
        module,
        max_cached_specializations=1,
    )

    left.specialize({batch: 2})
    shared_directories = _artifact_directories()
    assert len(shared_directories) == 1
    right.specialize({batch: 2})
    assert _artifact_directories() == shared_directories

    left.specialize({batch: 3})
    assert left.eviction_count == 1
    assert left.released_native_artifact_count == 0
    assert shared_directories <= _artifact_directories()
    assert all(path.exists() for path in shared_directories)

    right.specialize({batch: 4})
    assert right.eviction_count == 1
    assert right.released_native_artifact_count == 1
    assert all(not path.exists() for path in shared_directories)


def test_resource_managed_ordinary_and_adaptive_handles_share_native_ownership():
    native_module.clear_native_cache()
    batch, module = _dynamic_relu_module()
    ordinary = compile_resource_managed_dynamic_module(
        module,
        max_cached_specializations=1,
    )
    adaptive = compile_resource_managed_adaptive_dynamic_module(
        module,
        budget=CompileBudget(),
        max_cached_specializations=1,
    )

    ordinary.specialize({batch: 2})
    shared_directories = _artifact_directories()
    adaptive_first = adaptive.specialize({batch: 2})
    assert adaptive_first.backend == "native"
    assert _artifact_directories() == shared_directories

    ordinary.specialize({batch: 3})
    assert ordinary.released_native_artifact_count == 0
    assert shared_directories <= _artifact_directories()

    adaptive.specialize({batch: 4})
    assert adaptive.released_native_artifact_count == 1
    assert all(not path.exists() for path in shared_directories)


def test_resource_managed_ownership_tolerates_explicit_global_cache_clear():
    native_module.clear_native_cache()
    batch, module = _dynamic_relu_module()
    left = compile_resource_managed_dynamic_module(
        module,
        max_cached_specializations=1,
    )
    right = compile_resource_managed_dynamic_module(
        module,
        max_cached_specializations=1,
    )

    left.specialize({batch: 2})
    right.specialize({batch: 2})
    native_module.clear_native_cache()
    assert not _artifact_directories()

    left.specialize({batch: 3})
    right.specialize({batch: 4})
    assert left.eviction_count == 1
    assert right.eviction_count == 1
    assert left.released_native_artifact_count == 0
    assert right.released_native_artifact_count == 0


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



def _dynamic_gradient_module():
    batch = SymbolicDim("B")
    builder = GraphBuilder("managed-gradient")
    value = builder.input((batch, 4), dtype="float64")
    weights = builder.input((batch, 4), dtype="float64")
    return batch, builder.finish((value * weights).sum())


def test_resource_managed_dynamic_gradient_evicts_and_releases_native_artifact():
    native_module.clear_native_cache()
    batch, module = _dynamic_gradient_module()
    executable = compile_resource_managed_dynamic_gradient_module(
        module,
        wrt=(0,),
        max_cached_specializations=1,
    )

    first = executable.specialize({batch: 2})
    first_directories = _artifact_directories()
    assert len(first_directories) == 1
    second = executable.specialize({batch: 3})
    assert second is not first
    assert executable.cached_bindings == ((("B", 3),),)
    assert executable.retained_bindings_lru == ((("B", 3),),)
    assert executable.eviction_count == 1
    assert executable.released_native_artifact_count == 1
    assert all(not path.exists() for path in first_directories)

    values = np.arange(8, dtype=np.float64).reshape(2, 4) - 2.0
    weights = np.arange(8, dtype=np.float64).reshape(2, 4) * 0.5 + 1.0
    np.testing.assert_array_equal(first(inputs=(values, weights)), weights)


def test_resource_managed_adaptive_gradient_releases_only_evicted_native_backend():
    native_module.clear_native_cache()
    batch, module = _dynamic_gradient_module()

    small_gradient = differentiate_module(
        specialize_module(module, {batch: 1}),
        wrt=(0,),
    )
    large_gradient = differentiate_module(
        specialize_module(module, {batch: 4}),
        wrt=(0,),
    )
    small_bytes = analyze_module(small_gradient).planned_owning_storage_bytes
    large_bytes = analyze_module(large_gradient).planned_owning_storage_bytes
    assert small_bytes < large_bytes

    executable = compile_resource_managed_adaptive_dynamic_gradient_module(
        module,
        budget=CompileBudget(max_planned_storage_bytes=small_bytes),
        wrt=(0,),
        max_cached_specializations=1,
    )

    small = executable.specialize({batch: 1})
    assert small.backend == "native"
    native_directories = _artifact_directories()
    assert len(native_directories) == 1

    large = executable.specialize({batch: 4})
    assert large.backend == "loop"
    assert executable.cached_binding_backends == (((("B", 4),), "loop"),)
    assert executable.eviction_count == 1
    assert executable.released_native_artifact_count == 1
    assert all(not path.exists() for path in native_directories)

    larger = executable.specialize({batch: 5})
    assert larger.backend == "loop"
    assert executable.eviction_count == 2
    assert executable.released_native_artifact_count == 1



def _dynamic_vjp_module():
    batch = SymbolicDim("B")
    builder = GraphBuilder("managed-vjp")
    value = builder.input((batch, 4), dtype="float64")
    return batch, builder.finish(value * value)


def test_resource_managed_dynamic_vjp_evicts_releases_and_reacquires_native_artifact():
    native_module.clear_native_cache()
    batch, module = _dynamic_vjp_module()
    executable = compile_resource_managed_dynamic_vjp_module(
        module,
        wrt=(0,),
        max_cached_specializations=1,
    )

    first = executable.specialize({batch: 2})
    first_directories = _artifact_directories()
    assert len(first_directories) == 1

    second = executable.specialize({batch: 3})
    assert second is not first
    assert executable.cached_bindings == ((("B", 3),),)
    assert executable.retained_bindings_lru == ((("B", 3),),)
    assert executable.eviction_count == 1
    assert executable.released_native_artifact_count == 1
    assert all(not path.exists() for path in first_directories)

    values = np.arange(8, dtype=np.float64).reshape(2, 4) - 3.0
    cotangent = np.arange(8, dtype=np.float64).reshape(2, 4) * 0.25 + 1.0
    np.testing.assert_allclose(
        first(inputs=(values, cotangent)),
        2.0 * values * cotangent,
        rtol=0.0,
        atol=0.0,
    )


def test_resource_managed_adaptive_dynamic_vjp_releases_only_evicted_native_backend():
    native_module.clear_native_cache()
    batch, module = _dynamic_vjp_module()

    small_vjp = vector_jacobian_product_module(
        specialize_module(module, {batch: 1}),
        wrt=(0,),
    )
    large_vjp = vector_jacobian_product_module(
        specialize_module(module, {batch: 4}),
        wrt=(0,),
    )
    small_bytes = analyze_module(small_vjp).planned_owning_storage_bytes
    large_bytes = analyze_module(large_vjp).planned_owning_storage_bytes
    assert small_bytes < large_bytes

    executable = compile_resource_managed_adaptive_dynamic_vjp_module(
        module,
        budget=CompileBudget(max_planned_storage_bytes=small_bytes),
        wrt=(0,),
        max_cached_specializations=1,
    )

    small = executable.specialize({batch: 1})
    assert small.backend == "native"
    native_directories = _artifact_directories()
    assert len(native_directories) == 1

    large = executable.specialize({batch: 4})
    assert large.backend == "loop"
    assert executable.cached_binding_backends == (((("B", 4),), "loop"),)
    assert executable.eviction_count == 1
    assert executable.released_native_artifact_count == 1
    assert all(not path.exists() for path in native_directories)

    larger = executable.specialize({batch: 5})
    assert larger.backend == "loop"
    assert executable.cached_binding_backends == (((("B", 5),), "loop"),)
    assert executable.eviction_count == 2
    assert executable.released_native_artifact_count == 1



def _dynamic_hvp_module():
    batch = SymbolicDim("B")
    builder = GraphBuilder("managed-hvp")
    value = builder.input((batch, 4), dtype="float64")
    return batch, builder.finish((value * value * value).sum())


def test_resource_managed_dynamic_hvp_evicts_releases_and_reacquires_native_artifact():
    native_module.clear_native_cache()
    batch, module = _dynamic_hvp_module()
    executable = compile_resource_managed_dynamic_hvp_module(
        module,
        wrt=(0,),
        max_cached_specializations=1,
    )

    first = executable.specialize({batch: 2})
    first_directories = _artifact_directories()
    assert len(first_directories) == 1

    second = executable.specialize({batch: 3})
    assert second is not first
    assert executable.cached_bindings == ((("B", 3),),)
    assert executable.retained_bindings_lru == ((("B", 3),),)
    assert executable.eviction_count == 1
    assert executable.released_native_artifact_count == 1
    assert all(not path.exists() for path in first_directories)

    values = np.arange(8, dtype=np.float64).reshape(2, 4) * 0.25 - 1.5
    vector = np.arange(8, dtype=np.float64).reshape(2, 4) * -0.5 + 2.0
    np.testing.assert_allclose(
        first(inputs=(values, vector)),
        6.0 * values * vector,
        rtol=0.0,
        atol=0.0,
    )


def test_resource_managed_adaptive_dynamic_hvp_releases_only_evicted_native_backend():
    native_module.clear_native_cache()
    batch, module = _dynamic_hvp_module()

    small_gradient = differentiate_module(
        specialize_module(module, {batch: 1}),
        wrt=(0,),
    )
    small_hvp = vector_jacobian_product_module(
        small_gradient,
        wrt=(0,),
    )
    large_gradient = differentiate_module(
        specialize_module(module, {batch: 4}),
        wrt=(0,),
    )
    large_hvp = vector_jacobian_product_module(
        large_gradient,
        wrt=(0,),
    )
    small_bytes = analyze_module(small_hvp).planned_owning_storage_bytes
    large_bytes = analyze_module(large_hvp).planned_owning_storage_bytes
    assert small_bytes < large_bytes

    executable = compile_resource_managed_adaptive_dynamic_hvp_module(
        module,
        budget=CompileBudget(max_planned_storage_bytes=small_bytes),
        wrt=(0,),
        max_cached_specializations=1,
    )

    small = executable.specialize({batch: 1})
    assert small.backend == "native"
    native_directories = _artifact_directories()
    assert len(native_directories) == 1

    large = executable.specialize({batch: 4})
    assert large.backend == "loop"
    assert executable.cached_binding_backends == (((("B", 4),), "loop"),)
    assert executable.eviction_count == 1
    assert executable.released_native_artifact_count == 1
    assert all(not path.exists() for path in native_directories)

    larger = executable.specialize({batch: 5})
    assert larger.backend == "loop"
    assert executable.cached_binding_backends == (((("B", 5),), "loop"),)
    assert executable.eviction_count == 2
    assert executable.released_native_artifact_count == 1



def _dynamic_jvp_module():
    batch = SymbolicDim("B")
    builder = GraphBuilder("managed-jvp")
    value = builder.input((batch, 4), dtype="float64")
    return batch, builder.finish(value * value)


def test_resource_managed_dynamic_jvp_evicts_releases_and_reacquires_native_artifact():
    native_module.clear_native_cache()
    batch, module = _dynamic_jvp_module()
    executable = compile_resource_managed_dynamic_jvp_module(
        module,
        wrt=(0,),
        max_cached_specializations=1,
    )

    first = executable.specialize({batch: 2})
    first_directories = _artifact_directories()
    assert len(first_directories) == 1

    second = executable.specialize({batch: 3})
    assert second is not first
    assert executable.cached_bindings == ((("B", 3),),)
    assert executable.retained_bindings_lru == ((("B", 3),),)
    assert executable.eviction_count == 1
    assert executable.released_native_artifact_count == 1
    assert all(not path.exists() for path in first_directories)

    values = np.arange(8, dtype=np.float64).reshape(2, 4) - 3.0
    tangent = np.arange(8, dtype=np.float64).reshape(2, 4) * 0.25 + 1.0
    np.testing.assert_allclose(
        first(inputs=(values, tangent)),
        2.0 * values * tangent,
        rtol=0.0,
        atol=0.0,
    )


def test_resource_managed_adaptive_dynamic_jvp_releases_only_evicted_native_backend():
    native_module.clear_native_cache()
    batch, module = _dynamic_jvp_module()

    small_jvp = jacobian_vector_product_module(
        specialize_module(module, {batch: 1}),
        wrt=(0,),
    )
    large_jvp = jacobian_vector_product_module(
        specialize_module(module, {batch: 4}),
        wrt=(0,),
    )
    small_bytes = analyze_module(small_jvp).planned_owning_storage_bytes
    large_bytes = analyze_module(large_jvp).planned_owning_storage_bytes
    assert small_bytes < large_bytes

    executable = compile_resource_managed_adaptive_dynamic_jvp_module(
        module,
        budget=CompileBudget(max_planned_storage_bytes=small_bytes),
        wrt=(0,),
        max_cached_specializations=1,
    )

    small = executable.specialize({batch: 1})
    assert small.backend == "native"
    native_directories = _artifact_directories()
    assert len(native_directories) == 1

    large = executable.specialize({batch: 4})
    assert large.backend == "loop"
    assert executable.cached_binding_backends == (((("B", 4),), "loop"),)
    assert executable.eviction_count == 1
    assert executable.released_native_artifact_count == 1
    assert all(not path.exists() for path in native_directories)

    larger = executable.specialize({batch: 5})
    assert larger.backend == "loop"
    assert executable.cached_binding_backends == (((("B", 5),), "loop"),)
    assert executable.eviction_count == 2
    assert executable.released_native_artifact_count == 1



def _dynamic_linearization_module():
    batch = SymbolicDim("B")
    builder = GraphBuilder("managed-linearization")
    value = builder.input((batch, 4), dtype="float64")
    return batch, builder.finish(value * value)


def _linearization_peak_storage_bytes(module, bindings):
    concrete = specialize_module(module, bindings)
    primal, pushforward, pullback, _ = _shared_linearization_modules(
        concrete,
        wrt=(0,),
    )
    return max(
        analyze_module(component).planned_owning_storage_bytes
        for component in (primal, pushforward, pullback)
    )


def test_resource_managed_dynamic_linearization_evicts_bundle_and_reacquires_components():
    native_module.clear_native_cache()
    batch, module = _dynamic_linearization_module()
    executable = compile_resource_managed_dynamic_linearization(
        module,
        wrt=(0,),
        max_cached_specializations=1,
    )

    first = executable.specialize({batch: 2})
    first_directories = _artifact_directories()
    assert len(first_directories) >= 2

    second = executable.specialize({batch: 3})
    assert second is not first
    assert executable.cached_bindings == ((("B", 3),),)
    assert executable.retained_bindings_lru == ((("B", 3),),)
    assert executable.eviction_count == 1
    assert executable.released_native_artifact_count == len(first_directories)
    assert all(not path.exists() for path in first_directories)

    values = np.arange(8, dtype=np.float64).reshape(2, 4) * 0.25 - 1.5
    tangent = np.arange(8, dtype=np.float64).reshape(2, 4) * -0.5 + 2.0
    cotangent = np.arange(8, dtype=np.float64).reshape(2, 4) * 0.125 + 0.75
    state = first.linearize((values,))
    np.testing.assert_allclose(state.primal, values * values, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        state.pushforward((tangent,)),
        2.0 * values * tangent,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        state.pullback(cotangent),
        2.0 * values * cotangent,
        rtol=0.0,
        atol=0.0,
    )


def test_resource_managed_adaptive_dynamic_linearization_releases_native_bundle_only():
    native_module.clear_native_cache()
    batch, module = _dynamic_linearization_module()
    small_bytes = _linearization_peak_storage_bytes(module, {batch: 1})
    large_bytes = _linearization_peak_storage_bytes(module, {batch: 8})
    assert small_bytes < large_bytes

    executable = compile_resource_managed_adaptive_dynamic_linearization(
        module,
        budget=CompileBudget(max_planned_storage_bytes=small_bytes),
        wrt=(0,),
        max_cached_specializations=1,
    )

    small = executable.specialize({batch: 1})
    assert small.backend == "native"
    native_directories = _artifact_directories()
    assert len(native_directories) >= 2

    large = executable.specialize({batch: 8})
    assert large.backend == "loop"
    assert executable.cached_binding_backends == (((("B", 8),), "loop"),)
    assert executable.eviction_count == 1
    assert executable.released_native_artifact_count == len(native_directories)
    assert all(not path.exists() for path in native_directories)

    larger = executable.specialize({batch: 9})
    assert larger.backend == "loop"
    assert executable.cached_binding_backends == (((("B", 9),), "loop"),)
    assert executable.eviction_count == 2
    assert executable.released_native_artifact_count == len(native_directories)


def test_resource_managed_linearization_bundle_unloads_only_after_final_handle_owner():
    native_module.clear_native_cache()
    batch, module = _dynamic_linearization_module()
    left = compile_resource_managed_dynamic_linearization(
        module,
        wrt=(0,),
        max_cached_specializations=1,
    )
    right = compile_resource_managed_dynamic_linearization(
        module,
        wrt=(0,),
        max_cached_specializations=1,
    )

    left.specialize({batch: 2})
    shared_directories = _artifact_directories()
    assert len(shared_directories) >= 2

    right.specialize({batch: 2})
    assert _artifact_directories() == shared_directories

    left.specialize({batch: 3})
    assert left.eviction_count == 1
    assert left.released_native_artifact_count == 0
    assert shared_directories <= _artifact_directories()
    assert all(path.exists() for path in shared_directories)

    right.specialize({batch: 4})
    assert right.eviction_count == 1
    assert right.released_native_artifact_count == len(shared_directories)
    assert all(not path.exists() for path in shared_directories)
