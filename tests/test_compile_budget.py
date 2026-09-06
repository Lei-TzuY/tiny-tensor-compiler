import numpy as np
import pytest

import tiny_tensor_compiler.compiler as compiler_module
from tiny_tensor_compiler import (
    CompileBudget,
    CompileBudgetExceeded,
    GraphBuilder,
    SymbolicDim,
    compile_dynamic_module,
)
from tiny_tensor_compiler.admission import enforce_compile_budget


def test_compile_budget_validates_bounds_strictly():
    assert CompileBudget() == CompileBudget(
        max_planned_storage_bytes=None,
        max_post_fusion_kernels=None,
    )
    assert CompileBudget(max_planned_storage_bytes=0).max_planned_storage_bytes == 0
    assert CompileBudget(max_post_fusion_kernels=3).max_post_fusion_kernels == 3

    for value in (-1, True, 1.5, "4"):
        with pytest.raises((TypeError, ValueError)):
            CompileBudget(max_planned_storage_bytes=value)  # type: ignore[arg-type]
        with pytest.raises((TypeError, ValueError)):
            CompileBudget(max_post_fusion_kernels=value)  # type: ignore[arg-type]


def test_storage_budget_is_inclusive_and_reports_exact_metric():
    builder = GraphBuilder("storage-budget")
    source = builder.input((8,), "int32")
    module = builder.finish(source)

    report = enforce_compile_budget(
        module,
        CompileBudget(max_planned_storage_bytes=32),
    )
    assert report.planned_owning_storage_bytes == 32

    with pytest.raises(CompileBudgetExceeded) as exc_info:
        enforce_compile_budget(
            module,
            CompileBudget(max_planned_storage_bytes=31),
        )

    error = exc_info.value
    assert error.metric == "planned_owning_storage_bytes"
    assert error.limit == 31
    assert error.actual == 32
    assert error.report.planned_owning_storage_bytes == 32
    assert str(error) == (
        "compile budget exceeded: planned_owning_storage_bytes actual 32 exceeds limit 31"
    )


def test_kernel_budget_observes_post_fusion_structure():
    builder = GraphBuilder("kernel-budget")
    lhs = builder.input((8,), "int32")
    rhs = builder.input((8,), "int32")
    result = (lhs + rhs).relu()
    module = builder.finish((lhs, rhs, result))

    report = enforce_compile_budget(
        module,
        CompileBudget(max_post_fusion_kernels=1),
    )
    assert report.pre_fusion_kernel_count == 2
    assert report.post_fusion_kernel_count == 1

    with pytest.raises(CompileBudgetExceeded) as exc_info:
        enforce_compile_budget(
            module,
            CompileBudget(max_post_fusion_kernels=0),
        )
    assert exc_info.value.metric == "post_fusion_kernel_count"
    assert exc_info.value.actual == 1


def test_compile_module_rejects_before_native_compiler(monkeypatch):
    builder = GraphBuilder("pre-native-budget")
    source = builder.input((8,), "int32")
    module = builder.finish(source)

    def forbidden_native_compile(*args, **kwargs):
        raise AssertionError("native compilation must not run after budget rejection")

    monkeypatch.setattr(compiler_module, "compile_native", forbidden_native_compile)

    with pytest.raises(CompileBudgetExceeded) as exc_info:
        compiler_module.compile_module(
            module,
            budget=CompileBudget(max_planned_storage_bytes=31),
        )
    assert exc_info.value.actual == 32


def test_dynamic_budget_is_checked_per_concrete_binding_and_rejection_is_not_cached():
    builder = GraphBuilder("dynamic-budget")
    batch = SymbolicDim("B")
    source = builder.input((batch,), "int32")
    module = builder.finish(source)

    executable = compile_dynamic_module(
        module,
        budget=CompileBudget(max_planned_storage_bytes=8),
    )

    accepted = np.array([3, 7], dtype=np.int32)
    np.testing.assert_array_equal(executable(inputs=[accepted]), accepted)
    assert executable.cached_bindings == ((('B', 2),),)

    rejected = np.array([1, 2, 3], dtype=np.int32)
    with pytest.raises(CompileBudgetExceeded) as exc_info:
        executable(inputs=[rejected])
    assert exc_info.value.metric == "planned_owning_storage_bytes"
    assert exc_info.value.limit == 8
    assert exc_info.value.actual == 12
    assert executable.cached_bindings == ((('B', 2),),)
