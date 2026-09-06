import numpy as np
import pytest

import tiny_tensor_compiler.compiler as compiler_module
from tiny_tensor_compiler import CompileBudget, GraphBuilder, SymbolicDim
from tiny_tensor_compiler.native import NativeCompilationError


def test_adaptive_compile_selects_native_within_budget():
    builder = GraphBuilder("adaptive-native")
    source = builder.input((2,), "int32")
    module = builder.finish(source.relu())

    executable = compiler_module.compile_adaptive_module(
        module,
        budget=CompileBudget(max_planned_storage_bytes=16),
    )

    assert executable.backend == "native"
    assert executable.budget_exceeded is None
    assert executable.report.planned_owning_storage_bytes <= 16
    values = np.array([-3, 7], dtype=np.int32)
    np.testing.assert_array_equal(executable(inputs=[values]), np.array([0, 7], dtype=np.int32))


def test_adaptive_compile_falls_back_only_for_budget_excess(monkeypatch):
    builder = GraphBuilder("adaptive-loop")
    source = builder.input((4,), "int32")
    result = source.relu()
    module = builder.finish((source, result))

    def forbidden_native_compile(*args, **kwargs):
        raise AssertionError("native compilation must not run for a budget fallback")

    monkeypatch.setattr(compiler_module, "compile_native", forbidden_native_compile)

    executable = compiler_module.compile_adaptive_module(
        module,
        budget=CompileBudget(max_post_fusion_kernels=0),
        borrow_inputs=True,
        parallel=True,
    )

    assert executable.backend == "loop"
    assert executable.budget_exceeded is not None
    assert executable.budget_exceeded.metric == "post_fusion_kernel_count"
    assert executable.budget_exceeded.actual == 1
    assert executable.report is executable.budget_exceeded.report

    values = np.array([-4, 2, -1, 9], dtype=np.int32)
    outputs = executable(inputs=[values])
    assert isinstance(outputs, tuple)
    np.testing.assert_array_equal(outputs[0], values)
    np.testing.assert_array_equal(outputs[1], np.array([0, 2, 0, 9], dtype=np.int32))
    np.testing.assert_array_equal(values, np.array([-4, 2, -1, 9], dtype=np.int32))


def test_adaptive_compile_does_not_swallow_native_compiler_errors(monkeypatch):
    builder = GraphBuilder("adaptive-native-error")
    source = builder.input((2,), "int32")
    module = builder.finish(source)

    def broken_native_compile(*args, **kwargs):
        raise NativeCompilationError("synthetic compiler failure")

    monkeypatch.setattr(compiler_module, "compile_native", broken_native_compile)

    with pytest.raises(NativeCompilationError, match="synthetic compiler failure"):
        compiler_module.compile_adaptive_module(
            module,
            budget=CompileBudget(max_planned_storage_bytes=8),
        )


def test_adaptive_dynamic_decides_and_caches_each_binding_independently():
    builder = GraphBuilder("adaptive-dynamic")
    batch = SymbolicDim("B")
    source = builder.input((batch,), "int32")
    module = builder.finish(source)

    executable = compiler_module.compile_adaptive_dynamic_module(
        module,
        budget=CompileBudget(max_planned_storage_bytes=8),
    )

    small = np.array([3, 7], dtype=np.int32)
    np.testing.assert_array_equal(executable(inputs=[small]), small)
    small_specialization = executable.specialize(2)
    assert small_specialization.backend == "native"

    large = np.array([1, 2, 3], dtype=np.int32)
    np.testing.assert_array_equal(executable(inputs=[large]), large)
    large_specialization = executable.specialize(3)
    assert large_specialization.backend == "loop"
    assert large_specialization.budget_exceeded is not None
    assert large_specialization.budget_exceeded.actual == 12

    assert executable.specialize(2) is small_specialization
    assert executable.specialize(3) is large_specialization
    assert executable.cached_bindings == ((("B", 2),), (("B", 3),))
    assert executable.cached_binding_backends == (
        ((("B", 2),), "native"),
        ((("B", 3),), "loop"),
    )


def test_adaptive_entrypoints_require_an_explicit_budget():
    builder = GraphBuilder("adaptive-budget-type")
    source = builder.input((1,), "int32")
    module = builder.finish(source)

    with pytest.raises(TypeError, match="budget must be a CompileBudget"):
        compiler_module.compile_adaptive_module(module, budget=None)  # type: ignore[arg-type]

    dynamic_builder = GraphBuilder("adaptive-dynamic-budget-type")
    batch = SymbolicDim("B")
    dynamic = dynamic_builder.finish(dynamic_builder.input((batch,), "int32"))
    with pytest.raises(TypeError, match="budget must be a CompileBudget"):
        compiler_module.compile_adaptive_dynamic_module(  # type: ignore[arg-type]
            dynamic,
            budget=None,
        )
