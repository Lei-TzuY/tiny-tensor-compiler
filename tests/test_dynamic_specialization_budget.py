import numpy as np
import pytest

import tiny_tensor_compiler.compiler as compiler_module
from tiny_tensor_compiler import CompileBudget, GraphBuilder, SymbolicDim
from tiny_tensor_compiler.admission import DynamicSpecializationBudgetExceeded


class _FakeNativeExecutable:
    def __call__(self, inputs=(), out=None):
        result = np.array(inputs[0], copy=True) if inputs else np.array(0, dtype=np.int32)
        if out is not None:
            np.copyto(out, result)
            return out
        return result


class _FakeAdaptiveExecutable:
    backend = "loop"


def _single_symbol_module():
    builder = GraphBuilder("dynamic-specialization-budget")
    batch = SymbolicDim("B")
    source = builder.input((batch,), "int32")
    return batch, builder.finish(source)


def test_compile_budget_validates_dynamic_specialization_limit():
    assert CompileBudget().max_dynamic_specializations is None
    assert CompileBudget(max_dynamic_specializations=0).max_dynamic_specializations == 0
    assert CompileBudget(max_dynamic_specializations=3).max_dynamic_specializations == 3

    for value in (-1, True, 1.5, "4"):
        with pytest.raises((TypeError, ValueError)):
            CompileBudget(max_dynamic_specializations=value)  # type: ignore[arg-type]


def test_dynamic_cap_allows_cache_hits_and_rejects_new_binding_before_specialization(
    monkeypatch,
):
    _batch, module = _single_symbol_module()
    specialize_calls = []
    compile_calls = []
    original_specialize = compiler_module.specialize_module

    def tracked_specialize(module, bindings):
        specialize_calls.append(tuple(sorted((symbol.name, size) for symbol, size in bindings.items())))
        return original_specialize(module, bindings)

    def fake_compile(module, **kwargs):
        compile_calls.append(module.function.name)
        return _FakeNativeExecutable()

    monkeypatch.setattr(compiler_module, "specialize_module", tracked_specialize)
    monkeypatch.setattr(compiler_module, "compile_module", fake_compile)

    executable = compiler_module.compile_dynamic_module(
        module,
        budget=CompileBudget(max_dynamic_specializations=2),
    )

    first = executable.specialize(2)
    second = executable.specialize(5)
    assert executable.specialize(2) is first
    assert executable.specialize(5) is second
    assert executable.cached_bindings == ((('B', 2),), (('B', 5),))
    assert len(specialize_calls) == 2
    assert len(compile_calls) == 2

    with pytest.raises(DynamicSpecializationBudgetExceeded) as exc_info:
        executable.specialize(7)

    error = exc_info.value
    assert error.limit == 2
    assert error.attempted_binding == (("B", 7),)
    assert error.cached_bindings == ((('B', 2),), (('B', 5),))
    assert len(specialize_calls) == 2
    assert len(compile_calls) == 2
    assert executable.cached_bindings == ((('B', 2),), (('B', 5),))


def test_dynamic_execute_obeys_zero_specialization_cap_before_compile(monkeypatch):
    _batch, module = _single_symbol_module()

    def forbidden_compile(*args, **kwargs):
        raise AssertionError("native compilation must not run after specialization-cap rejection")

    monkeypatch.setattr(compiler_module, "compile_module", forbidden_compile)

    executable = compiler_module.compile_dynamic_module(
        module,
        budget=CompileBudget(max_dynamic_specializations=0),
    )
    value = np.array([1, 2], dtype=np.int32)

    with pytest.raises(DynamicSpecializationBudgetExceeded) as exc_info:
        executable(inputs=[value])
    assert exc_info.value.limit == 0
    assert exc_info.value.attempted_binding == (("B", 2),)
    assert exc_info.value.cached_bindings == ()
    assert executable.cached_bindings == ()


def test_failed_dynamic_compile_does_not_consume_specialization_capacity(monkeypatch):
    _batch, module = _single_symbol_module()
    calls = 0

    def flaky_compile(module, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic compiler failure")
        return _FakeNativeExecutable()

    monkeypatch.setattr(compiler_module, "compile_module", flaky_compile)
    executable = compiler_module.compile_dynamic_module(
        module,
        budget=CompileBudget(max_dynamic_specializations=1),
    )

    with pytest.raises(RuntimeError, match="synthetic compiler failure"):
        executable.specialize(2)
    assert executable.cached_bindings == ()

    executable.specialize(3)
    assert executable.cached_bindings == ((('B', 3),),)
    with pytest.raises(DynamicSpecializationBudgetExceeded):
        executable.specialize(4)
    assert calls == 2


def test_multi_symbol_budget_diagnostics_use_canonical_symbol_order(monkeypatch):
    builder = GraphBuilder("multi-symbol-specialization-budget")
    batch = SymbolicDim("B")
    width = SymbolicDim("W")
    source = builder.input((batch, width), "int32")
    module = builder.finish(source)

    monkeypatch.setattr(
        compiler_module,
        "compile_module",
        lambda module, **kwargs: _FakeNativeExecutable(),
    )
    executable = compiler_module.compile_dynamic_module(
        module,
        budget=CompileBudget(max_dynamic_specializations=1),
    )

    executable.specialize({width: 3, batch: 2})
    assert executable.cached_bindings == ((('B', 2), ('W', 3)),)

    with pytest.raises(DynamicSpecializationBudgetExceeded) as exc_info:
        executable.specialize({width: 5, batch: 4})
    assert exc_info.value.attempted_binding == (("B", 4), ("W", 5))
    assert exc_info.value.cached_bindings == ((('B', 2), ('W', 3)),)


def test_adaptive_dynamic_cap_counts_backend_decisions(monkeypatch):
    _batch, module = _single_symbol_module()
    adaptive_calls = []

    def fake_adaptive_compile(module, **kwargs):
        adaptive_calls.append(module.function.name)
        return _FakeAdaptiveExecutable()

    monkeypatch.setattr(compiler_module, "compile_adaptive_module", fake_adaptive_compile)
    budget = CompileBudget(max_dynamic_specializations=1)
    executable = compiler_module.compile_adaptive_dynamic_module(module, budget=budget)

    first = executable.specialize(2)
    assert first.backend == "loop"
    assert executable.cached_binding_backends == (((('B', 2),), "loop"),)

    with pytest.raises(DynamicSpecializationBudgetExceeded):
        executable.specialize(3)
    assert len(adaptive_calls) == 1
    assert executable.cached_bindings == ((('B', 2),),)
