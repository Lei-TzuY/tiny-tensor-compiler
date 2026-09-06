import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

import tiny_tensor_compiler.compiler as compiler_module
import tiny_tensor_compiler.native as native_module
from tiny_tensor_compiler import (
    CompileBudget,
    GraphBuilder,
    NativeCompilationTimeout,
    SymbolicDim,
    clear_native_cache,
    compile_native,
    lower_to_cpu,
    lower_to_loops,
)


@pytest.fixture(autouse=True)
def _clear_native_artifact_cache():
    clear_native_cache()
    yield
    clear_native_cache()


def _loop_program():
    builder = GraphBuilder()
    value = builder.tensor([1, -2, 3], dtype="int32")
    return lower_to_loops(lower_to_cpu(builder.finish(value.relu())))


def _sleeping_compiler(monkeypatch):
    command = [sys.executable, "-c", "import time; time.sleep(10)"]
    monkeypatch.setattr(native_module, "_build_compile_command", lambda *args: command)
    return tuple(command)


@pytest.mark.parametrize("bad", [True, "1", object()])
def test_compiler_timeout_rejects_non_numeric_values_before_compiler_lookup(bad):
    with pytest.raises(TypeError):
        compile_native(_loop_program(), compiler="definitely-not-a-compiler", compiler_timeout=bad)


@pytest.mark.parametrize("bad", [0, -1, math.nan, math.inf, -math.inf])
def test_compiler_timeout_requires_positive_finite_seconds(bad):
    with pytest.raises(ValueError):
        compile_native(_loop_program(), compiler="definitely-not-a-compiler", compiler_timeout=bad)


@pytest.mark.parametrize("parallel", [False, True])
def test_compiler_timeout_bounds_real_external_process_and_cleans_transient_build(
    tmp_path,
    monkeypatch,
    parallel,
):
    expected_command = _sleeping_compiler(monkeypatch)
    created: list[Path] = []
    real_mkdtemp = native_module.tempfile.mkdtemp

    def tracking_mkdtemp(*args, **kwargs):
        directory = Path(real_mkdtemp(*args, **kwargs))
        created.append(directory)
        return str(directory)

    monkeypatch.setattr(native_module.tempfile, "mkdtemp", tracking_mkdtemp)

    started = time.monotonic()
    with pytest.raises(NativeCompilationTimeout) as caught:
        compile_native(
            _loop_program(),
            compiler="ignored-by-test",
            compiler_timeout=0.05,
            parallel=parallel,
        )
    elapsed = time.monotonic() - started

    assert caught.value.timeout == 0.05
    assert caught.value.command == expected_command
    assert elapsed < 3.0
    assert created
    assert all(not directory.exists() for directory in created)


def test_persistent_timeout_never_publishes_partial_artifact(tmp_path, monkeypatch):
    _sleeping_compiler(monkeypatch)
    cache_dir = tmp_path / "native-cache"

    with pytest.raises(NativeCompilationTimeout):
        compile_native(
            _loop_program(),
            compiler="ignored-by-test",
            cache_dir=cache_dir,
            compiler_timeout=0.05,
        )

    assert not list(cache_dir.rglob("manifest.json"))
    assert not list(cache_dir.rglob(native_module._library_name()))
    assert not [path for path in cache_dir.rglob(".build-*") if path.is_dir()]


def test_persistent_cache_hit_does_not_reinvoke_compiler_for_tighter_timeout(
    tmp_path,
    monkeypatch,
):
    compiler = "cl" if os.name == "nt" else "cc"
    if shutil.which(compiler) is None:
        pytest.skip(f"no platform default C compiler available: {compiler}")

    program = _loop_program()
    cache_dir = tmp_path / "native-cache"
    compile_native(program, compiler=compiler, cache_dir=cache_dir)
    clear_native_cache()

    def unexpected_run(*args, **kwargs):
        raise AssertionError("persistent cache hit must not launch the compiler")

    monkeypatch.setattr(native_module.subprocess, "run", unexpected_run)
    compile_native(
        program,
        compiler=compiler,
        cache_dir=cache_dir,
        compiler_timeout=1e-9,
    )


def test_compile_module_forwards_explicit_timeout_without_changing_default_call_shape(monkeypatch):
    builder = GraphBuilder()
    module = builder.finish(builder.tensor([1, 2], dtype="int32").relu())
    calls = []
    sentinel = object()

    def fake_compile_native(program, compiler=None, cache_dir=None, **kwargs):
        calls.append((compiler, cache_dir, kwargs))
        return sentinel

    monkeypatch.setattr(compiler_module, "compile_native", fake_compile_native)

    assert compiler_module.compile_module(module) is sentinel
    assert calls[-1] == (None, None, {})
    assert compiler_module.compile_module(module, compiler_timeout=1.25) is sentinel
    assert calls[-1] == (None, None, {"compiler_timeout": 1.25})


def test_dynamic_specialization_freezes_compiler_timeout(monkeypatch):
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    value = builder.input((batch,), dtype="int32")
    module = builder.finish(value.relu())
    captured = []
    sentinel = object()

    def fake_compile_module(module, **kwargs):
        captured.append(kwargs)
        return sentinel

    monkeypatch.setattr(compiler_module, "compile_module", fake_compile_module)
    executable = compiler_module.compile_dynamic_module(module, compiler_timeout=2.5)

    assert executable.specialize(3) is sentinel
    assert captured == [
        {
            "compiler": None,
            "cache_dir": None,
            "borrow_inputs": False,
            "parallel": False,
            "compiler_timeout": 2.5,
        }
    ]


def test_adaptive_native_timeout_propagates_instead_of_becoming_loop_fallback(monkeypatch):
    builder = GraphBuilder()
    module = builder.finish(builder.tensor([1, 2], dtype="int32").relu())
    timeout = NativeCompilationTimeout(("fake-cc",), 0.25)

    def failing_compile_native(*args, **kwargs):
        raise timeout

    monkeypatch.setattr(compiler_module, "compile_native", failing_compile_native)

    with pytest.raises(NativeCompilationTimeout) as caught:
        compiler_module.compile_adaptive_module(
            module,
            budget=CompileBudget(max_planned_storage_bytes=1_000_000),
            compiler_timeout=0.25,
        )

    assert caught.value is timeout


def test_adaptive_budget_fallback_does_not_start_compiler_even_with_timeout(monkeypatch):
    builder = GraphBuilder()
    module = builder.finish(builder.tensor([1, 2], dtype="int32").relu())

    def unexpected_compile_native(*args, **kwargs):
        raise AssertionError("budget fallback must not launch native compilation")

    monkeypatch.setattr(compiler_module, "compile_native", unexpected_compile_native)
    executable = compiler_module.compile_adaptive_module(
        module,
        budget=CompileBudget(max_planned_storage_bytes=0),
        compiler_timeout=0.25,
    )

    assert executable.backend == "loop"
