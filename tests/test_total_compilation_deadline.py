import math
import multiprocessing
import os
import shlex
import shutil
import sys
import time
from pathlib import Path

import pytest

import tiny_tensor_compiler.compiler as compiler_module
import tiny_tensor_compiler.native as native_module
from tiny_tensor_compiler import (
    CompileBudget,
    GraphBuilder,
    NativeCompilationDeadlineExceeded,
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


def _platform_compiler() -> str:
    return "cl" if os.name == "nt" else "cc"


def _python_compiler_command() -> str:
    return shlex.join([sys.executable])


def _sleeping_compiler(monkeypatch, seconds: float = 10.0):
    command = [sys.executable, "-c", f"import time; time.sleep({seconds!r})"]
    monkeypatch.setattr(native_module, "_build_compile_command", lambda *args: command)
    return tuple(command)


def _hold_lease(library: str, ready: str, release: str) -> None:
    library_path = Path(library)
    ready_path = Path(ready)
    release_path = Path(release)
    with native_module._persistent_cache_lease(library_path):
        ready_path.write_text("ready", encoding="utf-8")
        while not release_path.exists():
            time.sleep(0.02)


def _wait_for(path: Path, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.02)


@pytest.mark.parametrize("bad", [True, "1", object()])
def test_compilation_timeout_rejects_non_numeric_values_before_compiler_lookup(bad):
    with pytest.raises(TypeError):
        compile_native(
            _loop_program(),
            compiler="definitely-not-a-compiler",
            compilation_timeout=bad,
        )


@pytest.mark.parametrize("bad", [0, -1, math.nan, math.inf, -math.inf])
def test_compilation_timeout_requires_positive_finite_seconds(bad):
    with pytest.raises(ValueError):
        compile_native(
            _loop_program(),
            compiler="definitely-not-a-compiler",
            compilation_timeout=bad,
        )


@pytest.mark.parametrize("parallel", [False, True])
def test_total_deadline_bounds_compiler_and_cleans_transient_build(monkeypatch, parallel):
    expected_command = _sleeping_compiler(monkeypatch)
    created: list[Path] = []
    real_mkdtemp = native_module.tempfile.mkdtemp

    def tracking_mkdtemp(*args, **kwargs):
        directory = Path(real_mkdtemp(*args, **kwargs))
        created.append(directory)
        return str(directory)

    monkeypatch.setattr(native_module.tempfile, "mkdtemp", tracking_mkdtemp)

    started = time.monotonic()
    with pytest.raises(NativeCompilationDeadlineExceeded) as caught:
        compile_native(
            _loop_program(),
            compiler=_python_compiler_command(),
            compilation_timeout=0.08,
            parallel=parallel,
        )
    elapsed = time.monotonic() - started

    assert caught.value.timeout == 0.08
    assert caught.value.phase == "compiler process"
    assert caught.value.command == expected_command
    assert elapsed < 3.0
    assert created
    assert all(not directory.exists() for directory in created)


def test_total_deadline_kills_compiler_process_tree(monkeypatch, tmp_path):
    marker = tmp_path / "descendant-survived"
    child_code = (
        "import pathlib,time; "
        "time.sleep(0.6); "
        f"pathlib.Path({str(marker)!r}).write_text('alive', encoding='utf-8')"
    )
    parent_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "time.sleep(10)"
    )
    command = [sys.executable, "-c", parent_code]
    monkeypatch.setattr(native_module, "_build_compile_command", lambda *args: command)

    with pytest.raises(NativeCompilationDeadlineExceeded):
        compile_native(
            _loop_program(),
            compiler=_python_compiler_command(),
            compilation_timeout=0.1,
        )

    time.sleep(1.0)
    assert not marker.exists()


def test_total_deadline_includes_persistent_cache_lease_wait(tmp_path):
    compiler = _platform_compiler()
    if shutil.which(compiler) is None:
        pytest.skip(f"no platform default C compiler available: {compiler}")

    program = _loop_program()
    cache_dir = tmp_path / "native-cache"
    command = native_module._compiler_command(compiler)
    source = native_module.generate_c(program)
    library = native_module._persistent_library_path(cache_dir, source, command)
    assert library is not None

    ready = tmp_path / "ready"
    release = tmp_path / "release"
    context = multiprocessing.get_context("spawn")
    holder = context.Process(
        target=_hold_lease,
        args=(str(library), str(ready), str(release)),
    )
    holder.start()
    try:
        _wait_for(ready)
        started = time.monotonic()
        with pytest.raises(NativeCompilationDeadlineExceeded) as caught:
            compile_native(
                program,
                compiler=compiler,
                cache_dir=cache_dir,
                compilation_timeout=0.1,
            )
        elapsed = time.monotonic() - started
        assert caught.value.phase == "persistent-cache lease"
        assert elapsed < 3.0
    finally:
        release.write_text("release", encoding="utf-8")
        holder.join(timeout=20)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)
    assert holder.exitcode == 0


def test_shorter_compiler_timeout_remains_distinct_from_total_deadline(monkeypatch):
    _sleeping_compiler(monkeypatch)
    with pytest.raises(NativeCompilationTimeout):
        compile_native(
            _loop_program(),
            compiler=_python_compiler_command(),
            compiler_timeout=0.05,
            compilation_timeout=1.0,
        )


def test_shorter_total_deadline_wins_over_compiler_timeout(monkeypatch):
    _sleeping_compiler(monkeypatch)
    with pytest.raises(NativeCompilationDeadlineExceeded) as caught:
        compile_native(
            _loop_program(),
            compiler=_python_compiler_command(),
            compiler_timeout=1.0,
            compilation_timeout=0.05,
        )
    assert caught.value.phase == "compiler process"


def test_persistent_cache_hit_does_not_compile_under_total_deadline(tmp_path, monkeypatch):
    compiler = _platform_compiler()
    if shutil.which(compiler) is None:
        pytest.skip(f"no platform default C compiler available: {compiler}")

    program = _loop_program()
    cache_dir = tmp_path / "native-cache"
    compile_native(program, compiler=compiler, cache_dir=cache_dir)
    clear_native_cache()

    def unexpected_run(*args, **kwargs):
        raise AssertionError("persistent cache hit must not launch the compiler")

    def unexpected_popen(*args, **kwargs):
        raise AssertionError("persistent cache hit must not launch the compiler")

    monkeypatch.setattr(native_module.subprocess, "run", unexpected_run)
    monkeypatch.setattr(native_module.subprocess, "Popen", unexpected_popen)
    compile_native(
        program,
        compiler=compiler,
        cache_dir=cache_dir,
        compilation_timeout=2.0,
    )


def test_compile_module_forwards_total_deadline_without_changing_default_call_shape(monkeypatch):
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
    assert compiler_module.compile_module(module, compilation_timeout=1.25) is sentinel
    forwarded = calls[-1]
    assert forwarded[0:2] == (None, None)
    assert set(forwarded[2]) == {"compilation_timeout"}
    assert 0.0 < forwarded[2]["compilation_timeout"] <= 1.25


def test_dynamic_specialization_freezes_total_deadline(monkeypatch):
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
    executable = compiler_module.compile_dynamic_module(module, compilation_timeout=2.5)

    assert executable.specialize(3) is sentinel
    assert len(captured) == 1
    assert captured[0]["compiler"] is None
    assert captured[0]["cache_dir"] is None
    assert captured[0]["borrow_inputs"] is False
    assert captured[0]["parallel"] is False
    assert 0.0 < captured[0]["compilation_timeout"] <= 2.5


def test_adaptive_total_deadline_propagates_instead_of_becoming_loop_fallback(monkeypatch):
    builder = GraphBuilder()
    module = builder.finish(builder.tensor([1, 2], dtype="int32").relu())
    deadline = NativeCompilationDeadlineExceeded(0.25, "compiler process", ("fake-cc",))

    def failing_compile_native(*args, **kwargs):
        raise deadline

    monkeypatch.setattr(compiler_module, "compile_native", failing_compile_native)

    with pytest.raises(NativeCompilationDeadlineExceeded) as caught:
        compiler_module.compile_adaptive_module(
            module,
            budget=CompileBudget(max_planned_storage_bytes=1_000_000),
            compilation_timeout=0.25,
        )

    assert caught.value is deadline


def test_adaptive_budget_fallback_still_avoids_native_compiler_under_total_deadline(monkeypatch):
    builder = GraphBuilder()
    module = builder.finish(builder.tensor([1, 2], dtype="int32").relu())

    def unexpected_compile_native(*args, **kwargs):
        raise AssertionError("budget fallback must not launch native compilation")

    monkeypatch.setattr(compiler_module, "compile_native", unexpected_compile_native)
    executable = compiler_module.compile_adaptive_module(
        module,
        budget=CompileBudget(max_planned_storage_bytes=0),
        compilation_timeout=1.0,
    )

    assert executable.backend == "loop"
