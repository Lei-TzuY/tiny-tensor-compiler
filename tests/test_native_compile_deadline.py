import contextlib
import math
import multiprocessing
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


def _python_compiler_command() -> str:
    return shlex.join([sys.executable])


def _sleeping_compiler(monkeypatch):
    command = [sys.executable, "-c", "import time; time.sleep(10)"]
    monkeypatch.setattr(native_module, "_build_compile_command", lambda *args: command)
    return tuple(command)


def _wait_for(path: Path, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.02)


def _hold_lease(library: str, ready: str, release: str) -> None:
    library_path = Path(library)
    ready_path = Path(ready)
    release_path = Path(release)
    with native_module._persistent_cache_lease(library_path):
        ready_path.write_text("ready", encoding="utf-8")
        _wait_for(release_path)


@pytest.mark.parametrize("bad", [True, "1", object()])
def test_compile_deadline_rejects_non_numeric_values_before_compiler_lookup(bad):
    with pytest.raises(TypeError):
        compile_native(
            _loop_program(),
            compiler="definitely-not-a-compiler",
            compile_deadline=bad,
        )


@pytest.mark.parametrize("bad", [0, -1, math.nan, math.inf, -math.inf])
def test_compile_deadline_requires_positive_finite_seconds(bad):
    with pytest.raises(ValueError):
        compile_native(
            _loop_program(),
            compiler="definitely-not-a-compiler",
            compile_deadline=bad,
        )


def test_compile_deadline_bounds_real_external_compiler(monkeypatch):
    expected_command = _sleeping_compiler(monkeypatch)
    started = time.monotonic()

    with pytest.raises(NativeCompilationDeadlineExceeded) as caught:
        compile_native(
            _loop_program(),
            compiler=_python_compiler_command(),
            compile_deadline=0.05,
        )

    assert caught.value.stage == "compiler"
    assert caught.value.deadline == 0.05
    assert caught.value.command == expected_command
    assert time.monotonic() - started < 3.0


def test_compiler_timeout_wins_when_it_is_tighter_than_total_deadline(monkeypatch):
    expected_command = _sleeping_compiler(monkeypatch)

    with pytest.raises(NativeCompilationTimeout) as caught:
        compile_native(
            _loop_program(),
            compiler=_python_compiler_command(),
            compiler_timeout=0.05,
            compile_deadline=1.0,
        )

    assert caught.value.timeout == 0.05
    assert caught.value.command == expected_command


def test_total_deadline_wins_when_remaining_budget_is_tighter(monkeypatch):
    expected_command = _sleeping_compiler(monkeypatch)

    with pytest.raises(NativeCompilationDeadlineExceeded) as caught:
        compile_native(
            _loop_program(),
            compiler=_python_compiler_command(),
            compiler_timeout=1.0,
            compile_deadline=0.05,
        )

    assert caught.value.stage == "compiler"
    assert caught.value.command == expected_command


def test_total_deadline_bounds_persistent_cache_lease_before_compiler_launch(
    tmp_path,
    monkeypatch,
):
    program = _loop_program()
    cache_dir = tmp_path / "native-cache"
    compiler = _python_compiler_command()
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

        def must_not_compile(*args, **kwargs):
            raise AssertionError("compiler must not launch after lease deadline exhaustion")

        monkeypatch.setattr(native_module, "_compile_source", must_not_compile)
        started = time.monotonic()
        with pytest.raises(NativeCompilationDeadlineExceeded) as caught:
            compile_native(
                program,
                compiler=compiler,
                cache_dir=cache_dir,
                compile_deadline=0.1,
            )
        assert caught.value.stage == "persistent-cache lease"
        assert time.monotonic() - started < 3.0
    finally:
        release.write_text("release", encoding="utf-8")
        holder.join(timeout=20)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)


def test_persistent_cache_hit_does_not_launch_compiler_under_total_deadline(
    tmp_path,
    monkeypatch,
):
    compiler = "cl" if sys.platform == "win32" else "cc"
    if shutil.which(compiler) is None:
        pytest.skip(f"no platform default C compiler available: {compiler}")

    program = _loop_program()
    cache_dir = tmp_path / "native-cache"
    compile_native(program, compiler=compiler, cache_dir=cache_dir)
    clear_native_cache()

    @contextlib.contextmanager
    def no_compiler(*args, **kwargs):
        raise AssertionError("persistent cache hit must not launch the compiler")
        yield

    monkeypatch.setattr(native_module, "_compile_source", no_compiler)
    compile_native(
        program,
        compiler=compiler,
        cache_dir=cache_dir,
        compile_deadline=1.0,
    )


def test_compile_module_forwards_explicit_deadline_without_changing_default_call_shape(monkeypatch):
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
    assert compiler_module.compile_module(module, compile_deadline=1.25) is sentinel
    assert calls[-1] == (None, None, {"compile_deadline": 1.25})


def test_dynamic_specialization_freezes_compile_deadline(monkeypatch):
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
    executable = compiler_module.compile_dynamic_module(module, compile_deadline=2.5)

    assert executable.specialize(3) is sentinel
    assert captured == [
        {
            "compiler": None,
            "cache_dir": None,
            "borrow_inputs": False,
            "parallel": False,
            "compile_deadline": 2.5,
        }
    ]


def test_adaptive_budget_fallback_does_not_consume_native_deadline(monkeypatch):
    builder = GraphBuilder()
    module = builder.finish(builder.tensor([1, 2], dtype="int32").relu())

    def unexpected_compile_native(*args, **kwargs):
        raise AssertionError("budget fallback must not enter native compilation")

    monkeypatch.setattr(compiler_module, "compile_native", unexpected_compile_native)
    executable = compiler_module.compile_adaptive_module(
        module,
        budget=CompileBudget(max_planned_storage_bytes=0),
        compile_deadline=0.25,
    )

    assert executable.backend == "loop"
