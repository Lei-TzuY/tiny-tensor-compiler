import os
import shutil

import numpy as np
import pytest

import tiny_tensor_compiler.native as native_module
from tiny_tensor_compiler import (
    GraphBuilder,
    NativeCompilationError,
    execute_native,
    lower_to_cpu,
    lower_to_loops,
)


@pytest.fixture(autouse=True)
def _clear_process_native_cache():
    native_module.clear_native_cache()
    yield
    native_module.clear_native_cache()


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _constant_relu_loops():
    builder = GraphBuilder()
    module = builder.finish(builder.tensor([-3.0, -0.0, 2.0], dtype="float32").relu())
    return lower_to_loops(lower_to_cpu(module))


def test_persistent_native_cache_reuses_artifact_after_process_cache_clear(tmp_path, monkeypatch):
    _default_compiler_or_skip()
    loops = _constant_relu_loops()
    compile_calls = 0
    original_run = native_module.subprocess.run

    def counting_run(*args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return original_run(*args, **kwargs)

    monkeypatch.setattr(native_module.subprocess, "run", counting_run)

    first = execute_native(loops, cache_dir=tmp_path)
    cached_libraries = list(tmp_path.rglob(native_module._library_name()))
    assert len(cached_libraries) == 1
    cached_library = cached_libraries[0]

    native_module.clear_native_cache()
    assert cached_library.exists()

    second = execute_native(loops, cache_dir=tmp_path)

    assert compile_calls == 1
    np.testing.assert_array_equal(first, second)
    assert not np.signbit(second[1])


def test_process_local_artifact_does_not_prevent_persistent_cache_population(tmp_path, monkeypatch):
    _default_compiler_or_skip()
    loops = _constant_relu_loops()
    compile_calls = 0
    original_run = native_module.subprocess.run

    def counting_run(*args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return original_run(*args, **kwargs)

    monkeypatch.setattr(native_module.subprocess, "run", counting_run)

    execute_native(loops)
    execute_native(loops, cache_dir=tmp_path)
    native_module.clear_native_cache()
    execute_native(loops, cache_dir=tmp_path)

    assert compile_calls == 2
    assert len(list(tmp_path.rglob(native_module._library_name()))) == 1


def test_corrupt_persistent_native_artifact_is_rebuilt(tmp_path, monkeypatch):
    _default_compiler_or_skip()
    loops = _constant_relu_loops()
    compile_calls = 0
    original_run = native_module.subprocess.run

    def counting_run(*args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return original_run(*args, **kwargs)

    monkeypatch.setattr(native_module.subprocess, "run", counting_run)

    expected = execute_native(loops, cache_dir=tmp_path)
    native_module.clear_native_cache()

    cached_library = next(tmp_path.rglob(native_module._library_name()))
    corrupt_library = cached_library.with_name(f"corrupt-{cached_library.name}")
    corrupt_library.write_bytes(b"not a shared library")
    os.replace(corrupt_library, cached_library)

    rebuilt = execute_native(loops, cache_dir=tmp_path)

    assert compile_calls == 2
    np.testing.assert_array_equal(rebuilt, expected)


def test_failed_compilation_does_not_poison_persistent_native_cache(tmp_path, monkeypatch):
    _default_compiler_or_skip()
    loops = _constant_relu_loops()
    compile_calls = 0
    original_run = native_module.subprocess.run

    def fail_once_run(*args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        if compile_calls == 1:
            return native_module.subprocess.CompletedProcess(
                args=args[0], returncode=1, stdout="", stderr="synthetic persistent failure"
            )
        return original_run(*args, **kwargs)

    monkeypatch.setattr(native_module.subprocess, "run", fail_once_run)

    with pytest.raises(NativeCompilationError, match="synthetic persistent failure"):
        execute_native(loops, cache_dir=tmp_path)

    recovered = execute_native(loops, cache_dir=tmp_path)
    native_module.clear_native_cache()
    reused = execute_native(loops, cache_dir=tmp_path)

    assert compile_calls == 2
    np.testing.assert_array_equal(recovered, reused)
