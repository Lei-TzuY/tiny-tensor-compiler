import hashlib
import json
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


def _constant_relu_loops(values=(-3.0, -0.0, 2.0)):
    builder = GraphBuilder()
    module = builder.finish(builder.tensor(values, dtype="float32").relu())
    return lower_to_loops(lower_to_cpu(module))


def _cached_library(cache_root):
    libraries = list(cache_root.rglob(native_module._library_name()))
    assert len(libraries) == 1
    return libraries[0]


def _manifest_for(library):
    return library.with_name(native_module._PERSISTENT_MANIFEST_NAME)


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
    cached_library = _cached_library(tmp_path)

    native_module.clear_native_cache()
    assert cached_library.exists()

    second = execute_native(loops, cache_dir=tmp_path)

    assert compile_calls == 1
    np.testing.assert_array_equal(first, second)
    assert not np.signbit(second[1])


def test_persistent_manifest_records_exact_library_digest(tmp_path):
    _default_compiler_or_skip()
    execute_native(_constant_relu_loops(), cache_dir=tmp_path)

    cached_library = _cached_library(tmp_path)
    manifest = json.loads(_manifest_for(cached_library).read_text(encoding="utf-8"))

    assert manifest == {
        "schema": native_module._PERSISTENT_CACHE_SCHEMA,
        "digest": cached_library.parent.name,
        "library": cached_library.name,
        "library_sha256": hashlib.sha256(cached_library.read_bytes()).hexdigest(),
    }


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

    cached_library = _cached_library(tmp_path)
    corrupt_library = cached_library.with_name(f"corrupt-{cached_library.name}")
    corrupt_library.write_bytes(b"not a shared library")
    os.replace(corrupt_library, cached_library)

    rebuilt = execute_native(loops, cache_dir=tmp_path)

    assert compile_calls == 2
    np.testing.assert_array_equal(rebuilt, expected)


@pytest.mark.parametrize("manifest_state", ["missing", "malformed"])
def test_missing_or_malformed_persistent_manifest_is_rebuilt(
    tmp_path, monkeypatch, manifest_state
):
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
    cached_library = _cached_library(tmp_path)
    manifest_path = _manifest_for(cached_library)
    if manifest_state == "missing":
        manifest_path.unlink()
    else:
        manifest_path.write_text("{not-json", encoding="utf-8")

    rebuilt = execute_native(loops, cache_dir=tmp_path)

    assert compile_calls == 2
    np.testing.assert_array_equal(rebuilt, expected)
    assert manifest_path.is_file()


def test_loadable_wrong_persistent_library_is_rebuilt_before_execution(tmp_path, monkeypatch):
    _default_compiler_or_skip()
    expected_loops = _constant_relu_loops((-5.0, 1.0, 7.0))
    other_loops = _constant_relu_loops((11.0, 12.0, 13.0))
    compile_calls = 0
    original_run = native_module.subprocess.run

    def counting_run(*args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return original_run(*args, **kwargs)

    monkeypatch.setattr(native_module.subprocess, "run", counting_run)

    expected = execute_native(expected_loops, cache_dir=tmp_path)
    expected_library = _cached_library(tmp_path)

    other_cache = tmp_path / "other"
    execute_native(other_loops, cache_dir=other_cache)
    other_library = _cached_library(other_cache)
    native_module.clear_native_cache()

    shutil.copy2(other_library, expected_library)
    assert hashlib.sha256(expected_library.read_bytes()).hexdigest() != json.loads(
        _manifest_for(expected_library).read_text(encoding="utf-8")
    )["library_sha256"]

    rebuilt = execute_native(expected_loops, cache_dir=tmp_path)

    assert compile_calls == 3
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

    assert not list(tmp_path.rglob(native_module._PERSISTENT_MANIFEST_NAME))
    recovered = execute_native(loops, cache_dir=tmp_path)
    native_module.clear_native_cache()
    reused = execute_native(loops, cache_dir=tmp_path)

    assert compile_calls == 2
    np.testing.assert_array_equal(recovered, reused)
