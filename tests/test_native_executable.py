import os
import shutil

import numpy as np
import pytest

import tiny_tensor_compiler.native as native_module
from tiny_tensor_compiler import GraphBuilder, compile_native, lower_to_cpu, lower_to_loops


@pytest.fixture(autouse=True)
def _clear_native_artifact_cache():
    native_module.clear_native_cache()
    yield
    native_module.clear_native_cache()


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _input_program():
    builder = GraphBuilder()
    value = builder.input((3,), dtype="float32")
    module = builder.finish((value * 2 + 1).relu())
    return lower_to_loops(lower_to_cpu(module))


def test_compile_native_eagerly_compiles_and_reuses_for_multiple_calls(monkeypatch):
    _default_compiler_or_skip()
    loops = _input_program()
    compile_calls = 0
    original_run = native_module.subprocess.run

    def counting_run(*args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return original_run(*args, **kwargs)

    monkeypatch.setattr(native_module.subprocess, "run", counting_run)

    executable = compile_native(loops)
    assert compile_calls == 1

    first_input = np.array([-2.0, 0.0, 3.0], dtype=np.float32)
    second_input = np.array([-1.0, 4.0, 5.0], dtype=np.float32)
    first = executable(inputs=[first_input])
    second = executable([second_input])

    assert compile_calls == 1
    np.testing.assert_array_equal(
        first,
        np.maximum(first_input * np.float32(2.0) + np.float32(1.0), np.float32(0.0)),
    )
    np.testing.assert_array_equal(
        second,
        np.maximum(second_input * np.float32(2.0) + np.float32(1.0), np.float32(0.0)),
    )


def test_native_executable_survives_cache_clear_with_frozen_compiler(monkeypatch):
    _default_compiler_or_skip()
    loops = _input_program()
    compile_calls = 0
    original_run = native_module.subprocess.run

    def counting_run(*args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return original_run(*args, **kwargs)

    monkeypatch.setattr(native_module.subprocess, "run", counting_run)

    executable = compile_native(loops)
    assert compile_calls == 1

    native_module.clear_native_cache()
    monkeypatch.setenv("CC", "tiny-tensor-compiler-missing-cc")

    runtime_input = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = executable(inputs=[runtime_input])

    assert compile_calls == 2
    np.testing.assert_array_equal(result, np.array([3.0, 5.0, 7.0], dtype=np.float32))


def test_native_executable_preserves_runtime_input_validation(monkeypatch):
    _default_compiler_or_skip()
    loops = _input_program()
    compile_calls = 0
    original_run = native_module.subprocess.run

    def counting_run(*args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return original_run(*args, **kwargs)

    monkeypatch.setattr(native_module.subprocess, "run", counting_run)

    executable = compile_native(loops)

    with pytest.raises(ValueError, match="input 0 shape"):
        executable(inputs=[np.array([1.0, 2.0], dtype=np.float32)])
    with pytest.raises(ValueError, match="input 0 dtype"):
        executable(inputs=[np.array([1, 2, 3], dtype=np.int32)])

    assert compile_calls == 1
