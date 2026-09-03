import os
import shutil

import numpy as np
import pytest

import tiny_tensor_compiler.compiler as compiler_module
from tiny_tensor_compiler import (
    GraphBuilder,
    VerificationError,
    clear_native_cache,
    compile_module,
    execute_reference,
)


@pytest.fixture(autouse=True)
def _clear_native_artifact_cache():
    clear_native_cache()
    yield
    clear_native_cache()


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def test_compile_module_lowers_fuses_without_mutation_or_hidden_optimization(monkeypatch):
    builder = GraphBuilder()
    x = builder.input((2,), dtype="int32")
    _dead = (x + 1).relu()
    module = builder.finish(x * 2)
    before_dump = module.dump()
    captured = {}
    sentinel = object()

    def fake_compile_native(program, compiler=None, cache_dir=None):
        captured["program"] = program
        captured["compiler"] = compiler
        captured["cache_dir"] = cache_dir
        return sentinel

    monkeypatch.setattr(compiler_module, "compile_native", fake_compile_native)

    executable = compile_module(module, compiler="fake-cc --flag", cache_dir="native-cache")

    assert executable is sentinel
    assert module.dump() == before_dump
    assert [kernel.opcode for kernel in captured["program"].kernels] == [
        "const",
        "relu_add",
        "const",
        "mul",
    ]
    assert captured["compiler"] == "fake-cc --flag"
    assert captured["cache_dir"] == "native-cache"


def test_compile_module_verifies_tensor_ir_before_native_compilation(monkeypatch):
    builder = GraphBuilder()
    x = builder.input((1,), dtype="float32")
    module = builder.finish(x)
    module.function.add_op("return", operands=(module.function.ops[-1].operands[0],))

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("native compilation must not run for malformed tensor IR")

    monkeypatch.setattr(compiler_module, "compile_native", unexpected_compile)

    with pytest.raises(VerificationError):
        compile_module(module)


def test_compile_module_native_execution_matches_reference_for_repeated_inputs():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    x = builder.input((3,), dtype="float32")
    module = builder.finish((x * 2 + 1).relu())
    executable = compile_module(module)

    for values in (
        np.array([-2.0, 0.0, 3.0], dtype=np.float32),
        np.array([4.0, -1.0, 0.5], dtype=np.float32),
    ):
        actual = executable(inputs=[values])
        expected = execute_reference(module, inputs=[values])
        np.testing.assert_array_equal(actual, expected)
