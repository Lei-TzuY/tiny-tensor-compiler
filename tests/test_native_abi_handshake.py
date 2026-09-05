from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

import tiny_tensor_compiler.native as native_module
from tiny_tensor_compiler import GraphBuilder, NativeCompilationError, lower_to_cpu, lower_to_loops
from tiny_tensor_compiler.c_abi_codegen import generate_c
from tiny_tensor_compiler.native_abi import native_abi_sha256


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _loops(*, shape: tuple[int, ...] = (3,), dtype: str = "int32"):
    builder = GraphBuilder()
    value = builder.input(shape, dtype=dtype)
    return lower_to_loops(lower_to_cpu(builder.finish(value.relu())))


def _wrong_abi_source(fingerprint: str) -> str:
    return f"""
#include <stdint.h>
#if defined(_WIN32)
#define TINY_TENSOR_EXPORT __declspec(dllexport)
#else
#define TINY_TENSOR_EXPORT
#endif
TINY_TENSOR_EXPORT const char *tiny_tensor_abi_sha256(void) {{
    return \"{fingerprint}\";
}}
TINY_TENSOR_EXPORT void tiny_tensor_run(int32_t *out, const int32_t *input0) {{
    out[0] = input0[0];
}}
""".lstrip()


def test_native_abi_fingerprint_is_deterministic_and_type_sensitive():
    baseline = _loops()
    same = _loops()
    different_shape = _loops(shape=(4,))
    different_dtype = _loops(dtype="float32")

    assert native_abi_sha256(baseline) == native_abi_sha256(same)
    assert native_abi_sha256(baseline) != native_abi_sha256(different_shape)
    assert native_abi_sha256(baseline) != native_abi_sha256(different_dtype)
    assert len(native_abi_sha256(baseline)) == 64


def test_generated_c_exports_exact_native_abi_fingerprint():
    loops = _loops()
    fingerprint = native_abi_sha256(loops)
    source = generate_c(loops)

    assert "TINY_TENSOR_EXPORT const char *tiny_tensor_abi_sha256(void)" in source
    assert f'return "{fingerprint}";' in source


def test_native_loader_rejects_real_library_with_wrong_embedded_abi():
    _default_compiler_or_skip()
    expected = native_abi_sha256(_loops())
    wrong = "0" * 64 if expected != "0" * 64 else "1" * 64
    command = native_module._compiler_command(None)

    with tempfile.TemporaryDirectory(prefix="tiny_tensor_bad_abi_") as directory:
        library_path = native_module._compile_source(
            _wrong_abi_source(wrong),
            command,
            Path(directory),
        )
        library = native_module._load_library(library_path)
        try:
            with pytest.raises(NativeCompilationError, match="native ABI fingerprint mismatch"):
                native_module._verify_native_abi(library, expected)
        finally:
            native_module._release_library(library)


def test_persistent_cache_rebuilds_self_consistent_wrong_abi_artifact(tmp_path, monkeypatch):
    _default_compiler_or_skip()
    loops = _loops()
    command = native_module._compiler_command(None)
    source = generate_c(loops)
    expected = native_abi_sha256(loops)
    library_path = native_module._persistent_library_path(tmp_path, source, command)
    assert library_path is not None

    library_path.parent.mkdir(parents=True, exist_ok=True)
    wrong = "0" * 64 if expected != "0" * 64 else "1" * 64
    with tempfile.TemporaryDirectory(prefix="tiny_tensor_bad_persistent_abi_") as directory:
        compiled = native_module._compile_source(
            _wrong_abi_source(wrong),
            command,
            Path(directory),
        )
        shutil.copy2(compiled, library_path)

    manifest_path = native_module._persistent_manifest_path(library_path)
    manifest_path.write_text(
        json.dumps(
            {
                "schema": native_module._PERSISTENT_CACHE_SCHEMA,
                "digest": library_path.parent.name,
                "library": library_path.name,
                "library_sha256": native_module._sha256_file(library_path),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    compile_calls = 0
    original_compile_source = native_module._compile_source

    def counting_compile_source(source_text, compiler_command, directory_path):
        nonlocal compile_calls
        compile_calls += 1
        return original_compile_source(source_text, compiler_command, directory_path)

    monkeypatch.setattr(native_module, "_compile_source", counting_compile_source)
    executable = native_module.compile_native(loops, cache_dir=tmp_path)
    result = executable(inputs=[np.array([-2, 0, 3], dtype=np.int32)])

    assert compile_calls == 1
    np.testing.assert_array_equal(result, np.array([0, 0, 3], dtype=np.int32))


def test_reusable_native_handle_revalidates_abi_after_cache_clear(monkeypatch):
    _default_compiler_or_skip()
    loops = _loops()
    executable = native_module.compile_native(loops)
    expected = native_abi_sha256(loops)
    original_verify_native_abi = native_module._verify_native_abi
    observed: list[str] = []

    def recording_verify_native_abi(library, expected_abi_sha256):
        observed.append(expected_abi_sha256)
        return original_verify_native_abi(library, expected_abi_sha256)

    monkeypatch.setattr(native_module, "_verify_native_abi", recording_verify_native_abi)
    native_module.clear_native_cache()
    result = executable(inputs=[np.array([-2, 0, 3], dtype=np.int32)])

    assert observed == [expected]
    np.testing.assert_array_equal(result, np.array([0, 0, 3], dtype=np.int32))
