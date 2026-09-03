from __future__ import annotations

import atexit
import ctypes
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .c_codegen import generate_c
from .input_validation import prepare_runtime_inputs
from .loop_ir import LoopProgram


class NativeCompilationError(RuntimeError):
    """Raised when generated C cannot be compiled or loaded for native execution."""


class _NativeArtifact:
    def __init__(self, directory: Path, library: ctypes.CDLL) -> None:
        self.directory = directory
        self.library = library
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        _release_library(self.library)
        shutil.rmtree(self.directory)
        self.closed = True


_NATIVE_CACHE: dict[tuple[tuple[str, ...], str], _NativeArtifact] = {}
_NATIVE_CACHE_LOCK = threading.RLock()


def execute_native(
    program: LoopProgram,
    compiler: str | None = None,
    inputs: Sequence[Any] = (),
) -> np.ndarray:
    """Compile or reuse generated C and execute it on the native CPU."""
    runtime_inputs = prepare_runtime_inputs(program.input_types, inputs)
    command = _compiler_command(compiler)
    source = generate_c(program)
    return_type = _return_type(program)
    output = np.empty(return_type.shape, dtype=return_type.dtype.to_numpy())

    with _NATIVE_CACHE_LOCK:
        artifact = _get_or_compile_artifact(source, command)
        output_pointer_type = _pointer_type(return_type.dtype.to_numpy())
        input_pointer_types = tuple(
            _pointer_type(input_type.dtype.to_numpy()) for input_type in program.input_types
        )
        runner = artifact.library.tiny_tensor_run
        runner.argtypes = [output_pointer_type, *input_pointer_types]
        runner.restype = None
        arguments = [output.ctypes.data_as(output_pointer_type)]
        arguments.extend(
            array.ctypes.data_as(pointer_type)
            for array, pointer_type in zip(
                runtime_inputs,
                input_pointer_types,
                strict=True,
            )
        )
        runner(*arguments)

    return output


def clear_native_cache() -> None:
    """Release all cached native libraries and remove their build directories."""
    with _NATIVE_CACHE_LOCK:
        artifacts = list(_NATIVE_CACHE.values())
        _NATIVE_CACHE.clear()
        first_error: NativeCompilationError | OSError | None = None
        for artifact in artifacts:
            try:
                artifact.close()
            except (NativeCompilationError, OSError) as error:  # pragma: no cover
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise NativeCompilationError(f"failed to clear native artifact cache: {first_error}") from first_error


def _get_or_compile_artifact(source: str, command: list[str]) -> _NativeArtifact:
    key = (tuple(command), source)
    artifact = _NATIVE_CACHE.get(key)
    if artifact is None:
        artifact = _compile_artifact(source, command)
        _NATIVE_CACHE[key] = artifact
    return artifact


def _compile_artifact(source: str, command: list[str]) -> _NativeArtifact:
    directory_path = Path(tempfile.mkdtemp(prefix="tiny_tensor_compiler_"))
    source_path = directory_path / "program.c"
    library_path = directory_path / _library_name()

    try:
        source_path.write_text(source, encoding="utf-8")
        compile_command = _build_compile_command(command, source_path.name, library_path.name)
        completed = subprocess.run(
            compile_command,
            cwd=directory_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip() or "no compiler output"
            raise NativeCompilationError(
                f"native C compilation failed with exit code {completed.returncode}: {details}"
            )

        try:
            library = ctypes.CDLL(str(library_path))
        except OSError as error:
            raise NativeCompilationError(f"failed to load native shared library: {error}") from error
    except Exception:
        shutil.rmtree(directory_path, ignore_errors=True)
        raise

    return _NativeArtifact(directory_path, library)


def _compiler_command(compiler: str | None) -> list[str]:
    default = "cl" if os.name == "nt" else "cc"
    configured = compiler if compiler is not None else os.environ.get("CC", default)
    command = shlex.split(configured)
    if not command:
        raise NativeCompilationError("C compiler command is empty")
    if shutil.which(command[0]) is None:
        raise NativeCompilationError(f"C compiler executable not found: {command[0]}")
    return command


def _build_compile_command(command: list[str], source_name: str, library_name: str) -> list[str]:
    if _is_msvc(command):
        return [
            *command,
            "/nologo",
            "/std:c11",
            "/O2",
            "/LD",
            source_name,
            f"/Fe:{library_name}",
        ]
    return [
        *command,
        "-std=c11",
        "-O2",
        "-fwrapv",
        *_shared_library_flags(),
        source_name,
        "-o",
        library_name,
    ]


def _is_msvc(command: list[str]) -> bool:
    executable = Path(command[0]).name.casefold()
    return executable in {"cl", "cl.exe", "clang-cl", "clang-cl.exe"}


def _pointer_type(dtype: np.dtype[Any]):
    scalar_type = np.ctypeslib.as_ctypes_type(np.dtype(dtype))
    return ctypes.POINTER(scalar_type)


def _release_library(library: ctypes.CDLL) -> None:
    if os.name != "nt":
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    free_library = kernel32.FreeLibrary
    free_library.argtypes = [ctypes.c_void_p]
    free_library.restype = ctypes.c_int
    if free_library(ctypes.c_void_p(library._handle)) == 0:
        error = ctypes.get_last_error()
        raise NativeCompilationError(f"failed to unload native shared library: Windows error {error}")


def _return_type(program: LoopProgram):
    for allocation in program.allocations:
        if allocation.buffer == program.return_slot:
            return allocation.type
    raise RuntimeError("verified loop IR return buffer unexpectedly has no allocation")


def _library_name() -> str:
    if os.name == "nt":
        return "program.dll"
    if sys.platform == "darwin":
        return "program.dylib"
    if os.name == "posix":
        return "program.so"
    raise NativeCompilationError(f"unsupported native platform: {os.name}")


def _shared_library_flags() -> list[str]:
    if os.name == "nt":
        return ["-shared"]
    if sys.platform == "darwin":
        return ["-dynamiclib", "-fPIC"]
    if os.name == "posix":
        return ["-shared", "-fPIC"]
    raise NativeCompilationError(f"unsupported native platform: {os.name}")


atexit.register(clear_native_cache)
