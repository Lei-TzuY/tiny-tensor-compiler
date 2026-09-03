from __future__ import annotations

import ctypes
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .c_codegen import generate_c
from .loop_ir import LoopProgram


class NativeCompilationError(RuntimeError):
    """Raised when generated C cannot be compiled or loaded for native execution."""


def execute_native(program: LoopProgram, compiler: str | None = None) -> np.ndarray:
    """Compile generated C into a temporary shared library and execute it on the CPU."""
    command = _compiler_command(compiler)
    return_type = _return_type(program)
    output = np.empty(return_type.shape, dtype=return_type.dtype.to_numpy())

    with tempfile.TemporaryDirectory(prefix="tiny_tensor_compiler_") as directory:
        directory_path = Path(directory)
        source_path = directory_path / "program.c"
        library_path = directory_path / _library_name()
        source_path.write_text(generate_c(program), encoding="utf-8")

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

        try:
            scalar_type = np.ctypeslib.as_ctypes_type(np.dtype(return_type.dtype.to_numpy()))
            pointer_type = ctypes.POINTER(scalar_type)
            runner = library.tiny_tensor_run
            runner.argtypes = [pointer_type]
            runner.restype = None
            runner(output.ctypes.data_as(pointer_type))
        finally:
            _release_library(library)

    return output


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
