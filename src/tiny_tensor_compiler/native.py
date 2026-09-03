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

        compile_command = [
            *command,
            "-std=c11",
            "-O2",
            "-fwrapv",
            *_shared_library_flags(),
            str(source_path),
            "-o",
            str(library_path),
        ]
        completed = subprocess.run(
            compile_command,
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

        scalar_type = np.ctypeslib.as_ctypes_type(np.dtype(return_type.dtype.to_numpy()))
        pointer_type = ctypes.POINTER(scalar_type)
        runner = library.tiny_tensor_run
        runner.argtypes = [pointer_type]
        runner.restype = None
        runner(output.ctypes.data_as(pointer_type))

    return output


def _compiler_command(compiler: str | None) -> list[str]:
    configured = compiler if compiler is not None else os.environ.get("CC", "cc")
    command = shlex.split(configured)
    if not command:
        raise NativeCompilationError("C compiler command is empty")
    if shutil.which(command[0]) is None:
        raise NativeCompilationError(f"C compiler executable not found: {command[0]}")
    return command


def _return_type(program: LoopProgram):
    for allocation in program.allocations:
        if allocation.buffer == program.return_slot:
            return allocation.type
    raise RuntimeError("verified loop IR return buffer unexpectedly has no allocation")


def _library_name() -> str:
    if sys.platform == "darwin":
        return "program.dylib"
    if os.name == "posix":
        return "program.so"
    raise NativeCompilationError("native C execution currently requires a POSIX-like platform")


def _shared_library_flags() -> list[str]:
    if sys.platform == "darwin":
        return ["-dynamiclib", "-fPIC"]
    if os.name == "posix":
        return ["-shared", "-fPIC"]
    raise NativeCompilationError("native C execution currently requires a POSIX-like platform")
