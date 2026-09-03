from __future__ import annotations

import atexit
import ctypes
import hashlib
import json
import os
import platform
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


class NativeExecutable:
    """Reusable native executable produced by :func:`compile_native`."""

    def __init__(
        self,
        program: LoopProgram,
        command: tuple[str, ...],
        source: str,
        persistent_library: Path | None,
    ) -> None:
        self._program = program
        self._command = command
        self._source = source
        self._persistent_library = persistent_library

    def execute(self, inputs: Sequence[Any] = ()) -> np.ndarray:
        """Execute with exact runtime-input validation and cached native code."""
        runtime_inputs = prepare_runtime_inputs(self._program.input_types, inputs)
        command = list(self._command)
        with _NATIVE_CACHE_LOCK:
            artifact = _get_or_compile_artifact(
                self._source,
                command,
                self._persistent_library,
            )
            return _execute_artifact(self._program, artifact, runtime_inputs)

    def __call__(self, inputs: Sequence[Any] = ()) -> np.ndarray:
        return self.execute(inputs)


_PERSISTENT_CACHE_SCHEMA = "native-v1"
_NATIVE_CACHE: dict[tuple[tuple[str, ...], str, str | None], _NativeArtifact] = {}
_NATIVE_CACHE_LOCK = threading.RLock()


def compile_native(
    program: LoopProgram,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> NativeExecutable:
    """Eagerly compile/load a loop program and return a reusable executable."""
    command = _compiler_command(compiler)
    source = generate_c(program)
    persistent_library = _persistent_library_path(cache_dir, source, command)

    with _NATIVE_CACHE_LOCK:
        _get_or_compile_artifact(source, command, persistent_library)

    return NativeExecutable(
        program=program,
        command=tuple(command),
        source=source,
        persistent_library=persistent_library,
    )


def execute_native(
    program: LoopProgram,
    compiler: str | None = None,
    inputs: Sequence[Any] = (),
    cache_dir: str | os.PathLike[str] | None = None,
) -> np.ndarray:
    """Compile or reuse generated C and execute it on the native CPU."""
    runtime_inputs = prepare_runtime_inputs(program.input_types, inputs)
    command = _compiler_command(compiler)
    source = generate_c(program)
    persistent_library = _persistent_library_path(cache_dir, source, command)

    with _NATIVE_CACHE_LOCK:
        artifact = _get_or_compile_artifact(source, command, persistent_library)
        return _execute_artifact(program, artifact, runtime_inputs)


def clear_native_cache() -> None:
    """Release process-cached libraries and remove their process-owned directories."""
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


def _execute_artifact(
    program: LoopProgram,
    artifact: _NativeArtifact,
    runtime_inputs: Sequence[np.ndarray[Any, Any]],
) -> np.ndarray:
    return_type = _return_type(program)
    output = np.empty(return_type.shape, dtype=return_type.dtype.to_numpy())
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


def _get_or_compile_artifact(
    source: str,
    command: list[str],
    persistent_library: Path | None,
) -> _NativeArtifact:
    persistent_identity = str(persistent_library) if persistent_library is not None else None
    key = (tuple(command), source, persistent_identity)
    artifact = _NATIVE_CACHE.get(key)
    if artifact is not None:
        return artifact

    if persistent_library is None:
        artifact = _compile_artifact(source, command)
    else:
        artifact = _get_or_compile_persistent_artifact(source, command, persistent_library)
    _NATIVE_CACHE[key] = artifact
    return artifact


def _compile_artifact(source: str, command: list[str]) -> _NativeArtifact:
    directory_path = Path(tempfile.mkdtemp(prefix="tiny_tensor_compiler_"))
    try:
        library_path = _compile_source(source, command, directory_path)
        library = _load_library(library_path)
    except Exception:
        shutil.rmtree(directory_path, ignore_errors=True)
        raise

    return _NativeArtifact(directory_path, library)


def _get_or_compile_persistent_artifact(
    source: str,
    command: list[str],
    library_path: Path,
) -> _NativeArtifact:
    cached = _stage_existing_persistent_artifact(library_path)
    if cached is not None:
        return cached

    schema_root = library_path.parent.parent
    build_directory = Path(tempfile.mkdtemp(prefix=".build-", dir=schema_root))
    try:
        compiled_library = _compile_source(source, command, build_directory)
        library_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(compiled_library, library_path)
        except OSError as error:
            concurrent = _stage_existing_persistent_artifact(library_path)
            if concurrent is not None:
                return concurrent
            raise NativeCompilationError(
                f"failed to publish persistent native artifact: {error}"
            ) from error

        staged = _stage_existing_persistent_artifact(library_path)
        if staged is None:
            raise NativeCompilationError("newly compiled persistent native artifact could not be loaded")
        return staged
    finally:
        shutil.rmtree(build_directory, ignore_errors=True)


def _compile_source(source: str, command: list[str], directory_path: Path) -> Path:
    source_path = directory_path / "program.c"
    library_path = directory_path / _library_name()
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
    return library_path


def _load_library(library_path: Path) -> ctypes.CDLL:
    try:
        return ctypes.CDLL(str(library_path))
    except OSError as error:
        raise NativeCompilationError(f"failed to load native shared library: {error}") from error


def _stage_existing_persistent_artifact(library_path: Path) -> _NativeArtifact | None:
    if not library_path.is_file():
        return None

    staging_directory = Path(tempfile.mkdtemp(prefix="tiny_tensor_compiler_cached_"))
    staged_library = staging_directory / _library_name()
    try:
        shutil.copy2(library_path, staged_library)
    except OSError as error:
        shutil.rmtree(staging_directory, ignore_errors=True)
        raise NativeCompilationError(f"failed to stage persistent native artifact: {error}") from error

    try:
        library = _load_library(staged_library)
    except NativeCompilationError:
        shutil.rmtree(staging_directory, ignore_errors=True)
        _remove_file(library_path)
        return None
    return _NativeArtifact(staging_directory, library)


def _persistent_library_path(
    cache_dir: str | os.PathLike[str] | None,
    source: str,
    command: list[str],
) -> Path | None:
    if cache_dir is None:
        return None

    try:
        cache_root = Path(cache_dir).expanduser().resolve()
        schema_root = cache_root / _PERSISTENT_CACHE_SCHEMA
        schema_root.mkdir(parents=True, exist_ok=True)
    except (OSError, RuntimeError) as error:
        raise NativeCompilationError(f"failed to prepare persistent native cache: {error}") from error

    digest = _persistent_cache_digest(source, command)
    return schema_root / digest / _library_name()


def _persistent_cache_digest(source: str, command: list[str]) -> str:
    compiler_path = shutil.which(command[0])
    if compiler_path is None:
        raise NativeCompilationError(f"C compiler executable not found: {command[0]}")

    try:
        resolved_compiler = Path(compiler_path).resolve()
        compiler_stat = resolved_compiler.stat()
    except OSError as error:
        raise NativeCompilationError(f"failed to fingerprint C compiler: {error}") from error

    payload = {
        "schema": _PERSISTENT_CACHE_SCHEMA,
        "source": source,
        "command": command,
        "compiler": {
            "path": str(resolved_compiler),
            "size": compiler_stat.st_size,
            "mtime_ns": compiler_stat.st_mtime_ns,
        },
        "target": {
            "os_name": os.name,
            "sys_platform": sys.platform,
            "machine": platform.machine(),
            "pointer_bits": ctypes.sizeof(ctypes.c_void_p) * 8,
            "library_name": _library_name(),
        },
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _remove_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as error:
        raise NativeCompilationError(f"failed to remove stale persistent native artifact: {error}") from error


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
