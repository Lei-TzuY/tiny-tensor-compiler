from __future__ import annotations

import atexit
import ctypes
import hashlib
import json
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import native_cache_lock
from .c_abi_codegen import generate_c
from .compiler_control import (
    normalize_compile_deadline,
    normalize_compiler_timeout,
    remaining_compile_deadline,
    start_compile_deadline,
)
from .input_validation import prepare_runtime_inputs
from .ir import TensorType
from .loop_ir import LoopProgram

ExecutionResult = np.ndarray | tuple[np.ndarray, ...]
NativeOutput = np.ndarray | Sequence[np.ndarray] | None


class NativeCompilationError(RuntimeError):
    """Raised when generated C cannot be compiled or loaded for native execution."""


class NativeCompilationTimeout(NativeCompilationError):
    """Raised when one external compiler process exceeds its explicit timeout."""

    def __init__(self, command: Sequence[str], timeout: float) -> None:
        self.command = tuple(command)
        self.timeout = timeout
        super().__init__(
            f"native C compilation exceeded compiler timeout of {timeout:g}s: "
            f"{shlex.join(self.command)}"
        )


class NativeCompilationDeadlineExceeded(NativeCompilationError):
    """Raised when one total native build budget expires before an artifact is ready."""

    def __init__(
        self,
        stage: str,
        deadline: float,
        command: Sequence[str] = (),
    ) -> None:
        self.stage = stage
        self.deadline = deadline
        self.command = tuple(command)
        detail = f" during {stage}"
        if self.command:
            detail += f": {shlex.join(self.command)}"
        super().__init__(
            f"native compilation exceeded total compile deadline of {deadline:g}s{detail}"
        )


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
        compiler_timeout: float | None = None,
        compile_deadline: float | None = None,
    ) -> None:
        self._program = program
        self._command = command
        self._source = source
        self._persistent_library = persistent_library
        self._compiler_timeout = compiler_timeout
        self._compile_deadline = compile_deadline

    def execute(
        self,
        inputs: Sequence[Any] = (),
        out: NativeOutput = None,
    ) -> ExecutionResult:
        """Execute with exact runtime-input/output validation and cached native code."""
        runtime_inputs = prepare_runtime_inputs(self._program.input_types, inputs)
        outputs = _prepare_native_outputs(self._program, out, runtime_inputs)
        command = list(self._command)
        with _NATIVE_CACHE_LOCK:
            artifact = _get_or_compile_artifact(
                self._source,
                command,
                self._persistent_library,
                compiler_timeout=self._compiler_timeout,
                compile_deadline=self._compile_deadline,
            )
            return _execute_artifact(self._program, artifact, runtime_inputs, outputs)

    def __call__(
        self,
        inputs: Sequence[Any] = (),
        out: NativeOutput = None,
    ) -> ExecutionResult:
        return self.execute(inputs, out=out)


_PERSISTENT_CACHE_SCHEMA = "native-v2"
_PERSISTENT_MANIFEST_NAME = "manifest.json"
_persistent_cache_lease = native_cache_lock.persistent_cache_lease
_NATIVE_CACHE: dict[tuple[tuple[str, ...], str, str | None], _NativeArtifact] = {}
_NATIVE_CACHE_LOCK = threading.RLock()


def compile_native(
    program: LoopProgram,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    compiler_timeout: float | None = None,
    compile_deadline: float | None = None,
) -> NativeExecutable:
    """Eagerly compile/load a loop program and return a reusable executable."""
    normalized_timeout = normalize_compiler_timeout(compiler_timeout)
    normalized_deadline = normalize_compile_deadline(compile_deadline)
    command = _compiler_command(compiler)
    source = generate_c(program)
    persistent_library = _persistent_library_path(cache_dir, source, command)

    with _NATIVE_CACHE_LOCK:
        _get_or_compile_artifact(
            source,
            command,
            persistent_library,
            compiler_timeout=normalized_timeout,
            compile_deadline=normalized_deadline,
        )

    return NativeExecutable(
        program=program,
        command=tuple(command),
        source=source,
        persistent_library=persistent_library,
        compiler_timeout=normalized_timeout,
        compile_deadline=normalized_deadline,
    )


def execute_native(
    program: LoopProgram,
    compiler: str | None = None,
    inputs: Sequence[Any] = (),
    cache_dir: str | os.PathLike[str] | None = None,
    out: NativeOutput = None,
    *,
    compiler_timeout: float | None = None,
    compile_deadline: float | None = None,
) -> ExecutionResult:
    """Compile or reuse generated C and execute it on the native CPU."""
    normalized_timeout = normalize_compiler_timeout(compiler_timeout)
    normalized_deadline = normalize_compile_deadline(compile_deadline)
    runtime_inputs = prepare_runtime_inputs(program.input_types, inputs)
    outputs = _prepare_native_outputs(program, out, runtime_inputs)
    command = _compiler_command(compiler)
    source = generate_c(program)
    persistent_library = _persistent_library_path(cache_dir, source, command)

    with _NATIVE_CACHE_LOCK:
        artifact = _get_or_compile_artifact(
            source,
            command,
            persistent_library,
            compiler_timeout=normalized_timeout,
            compile_deadline=normalized_deadline,
        )
        return _execute_artifact(program, artifact, runtime_inputs, outputs)


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


def _prepare_native_outputs(
    program: LoopProgram,
    output: NativeOutput,
    runtime_inputs: Sequence[np.ndarray[Any, Any]],
) -> tuple[np.ndarray, ...]:
    return_types = _return_types(program)
    output_count = len(return_types)

    if output_count == 1:
        if output is None or isinstance(output, np.ndarray):
            candidates: tuple[np.ndarray | None, ...] = (output,)
        else:
            raise TypeError("output must be a numpy.ndarray")
    else:
        if output is None:
            candidates = (None,) * output_count
        elif isinstance(output, np.ndarray) or not isinstance(output, Sequence):
            raise TypeError("multi-output program requires a sequence of numpy.ndarray outputs")
        else:
            candidates = tuple(output)
            if len(candidates) != output_count:
                raise ValueError(
                    f"multi-output program requires {output_count} output arrays, got {len(candidates)}"
                )

    outputs = tuple(
        _prepare_native_output_array(
            return_type,
            candidate,
            runtime_inputs,
            label="output" if output_count == 1 else f"output {index}",
        )
        for index, (return_type, candidate) in enumerate(
            zip(return_types, candidates, strict=True)
        )
    )

    for left_index, left in enumerate(outputs):
        for right_index in range(left_index + 1, len(outputs)):
            if np.shares_memory(left, outputs[right_index]):
                raise ValueError(
                    f"outputs {left_index} and {right_index} must not overlap"
                )
    return outputs


def _prepare_native_output_array(
    return_type: TensorType,
    output: np.ndarray | None,
    runtime_inputs: Sequence[np.ndarray[Any, Any]],
    *,
    label: str,
) -> np.ndarray:
    expected_dtype = np.dtype(return_type.dtype.to_numpy())
    if output is None:
        return np.empty(return_type.shape, dtype=expected_dtype)
    if not isinstance(output, np.ndarray):
        raise TypeError(f"{label} must be a numpy.ndarray")
    if tuple(output.shape) != return_type.shape:
        raise ValueError(
            f"{label} shape {tuple(output.shape)} does not match expected {return_type.shape}"
        )
    if output.dtype != expected_dtype:
        raise ValueError(f"{label} dtype {output.dtype} does not match expected {expected_dtype}")
    if not output.flags.c_contiguous:
        raise ValueError(f"{label} must be C-contiguous")
    if not output.flags.writeable:
        raise ValueError(f"{label} must be writable")
    if not output.flags.aligned:
        raise ValueError(f"{label} must be aligned for its dtype")
    for index, runtime_input in enumerate(runtime_inputs):
        if np.shares_memory(output, runtime_input):
            raise ValueError(f"{label} must not overlap runtime input {index}")
    return output


def _execute_artifact(
    program: LoopProgram,
    artifact: _NativeArtifact,
    runtime_inputs: Sequence[np.ndarray[Any, Any]],
    outputs: tuple[np.ndarray, ...],
) -> ExecutionResult:
    return_types = _return_types(program)
    output_pointer_types = tuple(
        _pointer_type(return_type.dtype.to_numpy()) for return_type in return_types
    )
    input_pointer_types = tuple(
        _pointer_type(input_type.dtype.to_numpy()) for input_type in program.input_types
    )
    runner = artifact.library.tiny_tensor_run
    runner.argtypes = [*output_pointer_types, *input_pointer_types]
    runner.restype = None
    arguments = [
        array.ctypes.data_as(pointer_type)
        for array, pointer_type in zip(outputs, output_pointer_types, strict=True)
    ]
    arguments.extend(
        array.ctypes.data_as(pointer_type)
        for array, pointer_type in zip(
            runtime_inputs,
            input_pointer_types,
            strict=True,
        )
    )
    runner(*arguments)
    return outputs[0] if len(outputs) == 1 else outputs


def _get_or_compile_artifact(
    source: str,
    command: list[str],
    persistent_library: Path | None,
    *,
    compiler_timeout: float | None = None,
    compile_deadline: float | None = None,
) -> _NativeArtifact:
    persistent_identity = str(persistent_library) if persistent_library is not None else None
    key = (tuple(command), source, persistent_identity)
    artifact = _NATIVE_CACHE.get(key)
    if artifact is not None:
        return artifact

    deadline_at = start_compile_deadline(compile_deadline)
    if persistent_library is None:
        artifact = _compile_artifact(
            source,
            command,
            compiler_timeout=compiler_timeout,
            deadline_at=deadline_at,
            compile_deadline=compile_deadline,
        )
    else:
        artifact = _get_or_compile_persistent_artifact(
            source,
            command,
            persistent_library,
            compiler_timeout=compiler_timeout,
            deadline_at=deadline_at,
            compile_deadline=compile_deadline,
        )
    _NATIVE_CACHE[key] = artifact
    return artifact


def _compile_artifact(
    source: str,
    command: list[str],
    *,
    compiler_timeout: float | None = None,
    deadline_at: float | None = None,
    compile_deadline: float | None = None,
) -> _NativeArtifact:
    directory_path = Path(tempfile.mkdtemp(prefix="tiny_tensor_compiler_"))
    try:
        library_path = _compile_source(
            source,
            command,
            directory_path,
            compiler_timeout=compiler_timeout,
            deadline_at=deadline_at,
            compile_deadline=compile_deadline,
        )
        library = _load_library(library_path)
    except Exception:
        shutil.rmtree(directory_path, ignore_errors=True)
        raise

    return _NativeArtifact(directory_path, library)


def _get_or_compile_persistent_artifact(
    source: str,
    command: list[str],
    library_path: Path,
    *,
    compiler_timeout: float | None = None,
    deadline_at: float | None = None,
    compile_deadline: float | None = None,
) -> _NativeArtifact:
    try:
        lease = (
            _persistent_cache_lease(library_path)
            if deadline_at is None
            else _persistent_cache_lease(library_path, deadline_at=deadline_at)
        )
        with lease:
            _require_compile_deadline(
                deadline_at,
                compile_deadline,
                stage="persistent-cache lease",
            )
            cached = _stage_existing_persistent_artifact(library_path)
            if cached is not None:
                return cached

            schema_root = library_path.parent.parent
            build_directory = Path(tempfile.mkdtemp(prefix=".build-", dir=schema_root))
            try:
                compiled_library = _compile_source(
                    source,
                    command,
                    build_directory,
                    compiler_timeout=compiler_timeout,
                    deadline_at=deadline_at,
                    compile_deadline=compile_deadline,
                )
                compiled_manifest = build_directory / _PERSISTENT_MANIFEST_NAME
                _write_persistent_manifest(
                    compiled_manifest,
                    library_path,
                    _sha256_file(compiled_library),
                )
                library_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path = _persistent_manifest_path(library_path)
                try:
                    os.replace(compiled_library, library_path)
                    os.replace(compiled_manifest, manifest_path)
                except OSError as error:
                    _invalidate_persistent_entry(library_path)
                    raise NativeCompilationError(
                        f"failed to publish persistent native artifact: {error}"
                    ) from error

                staged = _stage_existing_persistent_artifact(library_path)
                if staged is None:
                    raise NativeCompilationError(
                        "newly compiled persistent native artifact could not be loaded"
                    )
                return staged
            finally:
                shutil.rmtree(build_directory, ignore_errors=True)
    except native_cache_lock.PersistentCacheLeaseDeadlineExceeded as error:
        if compile_deadline is None:  # pragma: no cover - constructor path requires a deadline
            raise NativeCompilationError(
                f"failed to acquire persistent native cache lease: {error}"
            ) from error
        raise NativeCompilationDeadlineExceeded(
            "persistent-cache lease",
            compile_deadline,
        ) from error
    except native_cache_lock.PersistentCacheLeaseError as error:
        raise NativeCompilationError(
            f"failed to acquire persistent native cache lease: {error}"
        ) from error


def _compile_source(
    source: str,
    command: list[str],
    directory_path: Path,
    *,
    compiler_timeout: float | None = None,
    deadline_at: float | None = None,
    compile_deadline: float | None = None,
) -> Path:
    source_path = directory_path / "program.c"
    library_path = directory_path / _library_name()
    source_path.write_text(source, encoding="utf-8")
    compile_command = _build_compile_command(command, source_path.name, library_path.name)
    effective_timeout, deadline_limited = _effective_compiler_timeout(
        compile_command,
        compiler_timeout=compiler_timeout,
        deadline_at=deadline_at,
        compile_deadline=compile_deadline,
    )
    if effective_timeout is None:
        completed = subprocess.run(
            compile_command,
            cwd=directory_path,
            capture_output=True,
            text=True,
            check=False,
        )
    else:
        completed = _run_compiler_with_timeout(
            compile_command,
            directory_path,
            effective_timeout,
            deadline_limited=deadline_limited,
            compile_deadline=compile_deadline,
        )
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip() or "no compiler output"
        raise NativeCompilationError(
            f"native C compilation failed with exit code {completed.returncode}: {details}"
        )
    return library_path


def _effective_compiler_timeout(
    compile_command: list[str],
    *,
    compiler_timeout: float | None,
    deadline_at: float | None,
    compile_deadline: float | None,
) -> tuple[float | None, bool]:
    remaining = remaining_compile_deadline(deadline_at)
    if remaining is None:
        return compiler_timeout, False
    if remaining <= 0.0:
        if compile_deadline is None:  # pragma: no cover - absolute deadline has an owner
            raise RuntimeError("compile deadline duration missing for active deadline")
        raise NativeCompilationDeadlineExceeded(
            "compiler",
            compile_deadline,
            compile_command,
        )
    if compiler_timeout is None or remaining < compiler_timeout:
        return remaining, True
    return compiler_timeout, False


def _require_compile_deadline(
    deadline_at: float | None,
    compile_deadline: float | None,
    *,
    stage: str,
) -> None:
    remaining = remaining_compile_deadline(deadline_at)
    if remaining is None or remaining > 0.0:
        return
    if compile_deadline is None:  # pragma: no cover - absolute deadline has an owner
        raise RuntimeError("compile deadline duration missing for active deadline")
    raise NativeCompilationDeadlineExceeded(stage, compile_deadline)


def _run_compiler_with_timeout(
    compile_command: list[str],
    directory_path: Path,
    timeout: float,
    *,
    deadline_limited: bool = False,
    compile_deadline: float | None = None,
) -> subprocess.CompletedProcess[str]:
    popen_kwargs: dict[str, Any] = {
        "cwd": directory_path,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    elif os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    process = subprocess.Popen(compile_command, **popen_kwargs)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        _terminate_compiler_process_tree(process)
        _reap_timed_out_compiler(process)
        if deadline_limited:
            if compile_deadline is None:  # pragma: no cover - deadline-limited path owns it
                raise RuntimeError("compile deadline duration missing for active deadline") from error
            raise NativeCompilationDeadlineExceeded(
                "compiler",
                compile_deadline,
                compile_command,
            ) from error
        raise NativeCompilationTimeout(compile_command, float(error.timeout)) from error

    return subprocess.CompletedProcess(
        compile_command,
        process.returncode,
        stdout,
        stderr,
    )


def _terminate_compiler_process_tree(process: subprocess.Popen[str]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            if process.poll() is None:
                process.kill()
        return

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            if process.poll() is None:
                process.kill()
        return

    if process.poll() is None:  # pragma: no cover - unsupported native platforms fail earlier
        process.kill()


def _reap_timed_out_compiler(process: subprocess.Popen[str]) -> None:
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


def _load_library(library_path: Path) -> ctypes.CDLL:
    try:
        return ctypes.CDLL(str(library_path))
    except OSError as error:
        raise NativeCompilationError(f"failed to load native shared library: {error}") from error


def _stage_existing_persistent_artifact(library_path: Path) -> _NativeArtifact | None:
    manifest_path = _persistent_manifest_path(library_path)
    if not library_path.is_file() and not manifest_path.is_file():
        return None
    if not library_path.is_file() or not manifest_path.is_file():
        _invalidate_persistent_entry(library_path)
        return None

    manifest = _read_persistent_manifest(manifest_path)
    if not _persistent_manifest_matches(manifest, library_path):
        _invalidate_persistent_entry(library_path)
        return None
    if manifest["library_sha256"] != _sha256_file(library_path):
        _invalidate_persistent_entry(library_path)
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
        _invalidate_persistent_entry(library_path)
        return None
    return _NativeArtifact(staging_directory, library)


def _persistent_manifest_path(library_path: Path) -> Path:
    return library_path.with_name(_PERSISTENT_MANIFEST_NAME)


def _read_persistent_manifest(manifest_path: Path) -> dict[str, object] | None:
    try:
        parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _persistent_manifest_matches(
    manifest: dict[str, object] | None,
    library_path: Path,
) -> bool:
    if manifest is None:
        return False
    expected = {
        "schema": _PERSISTENT_CACHE_SCHEMA,
        "digest": library_path.parent.name,
        "library": library_path.name,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return False
    library_sha256 = manifest.get("library_sha256")
    return (
        isinstance(library_sha256, str)
        and len(library_sha256) == 64
        and all(character in "0123456789abcdef" for character in library_sha256)
    )


def _write_persistent_manifest(
    manifest_path: Path,
    library_path: Path,
    library_sha256: str,
) -> None:
    manifest = {
        "schema": _PERSISTENT_CACHE_SCHEMA,
        "digest": library_path.parent.name,
        "library": library_path.name,
        "library_sha256": library_sha256,
    }
    try:
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise NativeCompilationError(f"failed to write persistent native manifest: {error}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise NativeCompilationError(f"failed to hash persistent native artifact: {error}") from error
    return digest.hexdigest()


def _invalidate_persistent_entry(library_path: Path) -> None:
    _remove_file(_persistent_manifest_path(library_path))
    _remove_file(library_path)


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


def _return_types(program: LoopProgram) -> tuple[TensorType, ...]:
    types = program.value_types
    try:
        return tuple(types[slot] for slot in program.return_slots)
    except KeyError as error:
        raise RuntimeError("verified loop IR return value unexpectedly has no type") from error


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