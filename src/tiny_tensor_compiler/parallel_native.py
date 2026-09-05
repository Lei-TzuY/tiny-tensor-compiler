from __future__ import annotations

import ctypes
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import native as native_module
from .c_abi_codegen import generate_c
from .input_binding import BorrowedLoopProgram
from .input_validation import prepare_runtime_inputs
from .loop_ir import LoopProgram

LoopExecutionProgram = LoopProgram | BorrowedLoopProgram

_WINDOWS_PIN_MARKER = ".tiny_tensor_compiler_openmp_pin"
_WINDOWS_PINNED_ARTIFACTS: dict[
    tuple[tuple[str, ...], str, str | None], native_module._NativeArtifact
] = {}


def _enable_openmp(command: list[str]) -> list[str]:
    flag = "/openmp" if native_module._is_msvc(command) else "-fopenmp"
    normalized = {argument.casefold() for argument in command}
    if flag.casefold() in normalized:
        return command
    return [*command, flag]


class ParallelNativeExecutable(native_module.NativeExecutable):
    """Native executable whose Windows OpenMP DLL is pinned until process exit."""

    def __init__(
        self,
        program: LoopExecutionProgram,
        command: tuple[str, ...],
        source: str,
        persistent_library: Path | None,
        *,
        pinned_artifact: native_module._NativeArtifact | None,
    ) -> None:
        super().__init__(
            program=program,
            command=command,
            source=source,
            persistent_library=persistent_library,
        )
        self._pinned_artifact = pinned_artifact

    def execute(
        self,
        inputs: Sequence[Any] = (),
        out: native_module.NativeOutput = None,
    ) -> native_module.ExecutionResult:
        if self._pinned_artifact is None:
            return super().execute(inputs=inputs, out=out)

        runtime_inputs = prepare_runtime_inputs(self._program.input_types, inputs)
        outputs = native_module._prepare_native_outputs(self._program, out, runtime_inputs)
        with native_module._NATIVE_CACHE_LOCK:
            return native_module._execute_artifact(
                self._program,
                self._pinned_artifact,
                runtime_inputs,
                outputs,
            )


def compile_parallel_native(
    program: LoopExecutionProgram,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> native_module.NativeExecutable:
    """Compile one verified Loop IR program with barriered OpenMP kernel scheduling."""
    command = _enable_openmp(native_module._compiler_command(compiler))
    source = generate_c(program, parallel=True)
    persistent_library = native_module._persistent_library_path(cache_dir, source, command)

    pinned_artifact: native_module._NativeArtifact | None = None
    with native_module._NATIVE_CACHE_LOCK:
        if os.name == "nt":
            key = _artifact_key(source, command, persistent_library)
            pinned_artifact = _WINDOWS_PINNED_ARTIFACTS.get(key)
            if pinned_artifact is None:
                pinned_artifact = native_module._get_or_compile_artifact(
                    source,
                    command,
                    persistent_library,
                )
                native_module._NATIVE_CACHE.pop(key, None)
                _mark_windows_process_pin(pinned_artifact)
                _WINDOWS_PINNED_ARTIFACTS[key] = pinned_artifact
        else:
            native_module._get_or_compile_artifact(source, command, persistent_library)

    return ParallelNativeExecutable(
        program=program,
        command=tuple(command),
        source=source,
        persistent_library=persistent_library,
        pinned_artifact=pinned_artifact,
    )


def _artifact_key(
    source: str,
    command: list[str],
    persistent_library: Path | None,
) -> tuple[tuple[str, ...], str, str | None]:
    persistent_identity = str(persistent_library) if persistent_library is not None else None
    return (tuple(command), source, persistent_identity)


def _mark_windows_process_pin(artifact: native_module._NativeArtifact) -> None:
    marker = artifact.directory / _WINDOWS_PIN_MARKER
    try:
        marker.write_text(str(os.getpid()), encoding="ascii")
    except OSError:
        # Pinning correctness comes from the in-process registry. The marker is only for
        # best-effort stale-directory cleanup by a later process.
        return


def _cleanup_stale_windows_pins() -> None:
    if os.name != "nt":
        return
    temp_root = Path(tempfile.gettempdir())
    for directory in temp_root.glob("tiny_tensor_compiler_*"):
        marker = directory / _WINDOWS_PIN_MARKER
        try:
            pid_text = marker.read_text(encoding="ascii").strip()
            pid = int(pid_text)
        except (FileNotFoundError, OSError, ValueError):
            continue
        if _windows_process_is_alive(pid):
            continue
        shutil.rmtree(directory, ignore_errors=True)


def _windows_process_is_alive(pid: int) -> bool:
    if os.name != "nt" or pid <= 0:
        return False

    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    open_process.restype = ctypes.c_void_p
    get_exit_code = kernel32.GetExitCodeProcess
    get_exit_code.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    get_exit_code.restype = ctypes.c_int
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int

    handle = open_process(process_query_limited_information, 0, pid)
    if not handle:
        # Access denied is safer to treat as live; an invalid/nonexistent pid is stale.
        return ctypes.get_last_error() == 5
    try:
        exit_code = ctypes.c_uint32()
        if get_exit_code(handle, ctypes.byref(exit_code)) == 0:
            return True
        return exit_code.value == still_active
    finally:
        close_handle(handle)


_cleanup_stale_windows_pins()
