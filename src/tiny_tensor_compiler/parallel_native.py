from __future__ import annotations

import os

from . import native as native_module
from .c_abi_codegen import generate_c
from .input_binding import BorrowedLoopProgram
from .loop_ir import LoopProgram

LoopExecutionProgram = LoopProgram | BorrowedLoopProgram


def _enable_openmp(command: list[str]) -> list[str]:
    flag = "/openmp" if native_module._is_msvc(command) else "-fopenmp"
    normalized = {argument.casefold() for argument in command}
    if flag.casefold() in normalized:
        return command
    return [*command, flag]


def compile_parallel_native(
    program: LoopExecutionProgram,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> native_module.NativeExecutable:
    """Compile one verified Loop IR program with barriered OpenMP kernel scheduling."""
    command = _enable_openmp(native_module._compiler_command(compiler))
    source = generate_c(program, parallel=True)
    persistent_library = native_module._persistent_library_path(cache_dir, source, command)

    with native_module._NATIVE_CACHE_LOCK:
        native_module._get_or_compile_artifact(source, command, persistent_library)

    return native_module.NativeExecutable(
        program=program,
        command=tuple(command),
        source=source,
        persistent_library=persistent_library,
    )
