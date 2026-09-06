from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from .input_binding import BorrowedLoopProgram
from .loop_ir import LoopProgram
from .native import (
    NativeCompilationError,
    NativeCompilationTimeout,
    NativeExecutable,
    NativeOutput,
    clear_native_cache,
)
from .native import compile_native as _compile_native_serial
from .native import execute_native as _execute_native_serial
from .parallel_native import compile_parallel_native

LoopExecutionProgram = LoopProgram | BorrowedLoopProgram


def compile_native(
    program: LoopExecutionProgram,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    parallel: bool = False,
    compiler_timeout: float | None = None,
) -> NativeExecutable:
    """Compile verified Loop IR, optionally selecting barriered OpenMP kernel scheduling."""
    if parallel:
        if compiler_timeout is None:
            return compile_parallel_native(program, compiler=compiler, cache_dir=cache_dir)
        return compile_parallel_native(
            program,
            compiler=compiler,
            cache_dir=cache_dir,
            compiler_timeout=compiler_timeout,
        )
    if compiler_timeout is None:
        return _compile_native_serial(program, compiler=compiler, cache_dir=cache_dir)
    return _compile_native_serial(
        program,
        compiler=compiler,
        cache_dir=cache_dir,
        compiler_timeout=compiler_timeout,
    )


def execute_native(
    program: LoopExecutionProgram,
    compiler: str | None = None,
    inputs: Sequence[Any] = (),
    cache_dir: str | os.PathLike[str] | None = None,
    out: NativeOutput = None,
    *,
    parallel: bool = False,
    compiler_timeout: float | None = None,
):
    """Execute verified Loop IR through the serial or OpenMP native backend."""
    if not parallel:
        if compiler_timeout is None:
            return _execute_native_serial(
                program,
                compiler=compiler,
                inputs=inputs,
                cache_dir=cache_dir,
                out=out,
            )
        return _execute_native_serial(
            program,
            compiler=compiler,
            inputs=inputs,
            cache_dir=cache_dir,
            out=out,
            compiler_timeout=compiler_timeout,
        )
    if compiler_timeout is None:
        executable = compile_parallel_native(program, compiler=compiler, cache_dir=cache_dir)
    else:
        executable = compile_parallel_native(
            program,
            compiler=compiler,
            cache_dir=cache_dir,
            compiler_timeout=compiler_timeout,
        )
    return executable(inputs=inputs, out=out)


__all__ = [
    "NativeCompilationError",
    "NativeCompilationTimeout",
    "NativeExecutable",
    "clear_native_cache",
    "compile_native",
    "execute_native",
]
