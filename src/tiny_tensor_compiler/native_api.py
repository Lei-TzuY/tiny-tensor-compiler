from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from .input_binding import BorrowedLoopProgram
from .loop_ir import LoopProgram
from .native import (
    NativeCompilationDeadlineExceeded,
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
    compilation_timeout: float | None = None,
) -> NativeExecutable:
    """Compile verified Loop IR, optionally selecting barriered OpenMP kernel scheduling."""
    policies = {}
    if compiler_timeout is not None:
        policies["compiler_timeout"] = compiler_timeout
    if compilation_timeout is not None:
        policies["compilation_timeout"] = compilation_timeout
    if parallel:
        return compile_parallel_native(
            program,
            compiler=compiler,
            cache_dir=cache_dir,
            **policies,
        )
    return _compile_native_serial(
        program,
        compiler=compiler,
        cache_dir=cache_dir,
        **policies,
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
    compilation_timeout: float | None = None,
):
    """Execute verified Loop IR through the serial or OpenMP native backend."""
    policies = {}
    if compiler_timeout is not None:
        policies["compiler_timeout"] = compiler_timeout
    if compilation_timeout is not None:
        policies["compilation_timeout"] = compilation_timeout
    if not parallel:
        return _execute_native_serial(
            program,
            compiler=compiler,
            inputs=inputs,
            cache_dir=cache_dir,
            out=out,
            **policies,
        )
    executable = compile_parallel_native(
        program,
        compiler=compiler,
        cache_dir=cache_dir,
        **policies,
    )
    return executable(inputs=inputs, out=out)


__all__ = [
    "NativeCompilationDeadlineExceeded",
    "NativeCompilationError",
    "NativeCompilationTimeout",
    "NativeExecutable",
    "clear_native_cache",
    "compile_native",
    "execute_native",
]
