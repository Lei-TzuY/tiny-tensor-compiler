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
    compile_deadline: float | None = None,
) -> NativeExecutable:
    """Compile verified Loop IR with optional process and total wall-clock bounds."""
    kwargs = _compile_control_kwargs(compiler_timeout, compile_deadline)
    if parallel:
        return compile_parallel_native(
            program,
            compiler=compiler,
            cache_dir=cache_dir,
            **kwargs,
        )
    return _compile_native_serial(
        program,
        compiler=compiler,
        cache_dir=cache_dir,
        **kwargs,
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
    compile_deadline: float | None = None,
):
    """Execute verified Loop IR through the serial or OpenMP native backend."""
    kwargs = _compile_control_kwargs(compiler_timeout, compile_deadline)
    if not parallel:
        return _execute_native_serial(
            program,
            compiler=compiler,
            inputs=inputs,
            cache_dir=cache_dir,
            out=out,
            **kwargs,
        )
    executable = compile_parallel_native(
        program,
        compiler=compiler,
        cache_dir=cache_dir,
        **kwargs,
    )
    return executable(inputs=inputs, out=out)


def _compile_control_kwargs(
    compiler_timeout: float | None,
    compile_deadline: float | None,
) -> dict[str, float]:
    kwargs: dict[str, float] = {}
    if compiler_timeout is not None:
        kwargs["compiler_timeout"] = compiler_timeout
    if compile_deadline is not None:
        kwargs["compile_deadline"] = compile_deadline
    return kwargs


__all__ = [
    "NativeCompilationDeadlineExceeded",
    "NativeCompilationError",
    "NativeCompilationTimeout",
    "NativeExecutable",
    "clear_native_cache",
    "compile_native",
    "execute_native",
]
