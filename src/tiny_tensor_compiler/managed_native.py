from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from .admission import CompileBudget
from .compiler import compile_module
from .ir import Module
from .native import NativeOutput
from .native_api import NativeExecutable
from .parallel_native import ParallelNativeExecutable
from .specialization_cache import (
    _release_managed_serial_artifact,
    _retain_managed_serial_artifact,
)


class ResourceManagedNativeExecutable:
    """Explicit serial-native ownership lease with deterministic close semantics."""

    def __init__(self, executable: NativeExecutable) -> None:
        if not isinstance(executable, NativeExecutable):
            raise TypeError("managed native ownership requires a NativeExecutable")
        if isinstance(executable, ParallelNativeExecutable):
            raise ValueError(
                "managed native ownership does not support process-pinned OpenMP artifacts"
            )
        self._executable = executable
        self._closed = False
        _retain_managed_serial_artifact(self, executable)

    @property
    def executable(self) -> NativeExecutable:
        """Return the ordinary executable whose artifact identity is retained by this lease."""
        return self._executable

    @property
    def closed(self) -> bool:
        return self._closed

    def execute(
        self,
        inputs: Sequence[Any] = (),
        out: NativeOutput = None,
    ):
        if self._closed:
            raise RuntimeError("resource-managed native executable is closed")
        return self._executable(inputs=inputs, out=out)

    def __call__(
        self,
        inputs: Sequence[Any] = (),
        out: NativeOutput = None,
    ):
        return self.execute(inputs=inputs, out=out)

    def close(self) -> bool:
        """Release this ownership reference and report whether the artifact was unloaded."""
        if self._closed:
            return False
        self._closed = True
        return _release_managed_serial_artifact(self, self._executable)

    def __enter__(self) -> ResourceManagedNativeExecutable:
        if self._closed:
            raise RuntimeError("resource-managed native executable is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def manage_native_executable(executable: NativeExecutable) -> ResourceManagedNativeExecutable:
    """Adopt an existing serial NativeExecutable into explicit managed ownership."""
    return ResourceManagedNativeExecutable(executable)


def compile_resource_managed_module(
    module: Module,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    borrow_inputs: bool = False,
    parallel: bool = False,
    budget: CompileBudget | None = None,
    compiler_timeout: float | None = None,
    compile_deadline: float | None = None,
) -> ResourceManagedNativeExecutable:
    """Compile one concrete module and retain its serial native artifact until close()."""
    if parallel:
        raise ValueError(
            "resource-managed concrete native ownership does not support parallel=True "
            "because OpenMP artifacts are process-pinned"
        )
    executable = compile_module(
        module,
        compiler=compiler,
        cache_dir=cache_dir,
        borrow_inputs=borrow_inputs,
        parallel=False,
        budget=budget,
        compiler_timeout=compiler_timeout,
        compile_deadline=compile_deadline,
    )
    return ResourceManagedNativeExecutable(executable)
