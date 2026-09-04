from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from typing import Any

from .input_binding import borrow_inputs as bind_borrowed_inputs
from .ir import Module, SymbolicDim
from .loop_ir import fuse_elementwise, lower_to_loops
from .lowering import lower_to_cpu
from .native import NativeExecutable, compile_native
from .symbolic import (
    bind_dynamic_batch,
    has_symbolic_shapes,
    specialize_module,
    validate_dynamic_batch_module,
)


class DynamicExecutable:
    """Reusable runtime-specialized executable for one shared leading symbolic dimension."""

    def __init__(
        self,
        module: Module,
        compiler: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        borrow_inputs: bool = False,
    ) -> None:
        self._module = module
        self._symbol = validate_dynamic_batch_module(module)
        self._compiler = compiler
        self._cache_dir = cache_dir
        self._borrow_inputs = borrow_inputs
        self._specializations: dict[int, NativeExecutable] = {}
        self._lock = threading.RLock()

    @property
    def symbolic_dim(self) -> SymbolicDim:
        return self._symbol

    @property
    def cached_batch_sizes(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(sorted(self._specializations))

    def specialize(self, batch_size: int) -> NativeExecutable:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 0:
            raise ValueError("batch size must be a non-negative integer")
        with self._lock:
            executable = self._specializations.get(batch_size)
            if executable is not None:
                return executable
            concrete = specialize_module(self._module, {self._symbol: batch_size})
            executable = compile_module(
                concrete,
                compiler=self._compiler,
                cache_dir=self._cache_dir,
                borrow_inputs=self._borrow_inputs,
            )
            self._specializations[batch_size] = executable
            return executable

    def execute(
        self,
        inputs: Sequence[Any] = (),
        out: Any = None,
    ):
        _, batch_size = bind_dynamic_batch(self._module, inputs)
        return self.specialize(batch_size)(inputs=inputs, out=out)

    def __call__(
        self,
        inputs: Sequence[Any] = (),
        out: Any = None,
    ):
        return self.execute(inputs=inputs, out=out)


def compile_module(
    module: Module,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    borrow_inputs: bool = False,
) -> NativeExecutable:
    """Lower verified concrete tensor IR through the native pipeline and compile eagerly."""
    if has_symbolic_shapes(module):
        raise ValueError(
            "compile_module requires concrete tensor shapes; use compile_dynamic_module "
            "for runtime symbolic batch specialization"
        )
    loops = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))
    if borrow_inputs:
        loops = bind_borrowed_inputs(loops)
    return compile_native(loops, compiler=compiler, cache_dir=cache_dir)


def compile_dynamic_module(
    module: Module,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    borrow_inputs: bool = False,
) -> DynamicExecutable:
    """Prepare lazy native specializations for one shared leading symbolic batch dimension."""
    return DynamicExecutable(
        module,
        compiler=compiler,
        cache_dir=cache_dir,
        borrow_inputs=borrow_inputs,
    )
