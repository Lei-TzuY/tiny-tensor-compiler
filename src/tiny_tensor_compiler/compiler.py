from __future__ import annotations

import os
import threading
from collections.abc import Mapping, Sequence
from typing import Any

from .alias_views import alias_contiguous_reshapes
from .fusion_planner import fuse_elementwise
from .input_binding import borrow_inputs as bind_borrowed_inputs
from .ir import Module, SymbolicDim
from .loop_ir import lower_to_loops
from .lowering import lower_to_cpu
from .native_api import NativeExecutable, compile_native
from .symbolic import (
    SymbolicShapeError,
    bind_dynamic_shapes,
    clone_module,
    has_symbolic_shapes,
    normalize_symbolic_bindings,
    specialize_module,
    validate_dynamic_module,
)


class DynamicExecutable:
    """Reusable native executable specialized by complete runtime symbolic bindings."""

    def __init__(
        self,
        module: Module,
        compiler: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        borrow_inputs: bool = False,
        parallel: bool = False,
    ) -> None:
        self._module = clone_module(module)
        self._symbols = validate_dynamic_module(self._module)
        self._compiler = compiler
        self._cache_dir = cache_dir
        self._borrow_inputs = borrow_inputs
        self._parallel = parallel
        self._specializations: dict[tuple[int, ...], NativeExecutable] = {}
        self._lock = threading.RLock()

    @property
    def symbolic_dims(self) -> tuple[SymbolicDim, ...]:
        return self._symbols

    @property
    def symbolic_dim(self) -> SymbolicDim:
        if len(self._symbols) != 1:
            raise SymbolicShapeError(
                "symbolic_dim is available only for a single symbolic dimension"
            )
        return self._symbols[0]

    @property
    def cached_bindings(self) -> tuple[tuple[tuple[str, int], ...], ...]:
        with self._lock:
            return tuple(
                tuple(
                    (symbol.name, size)
                    for symbol, size in zip(self._symbols, key, strict=True)
                )
                for key in sorted(self._specializations)
            )

    @property
    def cached_batch_sizes(self) -> tuple[int, ...]:
        if len(self._symbols) != 1:
            raise SymbolicShapeError(
                "cached_batch_sizes is available only for a single symbolic dimension"
            )
        with self._lock:
            return tuple(sorted(key[0] for key in self._specializations))

    def specialize(
        self,
        bindings: int | Mapping[SymbolicDim | str, int],
    ) -> NativeExecutable:
        if isinstance(bindings, bool):
            raise TypeError("symbolic specialization requires an integer size, not bool")
        if isinstance(bindings, int):
            if len(self._symbols) != 1:
                raise SymbolicShapeError(
                    "integer specialization requires a single symbolic dimension"
                )
            explicit: Mapping[SymbolicDim | str, int] = {
                self._symbols[0]: bindings
            }
        elif isinstance(bindings, Mapping):
            explicit = bindings
        else:
            raise TypeError(
                "specialization requires an integer for one symbol or a binding mapping"
            )

        normalized = normalize_symbolic_bindings(self._module, explicit)
        key = tuple(normalized[symbol] for symbol in self._symbols)
        with self._lock:
            executable = self._specializations.get(key)
            if executable is not None:
                return executable
            concrete = specialize_module(self._module, normalized)
            executable = compile_module(
                concrete,
                compiler=self._compiler,
                cache_dir=self._cache_dir,
                borrow_inputs=self._borrow_inputs,
                parallel=self._parallel,
            )
            self._specializations[key] = executable
            return executable

    def execute(
        self,
        inputs: Sequence[Any] = (),
        out: Any = None,
    ):
        bindings = bind_dynamic_shapes(self._module, inputs)
        return self.specialize(bindings)(inputs=inputs, out=out)

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
    parallel: bool = False,
) -> NativeExecutable:
    """Lower verified concrete tensor IR through the native pipeline and compile eagerly."""
    if has_symbolic_shapes(module):
        raise ValueError(
            "compile_module requires concrete tensor shapes; use compile_dynamic_module "
            "for runtime symbolic specialization"
        )
    loops = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))
    if borrow_inputs:
        loops = bind_borrowed_inputs(loops)
    loops = alias_contiguous_reshapes(loops)
    if parallel:
        return compile_native(
            loops,
            compiler=compiler,
            cache_dir=cache_dir,
            parallel=True,
        )
    return compile_native(
        loops,
        compiler=compiler,
        cache_dir=cache_dir,
    )


def compile_dynamic_module(
    module: Module,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    borrow_inputs: bool = False,
    parallel: bool = False,
) -> DynamicExecutable:
    """Prepare lazy native specializations for runtime symbolic dimensions."""
    return DynamicExecutable(
        module,
        compiler=compiler,
        cache_dir=cache_dir,
        borrow_inputs=borrow_inputs,
        parallel=parallel,
    )
