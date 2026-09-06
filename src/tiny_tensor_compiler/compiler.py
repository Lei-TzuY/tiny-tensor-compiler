from __future__ import annotations

import os
import threading
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from .admission import CompileBudget, CompileBudgetExceeded, enforce_compile_budget
from .analysis import CompilerReport
from .backends.cpu import execute_loop
from .compiler_control import normalize_compiler_timeout
from .fusion_planner import fuse_elementwise
from .input_binding import BorrowedLoopProgram
from .input_binding import borrow_inputs as bind_borrowed_inputs
from .ir import Module, SymbolicDim
from .loop_ir import LoopProgram, lower_to_loops
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

AdaptiveBackend = Literal["native", "loop"]
LoopExecutionProgram = LoopProgram | BorrowedLoopProgram


class AdaptiveExecutable:
    """Execute one concrete module through native code or verified Loop CPU fallback."""

    def __init__(
        self,
        *,
        backend: AdaptiveBackend,
        report: CompilerReport,
        native: NativeExecutable | None = None,
        loops: LoopExecutionProgram | None = None,
        budget_exceeded: CompileBudgetExceeded | None = None,
    ) -> None:
        if backend == "native":
            if native is None or loops is not None or budget_exceeded is not None:
                raise ValueError("native adaptive executable requires only a native backend")
        elif backend == "loop":
            if loops is None or native is not None or budget_exceeded is None:
                raise ValueError("loop adaptive executable requires a budget fallback program")
        else:  # pragma: no cover - internal construction is statically bounded
            raise ValueError(f"unsupported adaptive backend: {backend}")
        self._backend = backend
        self._report = report
        self._native = native
        self._loops = loops
        self._budget_exceeded = budget_exceeded

    @property
    def backend(self) -> AdaptiveBackend:
        return self._backend

    @property
    def report(self) -> CompilerReport:
        return self._report

    @property
    def budget_exceeded(self) -> CompileBudgetExceeded | None:
        return self._budget_exceeded

    def execute(self, inputs: Sequence[Any] = ()):
        """Execute without changing the selected backend or retrying native compilation."""
        if self._backend == "native":
            if self._native is None:  # pragma: no cover - constructor invariant
                raise RuntimeError("native adaptive executable is missing its backend")
            return self._native(inputs=inputs)
        if self._loops is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("loop adaptive executable is missing its program")
        return execute_loop(self._loops, inputs=inputs)

    def __call__(self, inputs: Sequence[Any] = ()):
        return self.execute(inputs=inputs)


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
        budget: CompileBudget | None = None,
        compiler_timeout: float | None = None,
    ) -> None:
        if budget is not None and not isinstance(budget, CompileBudget):
            raise TypeError("budget must be a CompileBudget or None")
        self._module = clone_module(module)
        self._symbols = validate_dynamic_module(self._module)
        self._compiler = compiler
        self._cache_dir = cache_dir
        self._borrow_inputs = borrow_inputs
        self._parallel = parallel
        self._budget = budget
        self._compiler_timeout = normalize_compiler_timeout(compiler_timeout)
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
                _display_binding(self._symbols, key)
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
        normalized, key = _normalize_specialization_bindings(
            self._module,
            self._symbols,
            bindings,
        )
        with self._lock:
            executable = self._specializations.get(key)
            if executable is not None:
                return executable
            concrete = specialize_module(self._module, normalized)
            if self._budget is None:
                if self._compiler_timeout is None:
                    executable = compile_module(
                        concrete,
                        compiler=self._compiler,
                        cache_dir=self._cache_dir,
                        borrow_inputs=self._borrow_inputs,
                        parallel=self._parallel,
                    )
                else:
                    executable = compile_module(
                        concrete,
                        compiler=self._compiler,
                        cache_dir=self._cache_dir,
                        borrow_inputs=self._borrow_inputs,
                        parallel=self._parallel,
                        compiler_timeout=self._compiler_timeout,
                    )
            elif self._compiler_timeout is None:
                executable = compile_module(
                    concrete,
                    compiler=self._compiler,
                    cache_dir=self._cache_dir,
                    borrow_inputs=self._borrow_inputs,
                    parallel=self._parallel,
                    budget=self._budget,
                )
            else:
                executable = compile_module(
                    concrete,
                    compiler=self._compiler,
                    cache_dir=self._cache_dir,
                    borrow_inputs=self._borrow_inputs,
                    parallel=self._parallel,
                    budget=self._budget,
                    compiler_timeout=self._compiler_timeout,
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


class AdaptiveDynamicExecutable:
    """Cache per-binding native-or-loop decisions under one explicit compile budget."""

    def __init__(
        self,
        module: Module,
        budget: CompileBudget,
        compiler: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        borrow_inputs: bool = False,
        parallel: bool = False,
        compiler_timeout: float | None = None,
    ) -> None:
        if not isinstance(budget, CompileBudget):
            raise TypeError("budget must be a CompileBudget")
        self._module = clone_module(module)
        self._symbols = validate_dynamic_module(self._module)
        self._budget = budget
        self._compiler = compiler
        self._cache_dir = cache_dir
        self._borrow_inputs = borrow_inputs
        self._parallel = parallel
        self._compiler_timeout = normalize_compiler_timeout(compiler_timeout)
        self._specializations: dict[tuple[int, ...], AdaptiveExecutable] = {}
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
                _display_binding(self._symbols, key)
                for key in sorted(self._specializations)
            )

    @property
    def cached_binding_backends(
        self,
    ) -> tuple[tuple[tuple[tuple[str, int], ...], AdaptiveBackend], ...]:
        with self._lock:
            return tuple(
                (
                    _display_binding(self._symbols, key),
                    self._specializations[key].backend,
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
    ) -> AdaptiveExecutable:
        normalized, key = _normalize_specialization_bindings(
            self._module,
            self._symbols,
            bindings,
        )
        with self._lock:
            executable = self._specializations.get(key)
            if executable is not None:
                return executable
            concrete = specialize_module(self._module, normalized)
            if self._compiler_timeout is None:
                executable = compile_adaptive_module(
                    concrete,
                    budget=self._budget,
                    compiler=self._compiler,
                    cache_dir=self._cache_dir,
                    borrow_inputs=self._borrow_inputs,
                    parallel=self._parallel,
                )
            else:
                executable = compile_adaptive_module(
                    concrete,
                    budget=self._budget,
                    compiler=self._compiler,
                    cache_dir=self._cache_dir,
                    borrow_inputs=self._borrow_inputs,
                    parallel=self._parallel,
                    compiler_timeout=self._compiler_timeout,
                )
            self._specializations[key] = executable
            return executable

    def execute(self, inputs: Sequence[Any] = ()):
        bindings = bind_dynamic_shapes(self._module, inputs)
        return self.specialize(bindings)(inputs=inputs)

    def __call__(self, inputs: Sequence[Any] = ()):
        return self.execute(inputs=inputs)


def compile_module(
    module: Module,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    borrow_inputs: bool = False,
    parallel: bool = False,
    budget: CompileBudget | None = None,
    compiler_timeout: float | None = None,
) -> NativeExecutable:
    """Lower verified concrete tensor IR through the native pipeline and compile eagerly."""
    normalized_timeout = normalize_compiler_timeout(compiler_timeout)
    if has_symbolic_shapes(module):
        raise ValueError(
            "compile_module requires concrete tensor shapes; use compile_dynamic_module "
            "for runtime symbolic specialization"
        )
    if budget is not None:
        enforce_compile_budget(module, budget)
    loops = _lower_concrete_module(module, borrow_inputs=borrow_inputs)
    if parallel:
        if normalized_timeout is None:
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
            parallel=True,
            compiler_timeout=normalized_timeout,
        )
    if normalized_timeout is None:
        return compile_native(
            loops,
            compiler=compiler,
            cache_dir=cache_dir,
        )
    return compile_native(
        loops,
        compiler=compiler,
        cache_dir=cache_dir,
        compiler_timeout=normalized_timeout,
    )


def compile_adaptive_module(
    module: Module,
    *,
    budget: CompileBudget,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    borrow_inputs: bool = False,
    parallel: bool = False,
    compiler_timeout: float | None = None,
) -> AdaptiveExecutable:
    """Select native compilation or verified Loop CPU from one structural budget decision."""
    if not isinstance(budget, CompileBudget):
        raise TypeError("budget must be a CompileBudget")
    normalized_timeout = normalize_compiler_timeout(compiler_timeout)
    if has_symbolic_shapes(module):
        raise ValueError(
            "compile_adaptive_module requires concrete tensor shapes; use "
            "compile_adaptive_dynamic_module for runtime symbolic specialization"
        )

    try:
        report = enforce_compile_budget(module, budget)
    except CompileBudgetExceeded as error:
        loops = _lower_concrete_module(module, borrow_inputs=borrow_inputs)
        return AdaptiveExecutable(
            backend="loop",
            report=error.report,
            loops=loops,
            budget_exceeded=error,
        )

    if normalized_timeout is None:
        native = compile_module(
            module,
            compiler=compiler,
            cache_dir=cache_dir,
            borrow_inputs=borrow_inputs,
            parallel=parallel,
        )
    else:
        native = compile_module(
            module,
            compiler=compiler,
            cache_dir=cache_dir,
            borrow_inputs=borrow_inputs,
            parallel=parallel,
            compiler_timeout=normalized_timeout,
        )
    return AdaptiveExecutable(
        backend="native",
        report=report,
        native=native,
    )


def compile_dynamic_module(
    module: Module,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    borrow_inputs: bool = False,
    parallel: bool = False,
    budget: CompileBudget | None = None,
    compiler_timeout: float | None = None,
) -> DynamicExecutable:
    """Prepare lazy native specializations for runtime symbolic dimensions."""
    return DynamicExecutable(
        module,
        compiler=compiler,
        cache_dir=cache_dir,
        borrow_inputs=borrow_inputs,
        parallel=parallel,
        budget=budget,
        compiler_timeout=compiler_timeout,
    )


def compile_adaptive_dynamic_module(
    module: Module,
    *,
    budget: CompileBudget,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    borrow_inputs: bool = False,
    parallel: bool = False,
    compiler_timeout: float | None = None,
) -> AdaptiveDynamicExecutable:
    """Prepare per-binding adaptive native-or-loop specializations."""
    if not isinstance(budget, CompileBudget):
        raise TypeError("budget must be a CompileBudget")
    return AdaptiveDynamicExecutable(
        module,
        budget,
        compiler=compiler,
        cache_dir=cache_dir,
        borrow_inputs=borrow_inputs,
        parallel=parallel,
        compiler_timeout=compiler_timeout,
    )


def _lower_concrete_module(
    module: Module,
    *,
    borrow_inputs: bool,
) -> LoopExecutionProgram:
    loops: LoopExecutionProgram = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))
    if borrow_inputs:
        loops = bind_borrowed_inputs(loops)
    return loops


def _normalize_specialization_bindings(
    module: Module,
    symbols: tuple[SymbolicDim, ...],
    bindings: int | Mapping[SymbolicDim | str, int],
):
    if isinstance(bindings, bool):
        raise TypeError("symbolic specialization requires an integer size, not bool")
    if isinstance(bindings, int):
        if len(symbols) != 1:
            raise SymbolicShapeError(
                "integer specialization requires a single symbolic dimension"
            )
        explicit: Mapping[SymbolicDim | str, int] = {symbols[0]: bindings}
    elif isinstance(bindings, Mapping):
        explicit = bindings
    else:
        raise TypeError(
            "specialization requires an integer for one symbol or a binding mapping"
        )

    normalized = normalize_symbolic_bindings(module, explicit)
    key = tuple(normalized[symbol] for symbol in symbols)
    return normalized, key


def _display_binding(
    symbols: tuple[SymbolicDim, ...],
    key: tuple[int, ...],
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (symbol.name, size)
        for symbol, size in zip(symbols, key, strict=True)
    )