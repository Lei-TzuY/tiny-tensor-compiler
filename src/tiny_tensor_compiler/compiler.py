from __future__ import annotations

import os
import threading
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from .admission import (
    CompileBudget,
    CompileBudgetExceeded,
    DynamicSpecializationBudgetExceeded,
    enforce_compile_budget,
)
from .analysis import CompilerReport
from .autodiff import (
    _pullback_linearization_modules,
    _pushforward_linearization_modules,
    _shared_linearization_modules,
    differentiate_module,
    jacobian_vector_product_module,
    vector_jacobian_product_module,
)
from .backends.cpu import execute_loop
from .compiler_control import normalize_compile_deadline, normalize_compiler_timeout
from .fusion_planner import fuse_elementwise
from .input_binding import BorrowedLoopProgram
from .input_binding import borrow_inputs as bind_borrowed_inputs
from .input_validation import prepare_runtime_inputs
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


class _LoopModuleExecutable:
    """One verified Loop program with the NativeExecutable call shape."""

    def __init__(self, program: LoopExecutionProgram) -> None:
        self._program = program

    def execute(self, inputs: Sequence[Any] = ()):
        return execute_loop(self._program, inputs=inputs)

    def __call__(self, inputs: Sequence[Any] = ()):
        return self.execute(inputs=inputs)


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


class PushforwardLinearizationState:
    """Frozen primal/tape state that can service repeated tangent queries."""

    def __init__(
        self,
        *,
        primal: np.ndarray,
        retained_inputs: tuple[np.ndarray, ...],
        tape_values: tuple[np.ndarray, ...],
        pushforward: NativeExecutable,
        tangent_count: int,
    ) -> None:
        self._primal = np.array(primal, copy=True)
        self._retained_inputs = tuple(
            np.array(value, copy=True, order="C") for value in retained_inputs
        )
        self._tape_values = tuple(
            np.array(value, copy=True, order="C") for value in tape_values
        )
        self._pushforward = pushforward
        self._tangent_count = tangent_count
        self._query_count = 0

    @property
    def primal(self) -> np.ndarray:
        return np.array(self._primal, copy=True)

    @property
    def query_count(self) -> int:
        return self._query_count

    def pushforward(self, tangents: Sequence[Any]):
        provided = tuple(tangents)
        if len(provided) != self._tangent_count:
            raise ValueError(
                f"expected {self._tangent_count} tangent inputs, got {len(provided)}"
            )
        result = self._pushforward(
            inputs=self._retained_inputs + self._tape_values + provided,
        )
        self._query_count += 1
        return result


class PushforwardLinearizationExecutable:
    """Compile one primal tape and one reusable pure pushforward program."""

    def __init__(
        self,
        module: Module,
        compiler: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        output_index: int = 0,
        wrt: Sequence[int] = (0,),
        parallel: bool = False,
        budget: CompileBudget | None = None,
    ) -> None:
        if has_symbolic_shapes(module):
            raise ValueError(
                "reusable pushforward linearization currently requires concrete tensor shapes"
            )
        primal_module, pushforward_module, tape_value_count = (
            _pushforward_linearization_modules(
                module,
                output_index=output_index,
                wrt=wrt,
            )
        )

        input_ops = sorted(
            (op for op in module.function.ops if op.opcode == "input"),
            key=lambda op: op.attrs["index"],
        )
        pushforward_input_count = sum(
            op.opcode == "input" for op in pushforward_module.function.ops
        )
        tangent_count = (
            pushforward_input_count - len(input_ops) - tape_value_count
        )
        if tangent_count < 1:
            raise RuntimeError(
                "internal compiler error: reusable pushforward has no tangent inputs"
            )

        compile_kwargs: dict[str, Any] = {
            "compiler": compiler,
            "cache_dir": cache_dir,
            "parallel": parallel,
        }
        if budget is not None:
            compile_kwargs["budget"] = budget

        self._input_types = tuple(op.results[0].type for op in input_ops)
        self._tape_value_count = tape_value_count
        self._tangent_count = tangent_count
        self._primal_tape = compile_module(primal_module, **compile_kwargs)
        self._pushforward = compile_module(pushforward_module, **compile_kwargs)

    @property
    def tape_value_count(self) -> int:
        return self._tape_value_count

    def linearize(self, inputs: Sequence[Any]) -> PushforwardLinearizationState:
        prepared = prepare_runtime_inputs(self._input_types, inputs)
        frozen_inputs = tuple(
            np.array(value, copy=True, order="C") for value in prepared
        )
        result = self._primal_tape(inputs=frozen_inputs)

        if self._tape_value_count:
            if not isinstance(result, tuple):
                raise RuntimeError(
                    "internal compiler error: primal tape returned one value unexpectedly"
                )
            expected = self._tape_value_count + 1
            if len(result) != expected:
                raise RuntimeError(
                    "internal compiler error: primal tape returned the wrong number of values"
                )
            primal = result[0]
            tape_values = tuple(result[1:])
        else:
            if isinstance(result, tuple):
                raise RuntimeError(
                    "internal compiler error: primal tape returned unexpected extra values"
                )
            primal = result
            tape_values = ()

        return PushforwardLinearizationState(
            primal=np.asarray(primal),
            retained_inputs=frozen_inputs,
            tape_values=tuple(np.asarray(value) for value in tape_values),
            pushforward=self._pushforward,
            tangent_count=self._tangent_count,
        )


class PullbackLinearizationState:
    """Frozen primal/tape state that can service repeated cotangent queries."""

    def __init__(
        self,
        *,
        primal: np.ndarray,
        retained_inputs: tuple[np.ndarray, ...],
        tape_values: tuple[np.ndarray, ...],
        pullback: NativeExecutable,
    ) -> None:
        self._primal = np.array(primal, copy=True)
        self._retained_inputs = tuple(
            np.array(value, copy=True, order="C") for value in retained_inputs
        )
        self._tape_values = tuple(
            np.array(value, copy=True, order="C") for value in tape_values
        )
        self._pullback = pullback
        self._query_count = 0

    @property
    def primal(self) -> np.ndarray:
        return np.array(self._primal, copy=True)

    @property
    def query_count(self) -> int:
        return self._query_count

    def pullback(self, cotangent: Any):
        result = self._pullback(
            inputs=self._retained_inputs + self._tape_values + (cotangent,),
        )
        self._query_count += 1
        return result


class PullbackLinearizationExecutable:
    """Compile one primal tape and one reusable pure pullback program."""

    def __init__(
        self,
        module: Module,
        compiler: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        output_index: int = 0,
        wrt: Sequence[int] = (0,),
        parallel: bool = False,
        budget: CompileBudget | None = None,
    ) -> None:
        if has_symbolic_shapes(module):
            raise ValueError(
                "reusable pullback linearization currently requires concrete tensor shapes"
            )
        primal_module, pullback_module, tape_value_count = (
            _pullback_linearization_modules(
                module,
                output_index=output_index,
                wrt=wrt,
            )
        )

        input_ops = sorted(
            (op for op in module.function.ops if op.opcode == "input"),
            key=lambda op: op.attrs["index"],
        )
        compile_kwargs: dict[str, Any] = {
            "compiler": compiler,
            "cache_dir": cache_dir,
            "parallel": parallel,
        }
        if budget is not None:
            compile_kwargs["budget"] = budget

        self._input_types = tuple(op.results[0].type for op in input_ops)
        self._tape_value_count = tape_value_count
        self._primal_tape = compile_module(primal_module, **compile_kwargs)
        self._pullback = compile_module(pullback_module, **compile_kwargs)

    @property
    def tape_value_count(self) -> int:
        return self._tape_value_count

    def linearize(self, inputs: Sequence[Any]) -> PullbackLinearizationState:
        prepared = prepare_runtime_inputs(self._input_types, inputs)
        frozen_inputs = tuple(
            np.array(value, copy=True, order="C") for value in prepared
        )
        result = self._primal_tape(inputs=frozen_inputs)

        if self._tape_value_count:
            if not isinstance(result, tuple):
                raise RuntimeError(
                    "internal compiler error: primal tape returned one value unexpectedly"
                )
            expected = self._tape_value_count + 1
            if len(result) != expected:
                raise RuntimeError(
                    "internal compiler error: primal tape returned the wrong number of values"
                )
            primal = result[0]
            tape_values = tuple(result[1:])
        else:
            if isinstance(result, tuple):
                raise RuntimeError(
                    "internal compiler error: primal tape returned unexpected extra values"
                )
            primal = result
            tape_values = ()

        return PullbackLinearizationState(
            primal=np.asarray(primal),
            retained_inputs=frozen_inputs,
            tape_values=tuple(np.asarray(value) for value in tape_values),
            pullback=self._pullback,
        )


class LinearizationState:
    """One frozen retained-state tape serving both forward and reverse queries."""

    def __init__(
        self,
        *,
        primal: np.ndarray,
        retained_inputs: tuple[np.ndarray, ...],
        tape_values: tuple[np.ndarray, ...],
        pushforward: NativeExecutable,
        pullback: NativeExecutable,
        tangent_count: int,
    ) -> None:
        self._primal = np.array(primal, copy=True)
        self._retained_inputs = tuple(
            np.array(value, copy=True, order="C") for value in retained_inputs
        )
        self._tape_values = tuple(
            np.array(value, copy=True, order="C") for value in tape_values
        )
        self._pushforward = pushforward
        self._pullback = pullback
        self._tangent_count = tangent_count
        self._pushforward_query_count = 0
        self._pullback_query_count = 0

    @property
    def primal(self) -> np.ndarray:
        return np.array(self._primal, copy=True)

    @property
    def pushforward_query_count(self) -> int:
        return self._pushforward_query_count

    @property
    def pullback_query_count(self) -> int:
        return self._pullback_query_count

    def pushforward(self, tangents: Sequence[Any]):
        provided = tuple(tangents)
        if len(provided) != self._tangent_count:
            raise ValueError(
                f"expected {self._tangent_count} tangent inputs, got {len(provided)}"
            )
        result = self._pushforward(
            inputs=self._retained_inputs + self._tape_values + provided,
        )
        self._pushforward_query_count += 1
        return result

    def pullback(self, cotangent: Any):
        result = self._pullback(
            inputs=self._retained_inputs + self._tape_values + (cotangent,),
        )
        self._pullback_query_count += 1
        return result


def _shared_linearization_runtime_contract(
    module: Module,
    *,
    output_index: int,
    wrt: Sequence[int],
) -> tuple[
    Module,
    Module,
    Module,
    int,
    tuple[Any, ...],
    int,
]:
    (
        primal_module,
        pushforward_module,
        pullback_module,
        tape_value_count,
    ) = _shared_linearization_modules(
        module,
        output_index=output_index,
        wrt=wrt,
    )

    input_ops = sorted(
        (op for op in module.function.ops if op.opcode == "input"),
        key=lambda op: op.attrs["index"],
    )
    pushforward_input_count = sum(
        op.opcode == "input" for op in pushforward_module.function.ops
    )
    tangent_count = (
        pushforward_input_count - len(input_ops) - tape_value_count
    )
    if tangent_count < 1:
        raise RuntimeError(
            "internal compiler error: shared linearization has no tangent inputs"
        )

    return (
        primal_module,
        pushforward_module,
        pullback_module,
        tape_value_count,
        tuple(op.results[0].type for op in input_ops),
        tangent_count,
    )


class LinearizationExecutable:
    """Compile one primal tape with reusable pushforward and pullback programs."""

    def __init__(
        self,
        module: Module,
        compiler: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        output_index: int = 0,
        wrt: Sequence[int] = (0,),
        parallel: bool = False,
        budget: CompileBudget | None = None,
    ) -> None:
        if has_symbolic_shapes(module):
            raise ValueError(
                "shared reusable linearization currently requires concrete tensor shapes"
            )
        (
            primal_module,
            pushforward_module,
            pullback_module,
            tape_value_count,
            input_types,
            tangent_count,
        ) = _shared_linearization_runtime_contract(
            module,
            output_index=output_index,
            wrt=wrt,
        )

        compile_kwargs: dict[str, Any] = {
            "compiler": compiler,
            "cache_dir": cache_dir,
            "parallel": parallel,
        }
        if budget is not None:
            compile_kwargs["budget"] = budget

        self._input_types = input_types
        self._tape_value_count = tape_value_count
        self._tangent_count = tangent_count
        self._primal_tape = compile_module(primal_module, **compile_kwargs)
        self._pushforward = compile_module(pushforward_module, **compile_kwargs)
        self._pullback = compile_module(pullback_module, **compile_kwargs)

    @property
    def tape_value_count(self) -> int:
        return self._tape_value_count

    def linearize(self, inputs: Sequence[Any]) -> LinearizationState:
        prepared = prepare_runtime_inputs(self._input_types, inputs)
        frozen_inputs = tuple(
            np.array(value, copy=True, order="C") for value in prepared
        )
        result = self._primal_tape(inputs=frozen_inputs)

        if self._tape_value_count:
            if not isinstance(result, tuple):
                raise RuntimeError(
                    "internal compiler error: primal tape returned one value unexpectedly"
                )
            expected = self._tape_value_count + 1
            if len(result) != expected:
                raise RuntimeError(
                    "internal compiler error: primal tape returned the wrong number of values"
                )
            primal = result[0]
            tape_values = tuple(result[1:])
        else:
            if isinstance(result, tuple):
                raise RuntimeError(
                    "internal compiler error: primal tape returned unexpected extra values"
                )
            primal = result
            tape_values = ()

        return LinearizationState(
            primal=np.asarray(primal),
            retained_inputs=frozen_inputs,
            tape_values=tuple(np.asarray(value) for value in tape_values),
            pushforward=self._pushforward,
            pullback=self._pullback,
            tangent_count=self._tangent_count,
        )


class _AdaptiveLinearizationExecutable(LinearizationExecutable):
    """One concrete retained-state bundle with one coherent backend decision."""

    def __init__(
        self,
        module: Module,
        budget: CompileBudget,
        compiler: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        output_index: int = 0,
        wrt: Sequence[int] = (0,),
        parallel: bool = False,
    ) -> None:
        if has_symbolic_shapes(module):
            raise ValueError(
                "adaptive reusable linearization currently requires concrete tensor shapes"
            )
        if not isinstance(budget, CompileBudget):
            raise TypeError("budget must be a CompileBudget")

        (
            primal_module,
            pushforward_module,
            pullback_module,
            tape_value_count,
            input_types,
            tangent_count,
        ) = _shared_linearization_runtime_contract(
            module,
            output_index=output_index,
            wrt=wrt,
        )
        components = (
            ("primal_tape", primal_module),
            ("pushforward", pushforward_module),
            ("pullback", pullback_module),
        )

        first_failure: tuple[str, CompileBudgetExceeded] | None = None
        for name, component in components:
            try:
                enforce_compile_budget(component, budget)
            except CompileBudgetExceeded as exc:
                if first_failure is None:
                    first_failure = (name, exc)

        self._backend: AdaptiveBackend = (
            "loop" if first_failure is not None else "native"
        )
        self._budget_exceeded_component = (
            first_failure[0] if first_failure is not None else None
        )
        self._budget_exceeded = (
            first_failure[1] if first_failure is not None else None
        )
        self._input_types = input_types
        self._tape_value_count = tape_value_count
        self._tangent_count = tangent_count

        if self._backend == "native":
            kwargs: dict[str, Any] = {
                "compiler": compiler,
                "cache_dir": cache_dir,
                "parallel": parallel,
            }
            self._primal_tape = compile_module(primal_module, **kwargs)
            self._pushforward = compile_module(pushforward_module, **kwargs)
            self._pullback = compile_module(pullback_module, **kwargs)
        else:
            self._primal_tape = _LoopModuleExecutable(
                _lower_concrete_module(primal_module, borrow_inputs=False)
            )
            self._pushforward = _LoopModuleExecutable(
                _lower_concrete_module(pushforward_module, borrow_inputs=False)
            )
            self._pullback = _LoopModuleExecutable(
                _lower_concrete_module(pullback_module, borrow_inputs=False)
            )

    @property
    def backend(self) -> AdaptiveBackend:
        return self._backend

    @property
    def budget_exceeded(self) -> CompileBudgetExceeded | None:
        return self._budget_exceeded

    @property
    def budget_exceeded_component(self) -> str | None:
        return self._budget_exceeded_component


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
        compile_deadline: float | None = None,
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
        self._compile_deadline = normalize_compile_deadline(compile_deadline)
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
            _enforce_dynamic_specialization_budget(
                self._symbols,
                self._specializations,
                key,
                self._budget,
            )
            concrete = specialize_module(self._module, normalized)
            kwargs: dict[str, Any] = {
                "compiler": self._compiler,
                "cache_dir": self._cache_dir,
                "borrow_inputs": self._borrow_inputs,
                "parallel": self._parallel,
            }
            if self._budget is not None:
                kwargs["budget"] = self._budget
            if self._compiler_timeout is not None:
                kwargs["compiler_timeout"] = self._compiler_timeout
            if self._compile_deadline is not None:
                kwargs["compile_deadline"] = self._compile_deadline
            executable = compile_module(concrete, **kwargs)
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


class DynamicLinearizationExecutable(DynamicExecutable):
    """Cache concrete shared linearizations by complete primal-input bindings."""

    def __init__(
        self,
        module: Module,
        compiler: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        output_index: int = 0,
        wrt: Sequence[int] = (0,),
        parallel: bool = False,
        budget: CompileBudget | None = None,
    ) -> None:
        if isinstance(wrt, (str, bytes)):
            raise TypeError("wrt must be a sequence of runtime input indices")
        try:
            frozen_wrt = tuple(wrt)
        except TypeError as exc:
            raise TypeError("wrt must be a sequence of runtime input indices") from exc

        self._output_index = output_index
        self._wrt = frozen_wrt
        super().__init__(
            module,
            compiler=compiler,
            cache_dir=cache_dir,
            borrow_inputs=False,
            parallel=parallel,
            budget=budget,
        )
        self._specializations: dict[
            tuple[int, ...], LinearizationExecutable
        ] = {}

    def specialize(
        self,
        bindings: int | Mapping[SymbolicDim | str, int],
    ) -> LinearizationExecutable:
        normalized, key = _normalize_specialization_bindings(
            self._module,
            self._symbols,
            bindings,
        )
        with self._lock:
            executable = self._specializations.get(key)
            if executable is not None:
                return executable
            _enforce_dynamic_specialization_budget(
                self._symbols,
                self._specializations,
                key,
                self._budget,
            )
            concrete = specialize_module(self._module, normalized)
            executable = compile_linearization(
                concrete,
                compiler=self._compiler,
                cache_dir=self._cache_dir,
                output_index=self._output_index,
                wrt=self._wrt,
                parallel=self._parallel,
                budget=self._budget,
            )
            self._specializations[key] = executable
            return executable

    def linearize(self, inputs: Sequence[Any]) -> LinearizationState:
        provided = tuple(inputs)
        bindings = bind_dynamic_shapes(self._module, provided)
        return self.specialize(bindings).linearize(provided)

    def execute(self, inputs: Sequence[Any] = ()) -> LinearizationState:
        return self.linearize(inputs)

    def __call__(self, inputs: Sequence[Any] = ()) -> LinearizationState:
        return self.linearize(inputs)


class AdaptiveDynamicLinearizationExecutable(
    DynamicLinearizationExecutable
):
    """Cache coherent native-or-Loop retained-state bundles per primal binding."""

    def __init__(
        self,
        module: Module,
        budget: CompileBudget,
        compiler: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        output_index: int = 0,
        wrt: Sequence[int] = (0,),
        parallel: bool = False,
    ) -> None:
        if not isinstance(budget, CompileBudget):
            raise TypeError("budget must be a CompileBudget")
        super().__init__(
            module,
            compiler=compiler,
            cache_dir=cache_dir,
            output_index=output_index,
            wrt=wrt,
            parallel=parallel,
            budget=budget,
        )
        self._specializations: dict[
            tuple[int, ...], _AdaptiveLinearizationExecutable
        ] = {}

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

    def specialize(
        self,
        bindings: int | Mapping[SymbolicDim | str, int],
    ) -> _AdaptiveLinearizationExecutable:
        normalized, key = _normalize_specialization_bindings(
            self._module,
            self._symbols,
            bindings,
        )
        with self._lock:
            executable = self._specializations.get(key)
            if executable is not None:
                return executable
            _enforce_dynamic_specialization_budget(
                self._symbols,
                self._specializations,
                key,
                self._budget,
            )
            concrete = specialize_module(self._module, normalized)
            if self._budget is None:  # pragma: no cover - constructor invariant
                raise RuntimeError(
                    "internal compiler error: adaptive linearization lost its budget"
                )
            executable = _AdaptiveLinearizationExecutable(
                concrete,
                self._budget,
                compiler=self._compiler,
                cache_dir=self._cache_dir,
                output_index=self._output_index,
                wrt=self._wrt,
                parallel=self._parallel,
            )
            self._specializations[key] = executable
            return executable


class DynamicGradientExecutable(DynamicExecutable):
    """Lazy native gradients specialized only after runtime symbolic binding."""

    def __init__(
        self,
        module: Module,
        compiler: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        output_index: int = 0,
        wrt: Sequence[int] = (0,),
        borrow_inputs: bool = False,
        parallel: bool = False,
        budget: CompileBudget | None = None,
        compiler_timeout: float | None = None,
        compile_deadline: float | None = None,
    ) -> None:
        if isinstance(wrt, (str, bytes)):
            raise TypeError("wrt must be a sequence of runtime input indices")
        try:
            frozen_wrt = tuple(wrt)
        except TypeError as exc:
            raise TypeError("wrt must be a sequence of runtime input indices") from exc

        self._output_index = output_index
        self._wrt = frozen_wrt
        super().__init__(
            module,
            compiler=compiler,
            cache_dir=cache_dir,
            borrow_inputs=borrow_inputs,
            parallel=parallel,
            budget=budget,
            compiler_timeout=compiler_timeout,
            compile_deadline=compile_deadline,
        )

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
            _enforce_dynamic_specialization_budget(
                self._symbols,
                self._specializations,
                key,
                self._budget,
            )
            concrete_forward = specialize_module(self._module, normalized)
            transformed = self._transform_concrete_forward(concrete_forward)
            kwargs: dict[str, Any] = {
                "compiler": self._compiler,
                "cache_dir": self._cache_dir,
                "borrow_inputs": self._borrow_inputs,
                "parallel": self._parallel,
            }
            if self._budget is not None:
                kwargs["budget"] = self._budget
            if self._compiler_timeout is not None:
                kwargs["compiler_timeout"] = self._compiler_timeout
            if self._compile_deadline is not None:
                kwargs["compile_deadline"] = self._compile_deadline
            executable = compile_module(transformed, **kwargs)
            self._specializations[key] = executable
            return executable

    def _transform_concrete_forward(self, concrete_forward: Module) -> Module:
        return differentiate_module(
            concrete_forward,
            output_index=self._output_index,
            wrt=self._wrt,
        )


class DynamicVJPExecutable(DynamicGradientExecutable):
    """Lazy runtime-seeded VJPs specialized after forward-input shape binding."""

    def _transform_concrete_forward(self, concrete_forward: Module) -> Module:
        return vector_jacobian_product_module(
            concrete_forward,
            output_index=self._output_index,
            wrt=self._wrt,
        )

    def execute(
        self,
        inputs: Sequence[Any] = (),
        out: Any = None,
    ):
        provided, bindings = _bind_runtime_seeded_forward_shapes(
            self._module,
            inputs,
        )
        return self.specialize(bindings)(inputs=provided, out=out)


class DynamicJVPExecutable(DynamicGradientExecutable):
    """Lazy runtime-seeded JVPs specialized after primal-input shape binding."""

    def _transform_concrete_forward(self, concrete_forward: Module) -> Module:
        return jacobian_vector_product_module(
            concrete_forward,
            output_index=self._output_index,
            wrt=self._wrt,
        )

    def execute(
        self,
        inputs: Sequence[Any] = (),
        out: Any = None,
    ):
        provided, bindings = _bind_runtime_tangent_forward_shapes(
            self._module,
            inputs,
            tangent_count=len(self._wrt),
        )
        return self.specialize(bindings)(inputs=provided, out=out)


class DynamicHVPExecutable(DynamicVJPExecutable):
    """Lazy single-input Hessian-vector products after forward specialization."""

    def __init__(
        self,
        module: Module,
        compiler: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        output_index: int = 0,
        wrt: Sequence[int] = (0,),
        borrow_inputs: bool = False,
        parallel: bool = False,
        budget: CompileBudget | None = None,
        compiler_timeout: float | None = None,
        compile_deadline: float | None = None,
    ) -> None:
        if isinstance(wrt, (str, bytes)):
            raise TypeError("wrt must be a sequence of runtime input indices")
        try:
            frozen_wrt = tuple(wrt)
        except TypeError as exc:
            raise TypeError("wrt must be a sequence of runtime input indices") from exc
        if len(frozen_wrt) != 1:
            raise ValueError("dynamic HVP requires exactly one wrt input")

        super().__init__(
            module,
            compiler=compiler,
            cache_dir=cache_dir,
            output_index=output_index,
            wrt=frozen_wrt,
            borrow_inputs=borrow_inputs,
            parallel=parallel,
            budget=budget,
            compiler_timeout=compiler_timeout,
            compile_deadline=compile_deadline,
        )

    def _transform_concrete_forward(self, concrete_forward: Module) -> Module:
        return _single_input_hvp_module(
            concrete_forward,
            output_index=self._output_index,
            wrt=self._wrt,
        )


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
        compile_deadline: float | None = None,
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
        self._compile_deadline = normalize_compile_deadline(compile_deadline)
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
            _enforce_dynamic_specialization_budget(
                self._symbols,
                self._specializations,
                key,
                self._budget,
            )
            concrete = specialize_module(self._module, normalized)
            kwargs: dict[str, Any] = {
                "budget": self._budget,
                "compiler": self._compiler,
                "cache_dir": self._cache_dir,
                "borrow_inputs": self._borrow_inputs,
                "parallel": self._parallel,
            }
            if self._compiler_timeout is not None:
                kwargs["compiler_timeout"] = self._compiler_timeout
            if self._compile_deadline is not None:
                kwargs["compile_deadline"] = self._compile_deadline
            executable = compile_adaptive_module(concrete, **kwargs)
            self._specializations[key] = executable
            return executable

    def execute(self, inputs: Sequence[Any] = ()):
        bindings = bind_dynamic_shapes(self._module, inputs)
        return self.specialize(bindings)(inputs=inputs)

    def __call__(self, inputs: Sequence[Any] = ()):
        return self.execute(inputs=inputs)


class AdaptiveDynamicGradientExecutable(AdaptiveDynamicExecutable):
    """Cache per-binding native-or-Loop gradient specializations."""

    def __init__(
        self,
        module: Module,
        budget: CompileBudget,
        compiler: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        output_index: int = 0,
        wrt: Sequence[int] = (0,),
        borrow_inputs: bool = False,
        parallel: bool = False,
        compiler_timeout: float | None = None,
        compile_deadline: float | None = None,
    ) -> None:
        if isinstance(wrt, (str, bytes)):
            raise TypeError("wrt must be a sequence of runtime input indices")
        try:
            frozen_wrt = tuple(wrt)
        except TypeError as exc:
            raise TypeError("wrt must be a sequence of runtime input indices") from exc

        self._output_index = output_index
        self._wrt = frozen_wrt
        super().__init__(
            module,
            budget,
            compiler=compiler,
            cache_dir=cache_dir,
            borrow_inputs=borrow_inputs,
            parallel=parallel,
            compiler_timeout=compiler_timeout,
            compile_deadline=compile_deadline,
        )

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
            _enforce_dynamic_specialization_budget(
                self._symbols,
                self._specializations,
                key,
                self._budget,
            )
            concrete_forward = specialize_module(self._module, normalized)
            transformed = self._transform_concrete_forward(concrete_forward)
            kwargs: dict[str, Any] = {
                "budget": self._budget,
                "compiler": self._compiler,
                "cache_dir": self._cache_dir,
                "borrow_inputs": self._borrow_inputs,
                "parallel": self._parallel,
            }
            if self._compiler_timeout is not None:
                kwargs["compiler_timeout"] = self._compiler_timeout
            if self._compile_deadline is not None:
                kwargs["compile_deadline"] = self._compile_deadline
            executable = compile_adaptive_module(transformed, **kwargs)
            self._specializations[key] = executable
            return executable

    def _transform_concrete_forward(self, concrete_forward: Module) -> Module:
        return differentiate_module(
            concrete_forward,
            output_index=self._output_index,
            wrt=self._wrt,
        )


class AdaptiveDynamicVJPExecutable(AdaptiveDynamicGradientExecutable):
    """Cache per-binding native-or-Loop runtime-seeded VJP specializations."""

    def _transform_concrete_forward(self, concrete_forward: Module) -> Module:
        return vector_jacobian_product_module(
            concrete_forward,
            output_index=self._output_index,
            wrt=self._wrt,
        )

    def execute(self, inputs: Sequence[Any] = ()):
        provided, bindings = _bind_runtime_seeded_forward_shapes(
            self._module,
            inputs,
        )
        return self.specialize(bindings)(inputs=provided)


class AdaptiveDynamicJVPExecutable(AdaptiveDynamicGradientExecutable):
    """Cache per-binding native-or-Loop runtime-seeded JVP specializations."""

    def _transform_concrete_forward(self, concrete_forward: Module) -> Module:
        return jacobian_vector_product_module(
            concrete_forward,
            output_index=self._output_index,
            wrt=self._wrt,
        )

    def execute(self, inputs: Sequence[Any] = ()):
        provided, bindings = _bind_runtime_tangent_forward_shapes(
            self._module,
            inputs,
            tangent_count=len(self._wrt),
        )
        return self.specialize(bindings)(inputs=provided)


class AdaptiveDynamicHVPExecutable(AdaptiveDynamicVJPExecutable):
    """Cache per-binding native-or-Loop single-input HVP specializations."""

    def __init__(
        self,
        module: Module,
        budget: CompileBudget,
        compiler: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        output_index: int = 0,
        wrt: Sequence[int] = (0,),
        borrow_inputs: bool = False,
        parallel: bool = False,
        compiler_timeout: float | None = None,
        compile_deadline: float | None = None,
    ) -> None:
        if isinstance(wrt, (str, bytes)):
            raise TypeError("wrt must be a sequence of runtime input indices")
        try:
            frozen_wrt = tuple(wrt)
        except TypeError as exc:
            raise TypeError("wrt must be a sequence of runtime input indices") from exc
        if len(frozen_wrt) != 1:
            raise ValueError("adaptive dynamic HVP requires exactly one wrt input")

        super().__init__(
            module,
            budget,
            compiler=compiler,
            cache_dir=cache_dir,
            output_index=output_index,
            wrt=frozen_wrt,
            borrow_inputs=borrow_inputs,
            parallel=parallel,
            compiler_timeout=compiler_timeout,
            compile_deadline=compile_deadline,
        )

    def _transform_concrete_forward(self, concrete_forward: Module) -> Module:
        return _single_input_hvp_module(
            concrete_forward,
            output_index=self._output_index,
            wrt=self._wrt,
        )


def compile_linearization(
    module: Module,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    output_index: int = 0,
    wrt: Sequence[int] = (0,),
    parallel: bool = False,
    budget: CompileBudget | None = None,
) -> LinearizationExecutable:
    """Compile one shared primal tape with reusable pushforward and pullback queries."""
    return LinearizationExecutable(
        module,
        compiler=compiler,
        cache_dir=cache_dir,
        output_index=output_index,
        wrt=wrt,
        parallel=parallel,
        budget=budget,
    )


def compile_pullback_linearization(
    module: Module,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    output_index: int = 0,
    wrt: Sequence[int] = (0,),
    parallel: bool = False,
    budget: CompileBudget | None = None,
) -> PullbackLinearizationExecutable:
    """Compile a one-shot primal tape with a reusable pure pullback."""
    return PullbackLinearizationExecutable(
        module,
        compiler=compiler,
        cache_dir=cache_dir,
        output_index=output_index,
        wrt=wrt,
        parallel=parallel,
        budget=budget,
    )


def compile_pushforward_linearization(
    module: Module,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    output_index: int = 0,
    wrt: Sequence[int] = (0,),
    parallel: bool = False,
    budget: CompileBudget | None = None,
) -> PushforwardLinearizationExecutable:
    """Compile a one-shot primal tape with a reusable pure pushforward."""
    return PushforwardLinearizationExecutable(
        module,
        compiler=compiler,
        cache_dir=cache_dir,
        output_index=output_index,
        wrt=wrt,
        parallel=parallel,
        budget=budget,
    )


def compile_module(
    module: Module,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    borrow_inputs: bool = False,
    parallel: bool = False,
    budget: CompileBudget | None = None,
    compiler_timeout: float | None = None,
    compile_deadline: float | None = None,
) -> NativeExecutable:
    """Lower verified concrete tensor IR through the native pipeline and compile eagerly."""
    normalized_timeout = normalize_compiler_timeout(compiler_timeout)
    normalized_deadline = normalize_compile_deadline(compile_deadline)
    if has_symbolic_shapes(module):
        raise ValueError(
            "compile_module requires concrete tensor shapes; use compile_dynamic_module "
            "for runtime symbolic specialization"
        )
    if budget is not None:
        enforce_compile_budget(module, budget)
    loops = _lower_concrete_module(module, borrow_inputs=borrow_inputs)
    kwargs: dict[str, Any] = {
        "compiler": compiler,
        "cache_dir": cache_dir,
    }
    if parallel:
        kwargs["parallel"] = True
    if normalized_timeout is not None:
        kwargs["compiler_timeout"] = normalized_timeout
    if normalized_deadline is not None:
        kwargs["compile_deadline"] = normalized_deadline
    return compile_native(loops, **kwargs)


def compile_adaptive_module(
    module: Module,
    *,
    budget: CompileBudget,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    borrow_inputs: bool = False,
    parallel: bool = False,
    compiler_timeout: float | None = None,
    compile_deadline: float | None = None,
) -> AdaptiveExecutable:
    """Select native compilation or verified Loop CPU from one structural budget decision."""
    if not isinstance(budget, CompileBudget):
        raise TypeError("budget must be a CompileBudget")
    normalized_timeout = normalize_compiler_timeout(compiler_timeout)
    normalized_deadline = normalize_compile_deadline(compile_deadline)
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

    kwargs: dict[str, Any] = {
        "compiler": compiler,
        "cache_dir": cache_dir,
        "borrow_inputs": borrow_inputs,
        "parallel": parallel,
    }
    if normalized_timeout is not None:
        kwargs["compiler_timeout"] = normalized_timeout
    if normalized_deadline is not None:
        kwargs["compile_deadline"] = normalized_deadline
    native = compile_module(module, **kwargs)
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
    compile_deadline: float | None = None,
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
        compile_deadline=compile_deadline,
    )


def compile_dynamic_linearization(
    module: Module,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    output_index: int = 0,
    wrt: Sequence[int] = (0,),
    parallel: bool = False,
    budget: CompileBudget | None = None,
) -> DynamicLinearizationExecutable:
    """Prepare cached concrete shared linearizations after primal shape binding."""
    return DynamicLinearizationExecutable(
        module,
        compiler=compiler,
        cache_dir=cache_dir,
        output_index=output_index,
        wrt=wrt,
        parallel=parallel,
        budget=budget,
    )


def compile_adaptive_dynamic_linearization(
    module: Module,
    *,
    budget: CompileBudget,
    output_index: int = 0,
    wrt: Sequence[int] = (0,),
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    parallel: bool = False,
) -> AdaptiveDynamicLinearizationExecutable:
    """Prepare coherent native-or-Loop retained-state bundles per binding."""
    if not isinstance(budget, CompileBudget):
        raise TypeError("budget must be a CompileBudget")
    return AdaptiveDynamicLinearizationExecutable(
        module,
        budget,
        compiler=compiler,
        cache_dir=cache_dir,
        output_index=output_index,
        wrt=wrt,
        parallel=parallel,
    )


def compile_dynamic_gradient_module(
    module: Module,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    output_index: int = 0,
    wrt: Sequence[int] = (0,),
    borrow_inputs: bool = False,
    parallel: bool = False,
    budget: CompileBudget | None = None,
    compiler_timeout: float | None = None,
    compile_deadline: float | None = None,
) -> DynamicGradientExecutable:
    """Prepare lazy gradients by specializing symbolic shapes before autodiff."""
    return DynamicGradientExecutable(
        module,
        compiler=compiler,
        cache_dir=cache_dir,
        output_index=output_index,
        wrt=wrt,
        borrow_inputs=borrow_inputs,
        parallel=parallel,
        budget=budget,
        compiler_timeout=compiler_timeout,
        compile_deadline=compile_deadline,
    )


def compile_dynamic_jvp_module(
    module: Module,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    output_index: int = 0,
    wrt: Sequence[int] = (0,),
    borrow_inputs: bool = False,
    parallel: bool = False,
    budget: CompileBudget | None = None,
    compiler_timeout: float | None = None,
    compile_deadline: float | None = None,
) -> DynamicJVPExecutable:
    """Prepare lazy runtime-seeded JVPs after symbolic primal specialization."""
    return DynamicJVPExecutable(
        module,
        compiler=compiler,
        cache_dir=cache_dir,
        output_index=output_index,
        wrt=wrt,
        borrow_inputs=borrow_inputs,
        parallel=parallel,
        budget=budget,
        compiler_timeout=compiler_timeout,
        compile_deadline=compile_deadline,
    )


def compile_dynamic_hvp_module(
    module: Module,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    output_index: int = 0,
    wrt: Sequence[int] = (0,),
    borrow_inputs: bool = False,
    parallel: bool = False,
    budget: CompileBudget | None = None,
    compiler_timeout: float | None = None,
    compile_deadline: float | None = None,
) -> DynamicHVPExecutable:
    """Prepare lazy single-input Hessian-vector products after specialization."""
    return DynamicHVPExecutable(
        module,
        compiler=compiler,
        cache_dir=cache_dir,
        output_index=output_index,
        wrt=wrt,
        borrow_inputs=borrow_inputs,
        parallel=parallel,
        budget=budget,
        compiler_timeout=compiler_timeout,
        compile_deadline=compile_deadline,
    )


def compile_dynamic_vjp_module(
    module: Module,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    output_index: int = 0,
    wrt: Sequence[int] = (0,),
    borrow_inputs: bool = False,
    parallel: bool = False,
    budget: CompileBudget | None = None,
    compiler_timeout: float | None = None,
    compile_deadline: float | None = None,
) -> DynamicVJPExecutable:
    """Prepare lazy runtime-seeded VJPs after symbolic forward specialization."""
    return DynamicVJPExecutable(
        module,
        compiler=compiler,
        cache_dir=cache_dir,
        output_index=output_index,
        wrt=wrt,
        borrow_inputs=borrow_inputs,
        parallel=parallel,
        budget=budget,
        compiler_timeout=compiler_timeout,
        compile_deadline=compile_deadline,
    )


def compile_adaptive_dynamic_jvp_module(
    module: Module,
    *,
    budget: CompileBudget,
    output_index: int = 0,
    wrt: Sequence[int] = (0,),
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    borrow_inputs: bool = False,
    parallel: bool = False,
    compiler_timeout: float | None = None,
    compile_deadline: float | None = None,
) -> AdaptiveDynamicJVPExecutable:
    """Prepare per-binding adaptive runtime-seeded JVP specializations."""
    if not isinstance(budget, CompileBudget):
        raise TypeError("budget must be a CompileBudget")
    return AdaptiveDynamicJVPExecutable(
        module,
        budget,
        compiler=compiler,
        cache_dir=cache_dir,
        output_index=output_index,
        wrt=wrt,
        borrow_inputs=borrow_inputs,
        parallel=parallel,
        compiler_timeout=compiler_timeout,
        compile_deadline=compile_deadline,
    )


def compile_adaptive_dynamic_hvp_module(
    module: Module,
    *,
    budget: CompileBudget,
    output_index: int = 0,
    wrt: Sequence[int] = (0,),
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    borrow_inputs: bool = False,
    parallel: bool = False,
    compiler_timeout: float | None = None,
    compile_deadline: float | None = None,
) -> AdaptiveDynamicHVPExecutable:
    """Prepare per-binding adaptive single-input HVP specializations."""
    if not isinstance(budget, CompileBudget):
        raise TypeError("budget must be a CompileBudget")
    return AdaptiveDynamicHVPExecutable(
        module,
        budget,
        compiler=compiler,
        cache_dir=cache_dir,
        output_index=output_index,
        wrt=wrt,
        borrow_inputs=borrow_inputs,
        parallel=parallel,
        compiler_timeout=compiler_timeout,
        compile_deadline=compile_deadline,
    )


def compile_adaptive_dynamic_vjp_module(
    module: Module,
    *,
    budget: CompileBudget,
    output_index: int = 0,
    wrt: Sequence[int] = (0,),
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    borrow_inputs: bool = False,
    parallel: bool = False,
    compiler_timeout: float | None = None,
    compile_deadline: float | None = None,
) -> AdaptiveDynamicVJPExecutable:
    """Prepare per-binding adaptive runtime-seeded VJP specializations."""
    if not isinstance(budget, CompileBudget):
        raise TypeError("budget must be a CompileBudget")
    return AdaptiveDynamicVJPExecutable(
        module,
        budget,
        compiler=compiler,
        cache_dir=cache_dir,
        output_index=output_index,
        wrt=wrt,
        borrow_inputs=borrow_inputs,
        parallel=parallel,
        compiler_timeout=compiler_timeout,
        compile_deadline=compile_deadline,
    )


def compile_adaptive_dynamic_gradient_module(
    module: Module,
    *,
    budget: CompileBudget,
    output_index: int = 0,
    wrt: Sequence[int] = (0,),
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    borrow_inputs: bool = False,
    parallel: bool = False,
    compiler_timeout: float | None = None,
    compile_deadline: float | None = None,
) -> AdaptiveDynamicGradientExecutable:
    """Prepare per-binding adaptive gradient specializations."""
    if not isinstance(budget, CompileBudget):
        raise TypeError("budget must be a CompileBudget")
    return AdaptiveDynamicGradientExecutable(
        module,
        budget,
        compiler=compiler,
        cache_dir=cache_dir,
        output_index=output_index,
        wrt=wrt,
        borrow_inputs=borrow_inputs,
        parallel=parallel,
        compiler_timeout=compiler_timeout,
        compile_deadline=compile_deadline,
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
    compile_deadline: float | None = None,
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
        compile_deadline=compile_deadline,
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


def _single_input_hvp_module(
    concrete_forward: Module,
    *,
    output_index: int,
    wrt: Sequence[int],
) -> Module:
    gradient = differentiate_module(
        concrete_forward,
        output_index=output_index,
        wrt=wrt,
    )
    return vector_jacobian_product_module(
        gradient,
        output_index=0,
        wrt=wrt,
    )


def _bind_runtime_seeded_forward_shapes(
    module: Module,
    inputs: Sequence[Any],
) -> tuple[tuple[Any, ...], dict[SymbolicDim, int]]:
    provided = tuple(inputs)
    forward_input_count = sum(
        op.opcode == "input" for op in module.function.ops
    )
    expected = forward_input_count + 1
    if len(provided) != expected:
        raise ValueError(
            f"expected {forward_input_count} forward inputs plus one cotangent, "
            f"got {len(provided)} runtime inputs"
        )
    bindings = bind_dynamic_shapes(
        module,
        provided[:forward_input_count],
    )
    return provided, bindings


def _bind_runtime_tangent_forward_shapes(
    module: Module,
    inputs: Sequence[Any],
    *,
    tangent_count: int,
) -> tuple[tuple[Any, ...], dict[SymbolicDim, int]]:
    provided = tuple(inputs)
    forward_input_count = sum(
        op.opcode == "input" for op in module.function.ops
    )
    expected = forward_input_count + tangent_count
    if len(provided) != expected:
        raise ValueError(
            f"expected {forward_input_count} forward inputs plus "
            f"{tangent_count} tangent inputs, got {len(provided)} runtime inputs"
        )
    bindings = bind_dynamic_shapes(
        module,
        provided[:forward_input_count],
    )
    return provided, bindings


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


def _enforce_dynamic_specialization_budget(
    symbols: tuple[SymbolicDim, ...],
    specializations: Mapping[tuple[int, ...], Any],
    attempted_key: tuple[int, ...],
    budget: CompileBudget | None,
) -> None:
    if budget is None or budget.max_dynamic_specializations is None:
        return
    limit = budget.max_dynamic_specializations
    if len(specializations) < limit:
        return
    cached_bindings = tuple(
        _display_binding(symbols, key)
        for key in sorted(specializations)
    )
    raise DynamicSpecializationBudgetExceeded(
        limit=limit,
        attempted_binding=_display_binding(symbols, attempted_key),
        cached_bindings=cached_bindings,
    )
