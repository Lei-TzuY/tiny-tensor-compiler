from __future__ import annotations

import os
import weakref
from collections.abc import Mapping

from . import native as native_module
from .admission import CompileBudget
from .compiler import (
    AdaptiveDynamicExecutable,
    AdaptiveExecutable,
    DynamicExecutable,
    _display_binding,
    _normalize_specialization_bindings,
)
from .ir import Module, SymbolicDim
from .native_api import NativeExecutable

BindingDisplay = tuple[tuple[str, int], ...]
ArtifactIdentity = tuple[tuple[str, ...], str, str | None]
_MANAGED_ARTIFACT_OWNERS: dict[
    ArtifactIdentity,
    dict[weakref.ReferenceType, int],
] = {}


class ResourceManagedDynamicExecutable(DynamicExecutable):
    """Dynamic executable with deterministic LRU retention and serial artifact release."""

    def __init__(
        self,
        module: Module,
        compiler: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        max_cached_specializations: int,
        borrow_inputs: bool = False,
        parallel: bool = False,
        budget: CompileBudget | None = None,
        compiler_timeout: float | None = None,
        compile_deadline: float | None = None,
    ) -> None:
        _validate_cache_limit(max_cached_specializations)
        _validate_managed_budget(budget)
        if parallel:
            raise ValueError(
                "resource-managed specialization eviction does not support parallel=True "
                "because Windows OpenMP artifacts are process-pinned"
            )
        super().__init__(
            module,
            compiler=compiler,
            cache_dir=cache_dir,
            borrow_inputs=borrow_inputs,
            parallel=False,
            budget=budget,
            compiler_timeout=compiler_timeout,
            compile_deadline=compile_deadline,
        )
        self._max_cached_specializations = max_cached_specializations
        self._eviction_count = 0
        self._released_native_artifact_count = 0

    @property
    def max_cached_specializations(self) -> int:
        return self._max_cached_specializations

    @property
    def retained_bindings_lru(self) -> tuple[BindingDisplay, ...]:
        """Return retained bindings from least to most recently used."""
        with self._lock:
            return tuple(_display_binding(self._symbols, key) for key in self._specializations)

    @property
    def eviction_count(self) -> int:
        with self._lock:
            return self._eviction_count

    @property
    def released_native_artifact_count(self) -> int:
        with self._lock:
            return self._released_native_artifact_count

    def specialize(
        self,
        bindings: int | Mapping[SymbolicDim | str, int],
    ) -> NativeExecutable:
        _, key = _normalize_specialization_bindings(self._module, self._symbols, bindings)
        with self._lock:
            cached = self._specializations.pop(key, None)
            if cached is not None:
                self._specializations[key] = cached
                return cached

            executable = super().specialize(bindings)
            self._specializations.pop(key)
            self._specializations[key] = executable
            _retain_managed_serial_artifact(self, executable)
            self._evict_to_limit()
            return executable

    def _evict_to_limit(self) -> None:
        while len(self._specializations) > self._max_cached_specializations:
            oldest_key = next(iter(self._specializations))
            executable = self._specializations.pop(oldest_key)
            self._eviction_count += 1
            if _release_managed_serial_artifact(self, executable):
                self._released_native_artifact_count += 1


class ResourceManagedAdaptiveDynamicExecutable(AdaptiveDynamicExecutable):
    """Adaptive dynamic executable with deterministic LRU specialization retention."""

    def __init__(
        self,
        module: Module,
        budget: CompileBudget,
        compiler: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        max_cached_specializations: int,
        borrow_inputs: bool = False,
        parallel: bool = False,
        compiler_timeout: float | None = None,
        compile_deadline: float | None = None,
    ) -> None:
        _validate_cache_limit(max_cached_specializations)
        _validate_managed_budget(budget)
        if parallel:
            raise ValueError(
                "resource-managed specialization eviction does not support parallel=True "
                "because Windows OpenMP artifacts are process-pinned"
            )
        super().__init__(
            module,
            budget,
            compiler=compiler,
            cache_dir=cache_dir,
            borrow_inputs=borrow_inputs,
            parallel=False,
            compiler_timeout=compiler_timeout,
            compile_deadline=compile_deadline,
        )
        self._max_cached_specializations = max_cached_specializations
        self._eviction_count = 0
        self._released_native_artifact_count = 0

    @property
    def max_cached_specializations(self) -> int:
        return self._max_cached_specializations

    @property
    def retained_bindings_lru(self) -> tuple[BindingDisplay, ...]:
        with self._lock:
            return tuple(_display_binding(self._symbols, key) for key in self._specializations)

    @property
    def eviction_count(self) -> int:
        with self._lock:
            return self._eviction_count

    @property
    def released_native_artifact_count(self) -> int:
        with self._lock:
            return self._released_native_artifact_count

    def specialize(
        self,
        bindings: int | Mapping[SymbolicDim | str, int],
    ) -> AdaptiveExecutable:
        _, key = _normalize_specialization_bindings(self._module, self._symbols, bindings)
        with self._lock:
            cached = self._specializations.pop(key, None)
            if cached is not None:
                self._specializations[key] = cached
                return cached

            executable = super().specialize(bindings)
            self._specializations.pop(key)
            self._specializations[key] = executable
            if executable.backend == "native" and executable._native is not None:
                _retain_managed_serial_artifact(self, executable._native)
            self._evict_to_limit()
            return executable

    def _evict_to_limit(self) -> None:
        while len(self._specializations) > self._max_cached_specializations:
            oldest_key = next(iter(self._specializations))
            executable = self._specializations.pop(oldest_key)
            self._eviction_count += 1
            if (
                executable.backend == "native"
                and executable._native is not None
                and _release_managed_serial_artifact(self, executable._native)
            ):
                self._released_native_artifact_count += 1


def compile_resource_managed_dynamic_module(
    module: Module,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    max_cached_specializations: int,
    borrow_inputs: bool = False,
    parallel: bool = False,
    budget: CompileBudget | None = None,
    compiler_timeout: float | None = None,
    compile_deadline: float | None = None,
) -> ResourceManagedDynamicExecutable:
    """Prepare serial native specializations with bounded LRU retention."""
    return ResourceManagedDynamicExecutable(
        module,
        compiler=compiler,
        cache_dir=cache_dir,
        max_cached_specializations=max_cached_specializations,
        borrow_inputs=borrow_inputs,
        parallel=parallel,
        budget=budget,
        compiler_timeout=compiler_timeout,
        compile_deadline=compile_deadline,
    )


def compile_resource_managed_adaptive_dynamic_module(
    module: Module,
    *,
    budget: CompileBudget,
    max_cached_specializations: int,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    borrow_inputs: bool = False,
    parallel: bool = False,
    compiler_timeout: float | None = None,
    compile_deadline: float | None = None,
) -> ResourceManagedAdaptiveDynamicExecutable:
    """Prepare adaptive specializations with bounded LRU retention."""
    return ResourceManagedAdaptiveDynamicExecutable(
        module,
        budget,
        compiler=compiler,
        cache_dir=cache_dir,
        max_cached_specializations=max_cached_specializations,
        borrow_inputs=borrow_inputs,
        parallel=parallel,
        compiler_timeout=compiler_timeout,
        compile_deadline=compile_deadline,
    )


def _artifact_identity(executable: NativeExecutable) -> ArtifactIdentity:
    persistent = executable._persistent_library
    persistent_identity = str(persistent) if persistent is not None else None
    return (tuple(executable._command), executable._source, persistent_identity)


def _owner_ref(owner: object, key: ArtifactIdentity) -> weakref.ReferenceType:
    def discard_dead_owner(reference: weakref.ReferenceType) -> None:
        with native_module._NATIVE_CACHE_LOCK:
            owners = _MANAGED_ARTIFACT_OWNERS.get(key)
            if owners is None:
                return
            owners.pop(reference, None)
            if not owners:
                _MANAGED_ARTIFACT_OWNERS.pop(key, None)

    return weakref.ref(owner, discard_dead_owner)


def _retain_managed_serial_artifact(owner: object, executable: NativeExecutable) -> None:
    key = _artifact_identity(executable)
    with native_module._NATIVE_CACHE_LOCK:
        owners = _MANAGED_ARTIFACT_OWNERS.setdefault(key, {})
        reference = _owner_ref(owner, key)
        owners[reference] = owners.get(reference, 0) + 1


def _release_managed_serial_artifact(owner: object, executable: NativeExecutable) -> bool:
    key = _artifact_identity(executable)
    with native_module._NATIVE_CACHE_LOCK:
        owners = _MANAGED_ARTIFACT_OWNERS.get(key)
        if owners is not None:
            reference = weakref.ref(owner)
            retained_count = owners.get(reference, 0)
            if retained_count > 1:
                owners[reference] = retained_count - 1
                return False
            if retained_count == 1:
                owners.pop(reference)
            if owners:
                return False
            _MANAGED_ARTIFACT_OWNERS.pop(key, None)

        artifact = native_module._NATIVE_CACHE.pop(key, None)
        if artifact is None:
            return False
        artifact.close()
        return True


def _validate_cache_limit(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_cached_specializations must be a non-negative integer")
    if value < 0:
        raise ValueError("max_cached_specializations must be non-negative")


def _validate_managed_budget(budget: CompileBudget | None) -> None:
    if budget is None:
        return
    if not isinstance(budget, CompileBudget):
        raise TypeError("budget must be a CompileBudget or None")
    if budget.max_dynamic_specializations is not None:
        raise ValueError(
            "resource-managed retention cannot be combined with "
            "CompileBudget.max_dynamic_specializations in this phase"
        )
