from __future__ import annotations

import os
import shutil
import tempfile
import threading
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .compiler import LinearizationState, _shared_linearization_runtime_contract
from .fusion_planner import fuse_elementwise
from .input_validation import prepare_runtime_inputs
from .ir import Module, SymbolicDim, TensorType
from .loop_ir import lower_to_loops
from .lowering import lower_to_cpu
from .native_bundle import (
    NativeBundleError,
    NativeBundleExecutable,
    _decode_type_sequence,
    _is_sha256,
    _read_manifest,
    _sha256_file,
    compile_native_bundle,
    load_native_bundle,
)
from .native_bundle_set import (
    NativeBundleSetError,
    _concrete_input_key,
    _decode_symbols,
    _decode_template_inputs,
    _encode_template_type,
    _evaluate_template_inputs,
    _input_types,
    _normalize_loaded_bindings,
    _read_json_object,
    _runtime_input_key,
    _target_identity,
    _template_symbol_names,
    _write_json,
)
from .symbolic import (
    clone_module,
    normalize_symbolic_bindings,
    specialize_module,
    validate_dynamic_module,
)

_SCHEMA = "native-linearization-bundle-set-v1"
_MANIFEST_NAME = "manifest.json"
_VARIANTS_DIRECTORY = "variants"
_COMPONENT_ROLES = ("primal_tape", "pushforward", "pullback")


class NativeLinearizationBundleSetError(RuntimeError):
    """Raised when a retained-linearization bundle set is malformed or incomplete."""


@dataclass(frozen=True)
class _Variant:
    bindings: tuple[tuple[str, int], ...]
    component_paths: tuple[Path, Path, Path]
    input_types: tuple[TensorType, ...]
    input_key: tuple[tuple[str, tuple[int, ...]], ...]
    tape_value_count: int
    tangent_count: int


class NativeLinearizationBundleExecutable:
    """One compiler-free concrete retained-state linearization bundle."""

    def __init__(
        self,
        component_paths: tuple[Path, Path, Path],
        *,
        input_types: tuple[TensorType, ...],
        tape_value_count: int,
        tangent_count: int,
    ) -> None:
        loaded: list[NativeBundleExecutable] = []
        try:
            for path in component_paths:
                loaded.append(load_native_bundle(path))
        except Exception:
            for executable in reversed(loaded):
                executable.close()
            raise

        self._primal_tape, self._pushforward, self._pullback = loaded
        self._input_types = input_types
        self._tape_value_count = tape_value_count
        self._tangent_count = tangent_count
        self._components = tuple(loaded)
        self._finalizer = weakref.finalize(
            self,
            _close_component_executables,
            self._components,
        )

    @property
    def tape_value_count(self) -> int:
        return self._tape_value_count

    @property
    def closed(self) -> bool:
        return not self._finalizer.alive

    def close(self) -> None:
        """Unload all component libraries staged for this concrete bundle."""
        if self._finalizer.alive:
            self._finalizer()

    def linearize(self, inputs: Sequence[Any]) -> LinearizationState:
        if self.closed:
            raise RuntimeError("native linearization bundle executable is closed")
        prepared = prepare_runtime_inputs(self._input_types, inputs)
        frozen_inputs = tuple(
            np.array(value, copy=True, order="C") for value in prepared
        )
        result = self._primal_tape(inputs=frozen_inputs)

        if self._tape_value_count:
            if not isinstance(result, tuple):
                raise RuntimeError(
                    "packaged primal tape returned one value unexpectedly"
                )
            expected = self._tape_value_count + 1
            if len(result) != expected:
                raise RuntimeError(
                    "packaged primal tape returned the wrong number of values"
                )
            primal = result[0]
            tape_values = tuple(result[1:])
        else:
            if isinstance(result, tuple):
                raise RuntimeError(
                    "packaged primal tape returned unexpected extra values"
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

    def __call__(self, inputs: Sequence[Any] = ()) -> LinearizationState:
        return self.linearize(inputs)


class NativeLinearizationBundleSetExecutable:
    """Compiler-free dispatcher over finite retained-linearization bundles."""

    def __init__(
        self,
        bundle_path: Path,
        symbols: tuple[str, ...],
        variants: tuple[_Variant, ...],
    ) -> None:
        self._bundle_path = bundle_path
        self._symbols = symbols
        self._variants = variants
        self._by_binding = {
            tuple(size for _, size in variant.bindings): variant
            for variant in variants
        }
        self._by_input = {variant.input_key: variant for variant in variants}
        self._loaded: dict[tuple[int, ...], NativeLinearizationBundleExecutable] = {}
        self._lock = threading.RLock()
        self._finalizer = weakref.finalize(
            self,
            _close_loaded_linearizations,
            self._loaded,
        )

    @property
    def symbolic_dims(self) -> tuple[str, ...]:
        return self._symbols

    @property
    def available_bindings(self) -> tuple[tuple[tuple[str, int], ...], ...]:
        return tuple(variant.bindings for variant in self._variants)

    @property
    def loaded_bindings(self) -> tuple[tuple[tuple[str, int], ...], ...]:
        with self._lock:
            return tuple(
                tuple(zip(self._symbols, key, strict=True))
                for key in sorted(self._loaded)
            )

    @property
    def closed(self) -> bool:
        return not self._finalizer.alive

    def close(self) -> None:
        """Close every loaded concrete retained-linearization bundle."""
        if self._finalizer.alive:
            self._finalizer()

    def specialize(
        self,
        bindings: Mapping[SymbolicDim | str, int],
    ) -> NativeLinearizationBundleExecutable:
        normalized = _normalize_loaded_bindings(self._symbols, bindings)
        key = tuple(normalized[name] for name in self._symbols)
        try:
            variant = self._by_binding[key]
        except KeyError as error:
            raise NativeLinearizationBundleSetError(
                f"symbolic binding {dict(zip(self._symbols, key, strict=True))} is not packaged"
            ) from error
        return self._load_variant(key, variant)

    def linearize(self, inputs: Sequence[Any]) -> LinearizationState:
        if self.closed:
            raise RuntimeError("native linearization bundle set executable is closed")
        provided = tuple(inputs)
        input_key = _runtime_input_key(provided)
        try:
            variant = self._by_input[input_key]
        except KeyError as error:
            raise NativeLinearizationBundleSetError(
                "runtime input ABI does not match any packaged specialization"
            ) from error
        key = tuple(size for _, size in variant.bindings)
        return self._load_variant(key, variant).linearize(provided)

    def __call__(self, inputs: Sequence[Any] = ()) -> LinearizationState:
        return self.linearize(inputs)

    def _load_variant(
        self,
        key: tuple[int, ...],
        variant: _Variant,
    ) -> NativeLinearizationBundleExecutable:
        if self.closed:
            raise RuntimeError("native linearization bundle set executable is closed")
        with self._lock:
            executable = self._loaded.get(key)
            if executable is None:
                executable = NativeLinearizationBundleExecutable(
                    variant.component_paths,
                    input_types=variant.input_types,
                    tape_value_count=variant.tape_value_count,
                    tangent_count=variant.tangent_count,
                )
                self._loaded[key] = executable
            return executable


def compile_dynamic_linearization_bundle_set(
    module: Module,
    bindings: Sequence[Mapping[SymbolicDim | str, int]],
    destination: str | os.PathLike[str],
    compiler: str | None = None,
    *,
    output_index: int = 0,
    wrt: Sequence[int] = (0,),
) -> Path:
    """Compile finite symbolic retained linearizations into one atomic AOT package."""
    if not isinstance(module, Module):
        raise TypeError("linearization bundle sets require a tensor Module")

    template = clone_module(module)
    symbols = validate_dynamic_module(template)
    symbol_names = tuple(symbol.name for symbol in symbols)
    requested = tuple(bindings)
    if not requested:
        raise ValueError(
            "linearization bundle set requires at least one specialization"
        )

    template_inputs = _input_types(template)
    frozen_wrt = _normalize_wrt_indices(wrt, len(template_inputs))
    encoded_template_inputs = [
        _encode_template_type(type_) for type_ in template_inputs
    ]

    normalized_variants: list[
        tuple[
            tuple[int, ...],
            dict[SymbolicDim, int],
            tuple[Module, Module, Module],
            tuple[TensorType, ...],
            int,
            int,
        ]
    ] = []
    seen_bindings: set[tuple[int, ...]] = set()
    seen_input_abis: set[tuple[tuple[str, tuple[int, ...]], ...]] = set()

    for binding in requested:
        if not isinstance(binding, Mapping):
            raise TypeError(
                "each linearization bundle specialization must be a binding mapping"
            )
        normalized = normalize_symbolic_bindings(template, binding)
        binding_key = tuple(normalized[symbol] for symbol in symbols)
        if binding_key in seen_bindings:
            raise ValueError(
                "linearization bundle specializations contain a duplicate binding"
            )

        concrete = specialize_module(template, normalized)
        (
            primal_module,
            pushforward_module,
            pullback_module,
            tape_value_count,
            input_types,
            tangent_count,
        ) = _shared_linearization_runtime_contract(
            concrete,
            output_index=output_index,
            wrt=frozen_wrt,
        )
        concrete_inputs = _input_types(concrete)
        if input_types != concrete_inputs:
            raise RuntimeError(
                "internal linearization packaging error: retained input ABI drifted"
            )
        if tangent_count != len(frozen_wrt):
            raise RuntimeError(
                "internal linearization packaging error: tangent count does not match wrt"
            )

        input_key = _concrete_input_key(concrete_inputs)
        if input_key in seen_input_abis:
            raise ValueError(
                "different symbolic bindings produce the same concrete runtime input ABI"
            )

        seen_bindings.add(binding_key)
        seen_input_abis.add(input_key)
        normalized_variants.append(
            (
                binding_key,
                normalized,
                (primal_module, pushforward_module, pullback_module),
                concrete_inputs,
                tape_value_count,
                tangent_count,
            )
        )

    normalized_variants.sort(key=lambda item: item[0])
    bundle_path = Path(destination).expanduser().resolve()
    if bundle_path.exists():
        raise FileExistsError(
            f"linearization bundle set destination already exists: {bundle_path}"
        )
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    build_directory = Path(
        tempfile.mkdtemp(
            prefix=f".{bundle_path.name}.build-",
            dir=bundle_path.parent,
        )
    )
    published = False
    try:
        variants_directory = build_directory / _VARIANTS_DIRECTORY
        variants_directory.mkdir()
        manifest_variants: list[dict[str, object]] = []

        for index, (
            binding_key,
            _,
            component_modules,
            concrete_inputs,
            tape_value_count,
            tangent_count,
        ) in enumerate(normalized_variants):
            variant_root = variants_directory / f"{index:04d}"
            variant_root.mkdir()
            descriptors: dict[str, dict[str, object]] = {}
            component_abis: dict[
                str,
                tuple[tuple[TensorType, ...], tuple[TensorType, ...]],
            ] = {}

            for role, component_module in zip(
                _COMPONENT_ROLES,
                component_modules,
                strict=True,
            ):
                component_path = variant_root / role
                loops = fuse_elementwise(
                    lower_to_loops(lower_to_cpu(component_module))
                )
                compile_native_bundle(
                    loops,
                    component_path,
                    compiler=compiler,
                )
                loaded = load_native_bundle(component_path)
                try:
                    component_abis[role] = (
                        loaded.input_types,
                        loaded.output_types,
                    )
                finally:
                    loaded.close()

                manifest_path = component_path / _MANIFEST_NAME
                child_manifest = _read_manifest(manifest_path)
                abi_sha256 = child_manifest.get("abi_sha256")
                if not isinstance(abi_sha256, str) or not _is_sha256(abi_sha256):
                    raise NativeLinearizationBundleSetError(
                        f"{role} child ABI hash is malformed"
                    )
                descriptors[role] = {
                    "path": (
                        f"{_VARIANTS_DIRECTORY}/{index:04d}/{role}"
                    ),
                    "manifest_sha256": _sha256_file(manifest_path),
                    "abi_sha256": abi_sha256,
                }

            _validate_component_abi_contract(
                concrete_inputs,
                frozen_wrt,
                tape_value_count,
                component_abis,
            )

            manifest_variants.append(
                {
                    "bindings": {
                        name: size
                        for name, size in zip(
                            symbol_names,
                            binding_key,
                            strict=True,
                        )
                    },
                    "tape_value_count": tape_value_count,
                    "components": descriptors,
                }
            )

        manifest = {
            "schema": _SCHEMA,
            "target": _target_identity(),
            "symbols": list(symbol_names),
            "inputs": encoded_template_inputs,
            "wrt": list(frozen_wrt),
            "variants": manifest_variants,
        }
        _write_json(build_directory / _MANIFEST_NAME, manifest)
        if bundle_path.exists():
            raise FileExistsError(
                f"linearization bundle set destination already exists: {bundle_path}"
            )
        os.replace(build_directory, bundle_path)
        published = True
        return bundle_path
    finally:
        if not published:
            shutil.rmtree(build_directory, ignore_errors=True)


def load_dynamic_linearization_bundle_set(
    bundle: str | os.PathLike[str],
) -> NativeLinearizationBundleSetExecutable:
    """Verify and load finite retained-linearization bundles without a compiler."""
    bundle_path = Path(bundle).expanduser().resolve()
    try:
        manifest = _read_json_object(
            bundle_path / _MANIFEST_NAME,
            label="linearization bundle-set manifest",
        )
    except NativeBundleSetError as exc:
        raise NativeLinearizationBundleSetError(str(exc)) from exc

    expected_keys = {
        "schema",
        "target",
        "symbols",
        "inputs",
        "wrt",
        "variants",
    }
    if set(manifest) != expected_keys:
        raise NativeLinearizationBundleSetError(
            "linearization bundle-set manifest has an unsupported field set"
        )
    if manifest["schema"] != _SCHEMA:
        raise NativeLinearizationBundleSetError(
            "linearization bundle-set schema is not supported"
        )
    if manifest["target"] != _target_identity():
        raise NativeLinearizationBundleSetError(
            "linearization bundle-set target does not match this process"
        )

    try:
        symbols = _decode_symbols(manifest["symbols"])
        template_inputs = _decode_template_inputs(
            manifest["inputs"],
            symbols,
        )
    except NativeBundleSetError as exc:
        raise NativeLinearizationBundleSetError(str(exc)) from exc
    if _template_symbol_names(template_inputs) != frozenset(symbols):
        raise NativeLinearizationBundleSetError(
            "linearization bundle-set input template does not reference every declared symbol"
        )

    wrt = _decode_wrt(manifest["wrt"], len(template_inputs))
    raw_variants = manifest["variants"]
    if not isinstance(raw_variants, list) or not raw_variants:
        raise NativeLinearizationBundleSetError(
            "linearization bundle set must contain at least one variant"
        )

    variants: list[_Variant] = []
    seen_bindings: set[tuple[int, ...]] = set()
    seen_input_abis: set[tuple[tuple[str, tuple[int, ...]], ...]] = set()
    previous_binding: tuple[int, ...] | None = None

    for index, raw_variant in enumerate(raw_variants):
        if not isinstance(raw_variant, dict) or set(raw_variant) != {
            "bindings",
            "tape_value_count",
            "components",
        }:
            raise NativeLinearizationBundleSetError(
                f"linearization bundle variant {index} is malformed"
            )

        try:
            normalized = _normalize_loaded_bindings(
                symbols,
                raw_variant["bindings"],
            )
        except (TypeError, NativeBundleSetError) as exc:
            raise NativeLinearizationBundleSetError(str(exc)) from exc
        binding_key = tuple(normalized[name] for name in symbols)
        if binding_key in seen_bindings:
            raise NativeLinearizationBundleSetError(
                "linearization bundle set contains duplicate bindings"
            )
        if previous_binding is not None and binding_key <= previous_binding:
            raise NativeLinearizationBundleSetError(
                "linearization bundle variants are not canonically ordered"
            )
        previous_binding = binding_key
        seen_bindings.add(binding_key)

        tape_value_count = raw_variant["tape_value_count"]
        if (
            isinstance(tape_value_count, bool)
            or not isinstance(tape_value_count, int)
            or tape_value_count < 0
        ):
            raise NativeLinearizationBundleSetError(
                "linearization bundle tape value count is malformed"
            )

        expected_inputs = _evaluate_template_inputs(
            template_inputs,
            normalized,
        )
        input_key = _concrete_input_key(expected_inputs)
        if input_key in seen_input_abis:
            raise NativeLinearizationBundleSetError(
                "linearization bundle set contains ambiguous duplicate concrete input ABIs"
            )
        seen_input_abis.add(input_key)

        component_paths, component_abis = _decode_components(
            bundle_path,
            index,
            raw_variant["components"],
        )
        _validate_component_abi_contract(
            expected_inputs,
            wrt,
            tape_value_count,
            component_abis,
        )

        variants.append(
            _Variant(
                bindings=tuple(zip(symbols, binding_key, strict=True)),
                component_paths=component_paths,
                input_types=expected_inputs,
                input_key=input_key,
                tape_value_count=tape_value_count,
                tangent_count=len(wrt),
            )
        )

    return NativeLinearizationBundleSetExecutable(
        bundle_path,
        symbols,
        tuple(variants),
    )


def _decode_components(
    bundle_path: Path,
    index: int,
    value: object,
) -> tuple[
    tuple[Path, Path, Path],
    dict[str, tuple[tuple[TensorType, ...], tuple[TensorType, ...]]],
]:
    if not isinstance(value, dict) or set(value) != set(_COMPONENT_ROLES):
        raise NativeLinearizationBundleSetError(
            "linearization bundle components are malformed"
        )

    seen_paths: set[str] = set()
    paths: list[Path] = []
    abis: dict[str, tuple[tuple[TensorType, ...], tuple[TensorType, ...]]] = {}

    for role in _COMPONENT_ROLES:
        descriptor = value[role]
        if not isinstance(descriptor, dict) or set(descriptor) != {
            "path",
            "manifest_sha256",
            "abi_sha256",
        }:
            raise NativeLinearizationBundleSetError(
                f"linearization {role} component descriptor is malformed"
            )
        relative_path = descriptor["path"]
        expected_relative_path = (
            f"{_VARIANTS_DIRECTORY}/{index:04d}/{role}"
        )
        if relative_path != expected_relative_path:
            raise NativeLinearizationBundleSetError(
                f"linearization {role} component path is not canonical"
            )
        if relative_path in seen_paths:
            raise NativeLinearizationBundleSetError(
                "linearization component paths are not unique"
            )
        seen_paths.add(relative_path)

        component_path = bundle_path / relative_path
        manifest_path = component_path / _MANIFEST_NAME
        expected_manifest_hash = descriptor["manifest_sha256"]
        if (
            not isinstance(expected_manifest_hash, str)
            or not _is_sha256(expected_manifest_hash)
        ):
            raise NativeLinearizationBundleSetError(
                f"linearization {role} manifest hash is malformed"
            )
        if (
            not manifest_path.is_file()
            or _sha256_file(manifest_path) != expected_manifest_hash
        ):
            raise NativeLinearizationBundleSetError(
                f"linearization {role} manifest hash does not match"
            )

        try:
            child_manifest = _read_manifest(manifest_path)
            inputs = _decode_type_sequence(
                child_manifest.get("inputs"),
                label=f"{role} input ABI",
            )
            outputs = _decode_type_sequence(
                child_manifest.get("outputs"),
                label=f"{role} output ABI",
            )
        except NativeBundleError as exc:
            raise NativeLinearizationBundleSetError(
                f"linearization {role} child manifest is invalid"
            ) from exc

        child_abi = child_manifest.get("abi_sha256")
        expected_abi = descriptor["abi_sha256"]
        if (
            not isinstance(expected_abi, str)
            or not _is_sha256(expected_abi)
            or child_abi != expected_abi
        ):
            raise NativeLinearizationBundleSetError(
                f"linearization {role} ABI hash does not match child manifest"
            )

        paths.append(component_path)
        abis[role] = (inputs, outputs)

    return (paths[0], paths[1], paths[2]), abis


def _validate_component_abi_contract(
    forward_inputs: tuple[TensorType, ...],
    wrt: tuple[int, ...],
    tape_value_count: int,
    components: Mapping[
        str,
        tuple[tuple[TensorType, ...], tuple[TensorType, ...]],
    ],
) -> None:
    try:
        primal_inputs, primal_outputs = components["primal_tape"]
        push_inputs, push_outputs = components["pushforward"]
        pull_inputs, pull_outputs = components["pullback"]
    except KeyError as exc:
        raise NativeLinearizationBundleSetError(
            "linearization component ABI contract is incomplete"
        ) from exc

    if primal_inputs != forward_inputs:
        raise NativeLinearizationBundleSetError(
            "linearization component ABI contract has mismatched primal inputs"
        )
    if len(primal_outputs) != tape_value_count + 1:
        raise NativeLinearizationBundleSetError(
            "linearization component ABI contract has mismatched tape outputs"
        )

    selected_output = primal_outputs[0]
    tape_types = primal_outputs[1:]
    tangent_types = tuple(forward_inputs[index] for index in wrt)
    retained_prefix = forward_inputs + tape_types

    if push_inputs != retained_prefix + tangent_types:
        raise NativeLinearizationBundleSetError(
            "linearization component ABI contract has mismatched pushforward inputs"
        )
    if push_outputs != (selected_output,):
        raise NativeLinearizationBundleSetError(
            "linearization component ABI contract has mismatched pushforward output"
        )
    if pull_inputs != retained_prefix + (selected_output,):
        raise NativeLinearizationBundleSetError(
            "linearization component ABI contract has mismatched pullback inputs"
        )
    if pull_outputs != tangent_types:
        raise NativeLinearizationBundleSetError(
            "linearization component ABI contract has mismatched pullback outputs"
        )


def _normalize_wrt_indices(
    wrt: Sequence[int],
    input_count: int,
) -> tuple[int, ...]:
    if isinstance(wrt, (str, bytes)):
        raise TypeError("wrt must be a sequence of runtime input indices")
    try:
        frozen = tuple(wrt)
    except TypeError as exc:
        raise TypeError("wrt must be a sequence of runtime input indices") from exc
    if not frozen:
        raise ValueError("linearization bundle requires at least one wrt input")
    seen: set[int] = set()
    for index in frozen:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("wrt entries must be integer runtime input indices")
        if index < 0 or index >= input_count:
            raise ValueError(f"wrt input index {index} is out of range")
        if index in seen:
            raise ValueError(f"wrt input index {index} is duplicated")
        seen.add(index)
    return frozen


def _decode_wrt(value: object, input_count: int) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise NativeLinearizationBundleSetError(
            "linearization bundle wrt contract must be a list"
        )
    try:
        return _normalize_wrt_indices(tuple(value), input_count)
    except (TypeError, ValueError) as exc:
        raise NativeLinearizationBundleSetError(str(exc)) from exc


def _close_component_executables(
    executables: tuple[NativeBundleExecutable, ...],
) -> None:
    for executable in executables:
        executable.close()


def _close_loaded_linearizations(
    loaded: dict[tuple[int, ...], NativeLinearizationBundleExecutable],
) -> None:
    for executable in tuple(loaded.values()):
        executable.close()
    loaded.clear()


__all__ = [
    "NativeLinearizationBundleExecutable",
    "NativeLinearizationBundleSetError",
    "NativeLinearizationBundleSetExecutable",
    "compile_dynamic_linearization_bundle_set",
    "load_dynamic_linearization_bundle_set",
]
