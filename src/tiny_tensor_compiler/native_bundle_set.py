from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import threading
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .fusion_planner import fuse_elementwise
from .ir import AffineDim, DType, LinearDim, Module, SymbolicDim, TensorType
from .loop_ir import lower_to_loops
from .lowering import lower_to_cpu
from .native_bundle import (
    NativeBundleExecutable,
    compile_native_bundle,
    load_native_bundle,
)
from .symbolic import (
    clone_module,
    normalize_symbolic_bindings,
    specialize_module,
    validate_dynamic_module,
)

_BUNDLE_SET_SCHEMA = "native-bundle-set-v1"
_MANIFEST_NAME = "manifest.json"
_VARIANTS_DIRECTORY = "variants"
_CHILD_MANIFEST_NAME = "manifest.json"


class NativeBundleSetError(RuntimeError):
    """Raised when a finite dynamic native bundle set is malformed or incomplete."""


@dataclass(frozen=True)
class _Variant:
    bindings: tuple[tuple[str, int], ...]
    path: Path
    input_key: tuple[tuple[str, tuple[int, ...]], ...]


class NativeBundleSetExecutable:
    """Compiler-free dispatcher over a finite set of verified concrete AOT bundles."""

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
            tuple(size for _, size in variant.bindings): variant for variant in variants
        }
        self._by_input = {variant.input_key: variant for variant in variants}
        self._loaded: dict[tuple[int, ...], NativeBundleExecutable] = {}
        self._lock = threading.RLock()
        self._finalizer = weakref.finalize(self, _close_loaded_executables, self._loaded)

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
                tuple(zip(self._symbols, key, strict=True)) for key in sorted(self._loaded)
            )

    @property
    def closed(self) -> bool:
        return not self._finalizer.alive

    def close(self) -> None:
        """Unload every child bundle loaded by this dispatcher."""
        if self._finalizer.alive:
            self._finalizer()

    def specialize(
        self,
        bindings: Mapping[SymbolicDim | str, int],
    ) -> NativeBundleExecutable:
        """Return one packaged concrete specialization without invoking a compiler."""
        normalized = _normalize_loaded_bindings(self._symbols, bindings)
        key = tuple(normalized[name] for name in self._symbols)
        try:
            variant = self._by_binding[key]
        except KeyError as error:
            raise NativeBundleSetError(
                f"symbolic binding {dict(zip(self._symbols, key, strict=True))} is not packaged"
            ) from error
        return self._load_variant(key, variant)

    def execute(
        self,
        inputs: Sequence[Any] = (),
        out: Any = None,
    ):
        """Dispatch runtime inputs to the unique packaged concrete input ABI."""
        if self.closed:
            raise RuntimeError("native bundle set executable is closed")
        input_key = _runtime_input_key(inputs)
        try:
            variant = self._by_input[input_key]
        except KeyError as error:
            raise NativeBundleSetError(
                "runtime input ABI does not match any packaged specialization"
            ) from error
        key = tuple(size for _, size in variant.bindings)
        return self._load_variant(key, variant)(inputs=inputs, out=out)

    def __call__(
        self,
        inputs: Sequence[Any] = (),
        out: Any = None,
    ):
        return self.execute(inputs=inputs, out=out)

    def _load_variant(
        self,
        key: tuple[int, ...],
        variant: _Variant,
    ) -> NativeBundleExecutable:
        if self.closed:
            raise RuntimeError("native bundle set executable is closed")
        with self._lock:
            executable = self._loaded.get(key)
            if executable is None:
                executable = load_native_bundle(variant.path)
                self._loaded[key] = executable
            return executable


def compile_dynamic_bundle_set(
    module: Module,
    bindings: Sequence[Mapping[SymbolicDim | str, int]],
    destination: str | os.PathLike[str],
    compiler: str | None = None,
) -> Path:
    """Compile a finite family of symbolic specializations into one atomic AOT package."""
    if not isinstance(module, Module):
        raise TypeError("dynamic native bundle sets require a tensor Module")

    template = clone_module(module)
    symbols = validate_dynamic_module(template)
    symbol_names = tuple(symbol.name for symbol in symbols)
    requested = tuple(bindings)
    if not requested:
        raise ValueError("dynamic native bundle set requires at least one specialization")

    template_inputs = _input_types(template)
    encoded_template_inputs = [_encode_template_type(type_) for type_ in template_inputs]
    normalized_variants: list[
        tuple[tuple[int, ...], dict[SymbolicDim, int], Module, tuple[TensorType, ...]]
    ] = []
    seen_bindings: set[tuple[int, ...]] = set()
    seen_input_abis: set[tuple[tuple[str, tuple[int, ...]], ...]] = set()

    for binding in requested:
        if not isinstance(binding, Mapping):
            raise TypeError("each dynamic bundle specialization must be a binding mapping")
        normalized = normalize_symbolic_bindings(template, binding)
        binding_key = tuple(normalized[symbol] for symbol in symbols)
        if binding_key in seen_bindings:
            raise ValueError("dynamic bundle specializations contain a duplicate binding")
        concrete = specialize_module(template, normalized)
        concrete_inputs = _input_types(concrete)
        input_key = _concrete_input_key(concrete_inputs)
        if input_key in seen_input_abis:
            raise ValueError(
                "different symbolic bindings produce the same concrete runtime input ABI"
            )
        seen_bindings.add(binding_key)
        seen_input_abis.add(input_key)
        normalized_variants.append((binding_key, normalized, concrete, concrete_inputs))

    normalized_variants.sort(key=lambda item: item[0])
    bundle_path = Path(destination).expanduser().resolve()
    if bundle_path.exists():
        raise FileExistsError(f"dynamic bundle set destination already exists: {bundle_path}")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    build_directory = Path(
        tempfile.mkdtemp(prefix=f".{bundle_path.name}.build-", dir=bundle_path.parent)
    )
    published = False
    try:
        variants_directory = build_directory / _VARIANTS_DIRECTORY
        variants_directory.mkdir()
        manifest_variants: list[dict[str, object]] = []

        for index, (binding_key, _, concrete, concrete_inputs) in enumerate(normalized_variants):
            relative_path = f"{_VARIANTS_DIRECTORY}/{index:04d}"
            child_path = build_directory / relative_path
            loops = fuse_elementwise(lower_to_loops(lower_to_cpu(concrete)))
            compile_native_bundle(loops, child_path, compiler=compiler)

            child_manifest_path = child_path / _CHILD_MANIFEST_NAME
            child_manifest = _read_json_object(
                child_manifest_path,
                label=f"child bundle {index} manifest",
            )
            child_inputs = _decode_child_inputs(child_manifest.get("inputs"))
            if child_inputs != concrete_inputs:
                raise NativeBundleSetError(
                    f"child bundle {index} input ABI does not match its specialization"
                )
            abi_sha256 = child_manifest.get("abi_sha256")
            if not isinstance(abi_sha256, str) or not _is_sha256(abi_sha256):
                raise NativeBundleSetError(f"child bundle {index} ABI hash is malformed")

            manifest_variants.append(
                {
                    "bindings": {
                        name: size
                        for name, size in zip(symbol_names, binding_key, strict=True)
                    },
                    "path": relative_path,
                    "manifest_sha256": _sha256_file(child_manifest_path),
                    "abi_sha256": abi_sha256,
                }
            )

        manifest = {
            "schema": _BUNDLE_SET_SCHEMA,
            "target": _target_identity(),
            "symbols": list(symbol_names),
            "inputs": encoded_template_inputs,
            "variants": manifest_variants,
        }
        _write_json(build_directory / _MANIFEST_NAME, manifest)
        if bundle_path.exists():
            raise FileExistsError(f"dynamic bundle set destination already exists: {bundle_path}")
        os.replace(build_directory, bundle_path)
        published = True
        return bundle_path
    finally:
        if not published:
            shutil.rmtree(build_directory, ignore_errors=True)


def load_dynamic_bundle_set(
    bundle: str | os.PathLike[str],
) -> NativeBundleSetExecutable:
    """Verify and load a finite AOT specialization family without a compiler or IR."""
    bundle_path = Path(bundle).expanduser().resolve()
    manifest = _read_json_object(bundle_path / _MANIFEST_NAME, label="bundle-set manifest")
    expected_keys = {"schema", "target", "symbols", "inputs", "variants"}
    if set(manifest) != expected_keys:
        raise NativeBundleSetError("dynamic bundle set manifest has an unsupported field set")
    if manifest["schema"] != _BUNDLE_SET_SCHEMA:
        raise NativeBundleSetError("dynamic bundle set schema is not supported")
    if manifest["target"] != _target_identity():
        raise NativeBundleSetError("dynamic bundle set target does not match this process")

    symbols = _decode_symbols(manifest["symbols"])
    template_inputs = _decode_template_inputs(manifest["inputs"], symbols)
    raw_variants = manifest["variants"]
    if not isinstance(raw_variants, list) or not raw_variants:
        raise NativeBundleSetError("dynamic bundle set must contain at least one variant")

    variants: list[_Variant] = []
    seen_bindings: set[tuple[int, ...]] = set()
    seen_input_abis: set[tuple[tuple[str, tuple[int, ...]], ...]] = set()
    previous_binding: tuple[int, ...] | None = None

    for index, raw_variant in enumerate(raw_variants):
        if not isinstance(raw_variant, dict) or set(raw_variant) != {
            "bindings",
            "path",
            "manifest_sha256",
            "abi_sha256",
        }:
            raise NativeBundleSetError(f"dynamic bundle set variant {index} is malformed")
        normalized = _normalize_loaded_bindings(symbols, raw_variant["bindings"])
        binding_key = tuple(normalized[name] for name in symbols)
        if binding_key in seen_bindings:
            raise NativeBundleSetError("dynamic bundle set contains duplicate bindings")
        if previous_binding is not None and binding_key <= previous_binding:
            raise NativeBundleSetError("dynamic bundle set variants are not canonically ordered")
        previous_binding = binding_key
        seen_bindings.add(binding_key)

        expected_relative_path = f"{_VARIANTS_DIRECTORY}/{index:04d}"
        if raw_variant["path"] != expected_relative_path:
            raise NativeBundleSetError("dynamic bundle set variant path is not canonical")
        variant_path = bundle_path / expected_relative_path
        child_manifest_path = variant_path / _CHILD_MANIFEST_NAME
        expected_manifest_hash = raw_variant["manifest_sha256"]
        if not isinstance(expected_manifest_hash, str) or not _is_sha256(expected_manifest_hash):
            raise NativeBundleSetError("dynamic bundle set child manifest hash is malformed")
        if not child_manifest_path.is_file() or _sha256_file(child_manifest_path) != expected_manifest_hash:
            raise NativeBundleSetError("dynamic bundle set child manifest hash does not match")

        child_manifest = _read_json_object(
            child_manifest_path,
            label=f"child bundle {index} manifest",
        )
        child_inputs = _decode_child_inputs(child_manifest.get("inputs"))
        expected_inputs = _evaluate_template_inputs(template_inputs, normalized)
        if child_inputs != expected_inputs:
            raise NativeBundleSetError(
                f"dynamic bundle set variant {index} binding does not match child input ABI"
            )
        child_abi_sha256 = child_manifest.get("abi_sha256")
        expected_abi_sha256 = raw_variant["abi_sha256"]
        if (
            not isinstance(expected_abi_sha256, str)
            or not _is_sha256(expected_abi_sha256)
            or child_abi_sha256 != expected_abi_sha256
        ):
            raise NativeBundleSetError(
                f"dynamic bundle set variant {index} ABI hash does not match child manifest"
            )

        input_key = _concrete_input_key(child_inputs)
        if input_key in seen_input_abis:
            raise NativeBundleSetError(
                "dynamic bundle set contains ambiguous duplicate concrete input ABIs"
            )
        seen_input_abis.add(input_key)
        variants.append(
            _Variant(
                bindings=tuple(zip(symbols, binding_key, strict=True)),
                path=variant_path,
                input_key=input_key,
            )
        )

    return NativeBundleSetExecutable(bundle_path, symbols, tuple(variants))


def _input_types(module: Module) -> tuple[TensorType, ...]:
    return tuple(
        op.results[0].type for op in module.function.ops if op.opcode == "input"
    )


def _encode_template_type(type_: TensorType) -> dict[str, object]:
    return {
        "dtype": type_.dtype.value,
        "shape": [_encode_template_dim(dim) for dim in type_.shape],
    }


def _encode_template_dim(dim: int | SymbolicDim | AffineDim | LinearDim) -> object:
    if isinstance(dim, int):
        return {"kind": "static", "value": dim}
    if isinstance(dim, SymbolicDim):
        return {"kind": "symbol", "name": dim.name}
    if isinstance(dim, AffineDim):
        return {
            "kind": "affine",
            "symbol": dim.symbol.name,
            "scale": dim.scale,
            "offset": dim.offset,
        }
    return {
        "kind": "linear",
        "terms": [[symbol.name, coefficient] for symbol, coefficient in dim.terms],
        "offset": dim.offset,
    }


def _decode_symbols(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise NativeBundleSetError("dynamic bundle set symbols must be a non-empty list")
    if any(not isinstance(name, str) or not name.isidentifier() for name in value):
        raise NativeBundleSetError("dynamic bundle set contains an invalid symbol name")
    symbols = tuple(value)
    if symbols != tuple(sorted(set(symbols))):
        raise NativeBundleSetError("dynamic bundle set symbols are not unique and sorted")
    return symbols


def _decode_template_inputs(
    value: object,
    symbols: tuple[str, ...],
) -> tuple[tuple[DType, tuple[object, ...]], ...]:
    if not isinstance(value, list):
        raise NativeBundleSetError("dynamic bundle set input template must be a list")
    return tuple(_decode_template_type(item, symbols) for item in value)


def _decode_template_type(
    value: object,
    symbols: tuple[str, ...],
) -> tuple[DType, tuple[object, ...]]:
    if not isinstance(value, dict) or set(value) != {"dtype", "shape"}:
        raise NativeBundleSetError("dynamic bundle set input template entry is malformed")
    dtype_value = value["dtype"]
    if not isinstance(dtype_value, str):
        raise NativeBundleSetError("dynamic bundle set input template dtype is malformed")
    try:
        dtype = DType(dtype_value)
    except ValueError as error:
        raise NativeBundleSetError("dynamic bundle set input template dtype is unsupported") from error
    shape = value["shape"]
    if not isinstance(shape, list):
        raise NativeBundleSetError("dynamic bundle set input template shape is malformed")
    return dtype, tuple(_decode_template_dim(dim, symbols) for dim in shape)


def _decode_template_dim(value: object, symbols: tuple[str, ...]) -> object:
    if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
        raise NativeBundleSetError("dynamic bundle set input template dimension is malformed")
    kind = value["kind"]
    if kind == "static":
        if set(value) != {"kind", "value"} or not _valid_extent(value["value"]):
            raise NativeBundleSetError("dynamic bundle set static dimension is malformed")
        return ("static", value["value"])
    if kind == "symbol":
        if set(value) != {"kind", "name"} or value["name"] not in symbols:
            raise NativeBundleSetError("dynamic bundle set symbolic dimension is malformed")
        return ("symbol", value["name"])
    if kind == "affine":
        if set(value) != {"kind", "symbol", "scale", "offset"}:
            raise NativeBundleSetError("dynamic bundle set affine dimension is malformed")
        symbol = value["symbol"]
        scale = value["scale"]
        offset = value["offset"]
        if symbol not in symbols or not _positive_int(scale) or not _valid_extent(offset):
            raise NativeBundleSetError("dynamic bundle set affine dimension is malformed")
        return ("affine", symbol, scale, offset)
    if kind == "linear":
        if set(value) != {"kind", "terms", "offset"}:
            raise NativeBundleSetError("dynamic bundle set linear dimension is malformed")
        terms = value["terms"]
        offset = value["offset"]
        if not isinstance(terms, list) or len(terms) < 2 or not _valid_extent(offset):
            raise NativeBundleSetError("dynamic bundle set linear dimension is malformed")
        decoded_terms: list[tuple[str, int]] = []
        for term in terms:
            if (
                not isinstance(term, list)
                or len(term) != 2
                or term[0] not in symbols
                or not _positive_int(term[1])
            ):
                raise NativeBundleSetError("dynamic bundle set linear dimension is malformed")
            decoded_terms.append((term[0], term[1]))
        if tuple(decoded_terms) != tuple(sorted(set(decoded_terms))):
            raise NativeBundleSetError("dynamic bundle set linear terms are not canonical")
        return ("linear", tuple(decoded_terms), offset)
    raise NativeBundleSetError("dynamic bundle set input template dimension kind is unsupported")


def _evaluate_template_inputs(
    template_inputs: tuple[tuple[DType, tuple[object, ...]], ...],
    bindings: Mapping[str, int],
) -> tuple[TensorType, ...]:
    return tuple(
        TensorType(
            tuple(_evaluate_template_dim(dim, bindings) for dim in shape),
            dtype,
        )
        for dtype, shape in template_inputs
    )


def _evaluate_template_dim(value: object, bindings: Mapping[str, int]) -> int:
    if not isinstance(value, tuple) or not value:
        raise NativeBundleSetError("internal malformed dynamic bundle template dimension")
    kind = value[0]
    if kind == "static":
        return value[1]
    if kind == "symbol":
        return bindings[value[1]]
    if kind == "affine":
        return value[2] * bindings[value[1]] + value[3]
    if kind == "linear":
        return value[2] + sum(
            coefficient * bindings[name] for name, coefficient in value[1]
        )
    raise NativeBundleSetError("internal unsupported dynamic bundle template dimension")


def _normalize_loaded_bindings(
    symbols: tuple[str, ...],
    value: object,
) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError("dynamic bundle specialization requires a binding mapping")
    normalized: dict[str, int] = {}
    symbol_set = set(symbols)
    for key, size in value.items():
        if isinstance(key, SymbolicDim):
            name = key.name
        elif isinstance(key, str):
            name = key
        else:
            raise TypeError("dynamic bundle binding keys must be SymbolicDim or str")
        if name not in symbol_set:
            raise NativeBundleSetError(f"binding references unknown symbolic dimension {name!r}")
        if not _valid_extent(size):
            raise NativeBundleSetError(
                f"symbolic dimension {name} requires a non-negative integer binding"
            )
        previous = normalized.get(name)
        if previous is not None and previous != size:
            raise NativeBundleSetError(f"symbolic dimension {name} has conflicting bindings")
        normalized[name] = size
    if set(normalized) != symbol_set:
        missing = ", ".join(sorted(symbol_set - normalized.keys()))
        extra = ", ".join(sorted(normalized.keys() - symbol_set))
        detail = missing if missing else extra
        raise NativeBundleSetError(f"dynamic bundle bindings do not match symbols: {detail}")
    return {name: normalized[name] for name in symbols}


def _decode_child_inputs(value: object) -> tuple[TensorType, ...]:
    if not isinstance(value, list):
        raise NativeBundleSetError("child bundle input ABI must be a list")
    decoded: list[TensorType] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"dtype", "shape"}:
            raise NativeBundleSetError("child bundle input ABI entry is malformed")
        dtype_value = item["dtype"]
        shape_value = item["shape"]
        if not isinstance(dtype_value, str):
            raise NativeBundleSetError("child bundle input ABI dtype is malformed")
        try:
            dtype = DType(dtype_value)
        except ValueError as error:
            raise NativeBundleSetError("child bundle input ABI dtype is unsupported") from error
        if not isinstance(shape_value, list) or any(not _valid_extent(dim) for dim in shape_value):
            raise NativeBundleSetError("child bundle input ABI shape is malformed")
        decoded.append(TensorType(tuple(shape_value), dtype))
    return tuple(decoded)


def _concrete_input_key(
    input_types: Sequence[TensorType],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    key: list[tuple[str, tuple[int, ...]]] = []
    for type_ in input_types:
        if not type_.is_static:
            raise NativeBundleSetError("dynamic bundle child input ABI is not concrete")
        key.append((type_.dtype.value, tuple(int(dim) for dim in type_.shape)))
    return tuple(key)


def _runtime_input_key(
    inputs: Sequence[Any],
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    key: list[tuple[str, tuple[int, ...]]] = []
    for index, value in enumerate(inputs):
        array = np.asarray(value)
        try:
            dtype = DType.from_numpy(array.dtype)
        except TypeError as error:
            raise NativeBundleSetError(
                f"runtime input {index} dtype {array.dtype} is not supported"
            ) from error
        key.append((dtype.value, tuple(array.shape)))
    return tuple(key)


def _target_identity() -> dict[str, object]:
    return {
        "os_name": os.name,
        "sys_platform": sys.platform,
        "machine": platform.machine(),
        "pointer_bits": ctypes.sizeof(ctypes.c_void_p) * 8,
    }


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NativeBundleSetError(f"failed to read {label}: {error}") from error
    if not isinstance(value, dict):
        raise NativeBundleSetError(f"{label} must be a JSON object")
    return value


def _write_json(path: Path, value: dict[str, object]) -> None:
    try:
        path.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise NativeBundleSetError(f"failed to write dynamic bundle manifest: {error}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise NativeBundleSetError(f"failed to hash bundle file: {error}") from error
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _valid_extent(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _close_loaded_executables(
    loaded: dict[tuple[int, ...], NativeBundleExecutable],
) -> None:
    for executable in tuple(loaded.values()):
        executable.close()
    loaded.clear()
