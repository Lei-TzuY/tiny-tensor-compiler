from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import weakref
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .c_abi_codegen import generate_c
from .input_validation import prepare_runtime_inputs
from .ir import DType, TensorType
from .loop_ir import LoopProgram
from .native import (
    ExecutionResult,
    NativeCompilationError,
    NativeOutput,
    _NativeArtifact,
    _compile_source,
    _compiler_command,
    _library_name,
    _load_library,
    _pointer_type,
)

_BUNDLE_SCHEMA = "native-bundle-v1"
_MANIFEST_NAME = "manifest.json"
_SOURCE_NAME = "program.c"


class NativeBundleError(RuntimeError):
    """Raised when a native AOT bundle is malformed, incompatible, or corrupt."""


class NativeBundleExecutable:
    """Reusable executable loaded from a verified native AOT bundle."""

    def __init__(
        self,
        artifact: _NativeArtifact,
        input_types: tuple[TensorType, ...],
        output_types: tuple[TensorType, ...],
    ) -> None:
        self._artifact = artifact
        self._input_types = input_types
        self._output_types = output_types
        self._finalizer = weakref.finalize(self, artifact.close)

    @property
    def input_types(self) -> tuple[TensorType, ...]:
        return self._input_types

    @property
    def output_types(self) -> tuple[TensorType, ...]:
        return self._output_types

    @property
    def closed(self) -> bool:
        return not self._finalizer.alive

    def close(self) -> None:
        """Unload the staged shared library and release process-owned staging files."""
        if self._finalizer.alive:
            self._finalizer()

    def execute(
        self,
        inputs: Sequence[Any] = (),
        out: NativeOutput = None,
    ) -> ExecutionResult:
        if self.closed:
            raise RuntimeError("native bundle executable is closed")
        runtime_inputs = prepare_runtime_inputs(self._input_types, inputs)
        outputs = _prepare_outputs(self._output_types, out, runtime_inputs)
        runner = self._artifact.library.tiny_tensor_run
        output_pointer_types = tuple(
            _pointer_type(output_type.dtype.to_numpy()) for output_type in self._output_types
        )
        input_pointer_types = tuple(
            _pointer_type(input_type.dtype.to_numpy()) for input_type in self._input_types
        )
        runner.argtypes = [*output_pointer_types, *input_pointer_types]
        runner.restype = None
        arguments = [
            array.ctypes.data_as(pointer_type)
            for array, pointer_type in zip(outputs, output_pointer_types, strict=True)
        ]
        arguments.extend(
            array.ctypes.data_as(pointer_type)
            for array, pointer_type in zip(
                runtime_inputs,
                input_pointer_types,
                strict=True,
            )
        )
        runner(*arguments)
        return outputs[0] if len(outputs) == 1 else outputs

    def __call__(
        self,
        inputs: Sequence[Any] = (),
        out: NativeOutput = None,
    ) -> ExecutionResult:
        return self.execute(inputs, out=out)

    def __enter__(self) -> NativeBundleExecutable:
        if self.closed:
            raise RuntimeError("native bundle executable is closed")
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()


def compile_native_bundle(
    program: LoopProgram,
    destination: str | os.PathLike[str],
    compiler: str | None = None,
) -> Path:
    """Compile one concrete LoopProgram into an atomically published AOT bundle."""
    if not isinstance(program, LoopProgram):
        raise TypeError("native bundle export requires a concrete LoopProgram")

    bundle_path = Path(destination).expanduser().resolve()
    if bundle_path.exists():
        raise FileExistsError(f"native bundle destination already exists: {bundle_path}")
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    command = _compiler_command(compiler)
    source = generate_c(program)
    input_types = tuple(program.input_types)
    output_types = _return_types(program)
    _require_static_types((*input_types, *output_types))

    build_directory = Path(
        tempfile.mkdtemp(prefix=f".{bundle_path.name}.build-", dir=bundle_path.parent)
    )
    published = False
    try:
        library_path = _compile_source(source, command, build_directory)
        source_path = build_directory / _SOURCE_NAME
        if source_path.read_text(encoding="utf-8") != source:
            raise NativeBundleError("compiled bundle source does not match generated source")

        manifest = {
            "schema": _BUNDLE_SCHEMA,
            "target": _target_identity(),
            "source": _SOURCE_NAME,
            "source_sha256": _sha256_file(source_path),
            "library": library_path.name,
            "library_sha256": _sha256_file(library_path),
            "inputs": [_encode_tensor_type(type_) for type_ in input_types],
            "outputs": [_encode_tensor_type(type_) for type_ in output_types],
        }
        _write_manifest(build_directory / _MANIFEST_NAME, manifest)

        if bundle_path.exists():
            raise FileExistsError(f"native bundle destination already exists: {bundle_path}")
        os.replace(build_directory, bundle_path)
        published = True
        return bundle_path
    finally:
        if not published:
            shutil.rmtree(build_directory, ignore_errors=True)


def load_native_bundle(
    bundle: str | os.PathLike[str],
) -> NativeBundleExecutable:
    """Verify and load an AOT bundle without requiring a compiler or LoopProgram."""
    bundle_path = Path(bundle).expanduser().resolve()
    manifest = _read_manifest(bundle_path / _MANIFEST_NAME)
    expected_keys = {
        "schema",
        "target",
        "source",
        "source_sha256",
        "library",
        "library_sha256",
        "inputs",
        "outputs",
    }
    if set(manifest) != expected_keys:
        raise NativeBundleError("native bundle manifest has an unsupported field set")
    if manifest["schema"] != _BUNDLE_SCHEMA:
        raise NativeBundleError("native bundle schema is not supported")
    if manifest["target"] != _target_identity():
        raise NativeBundleError("native bundle target does not match this process")
    if manifest["source"] != _SOURCE_NAME:
        raise NativeBundleError("native bundle source path is invalid")
    if manifest["library"] != _library_name():
        raise NativeBundleError("native bundle library name does not match this target")

    source_path = bundle_path / _SOURCE_NAME
    library_path = bundle_path / _library_name()
    _verify_hashed_file(source_path, manifest["source_sha256"], label="source")
    _verify_hashed_file(library_path, manifest["library_sha256"], label="library")

    input_types = _decode_type_sequence(manifest["inputs"], label="input ABI")
    output_types = _decode_type_sequence(manifest["outputs"], label="output ABI")
    if not output_types:
        raise NativeBundleError("native bundle output ABI must contain at least one tensor")

    staging_directory = Path(tempfile.mkdtemp(prefix="tiny_tensor_compiler_bundle_"))
    staged_library = staging_directory / _library_name()
    try:
        shutil.copy2(library_path, staged_library)
        library = _load_library(staged_library)
        artifact = _NativeArtifact(staging_directory, library)
        try:
            library.tiny_tensor_run
        except AttributeError as error:
            artifact.close()
            raise NativeBundleError("native bundle is missing tiny_tensor_run") from error
    except Exception:
        shutil.rmtree(staging_directory, ignore_errors=True)
        raise

    return NativeBundleExecutable(artifact, input_types, output_types)


def _return_types(program: LoopProgram) -> tuple[TensorType, ...]:
    types = program.value_types
    try:
        return tuple(types[slot] for slot in program.return_slots)
    except KeyError as error:
        raise RuntimeError("verified loop IR return value unexpectedly has no type") from error


def _require_static_types(types: Sequence[TensorType]) -> None:
    for type_ in types:
        if not type_.is_static:
            raise NativeBundleError("native bundle ABI requires fully concrete tensor shapes")


def _encode_tensor_type(type_: TensorType) -> dict[str, object]:
    if not type_.is_static:
        raise NativeBundleError("native bundle ABI requires fully concrete tensor shapes")
    return {
        "dtype": type_.dtype.value,
        "shape": [int(dim) for dim in type_.shape],
    }


def _decode_type_sequence(value: object, *, label: str) -> tuple[TensorType, ...]:
    if not isinstance(value, list):
        raise NativeBundleError(f"native bundle {label} must be a list")
    return tuple(_decode_tensor_type(item, label=f"{label} entry {index}") for index, item in enumerate(value))


def _decode_tensor_type(value: object, *, label: str) -> TensorType:
    if not isinstance(value, dict) or set(value) != {"dtype", "shape"}:
        raise NativeBundleError(f"native bundle {label} is malformed")
    dtype_value = value["dtype"]
    shape_value = value["shape"]
    if not isinstance(dtype_value, str):
        raise NativeBundleError(f"native bundle {label} dtype is malformed")
    try:
        dtype = DType(dtype_value)
    except ValueError as error:
        raise NativeBundleError(f"native bundle {label} dtype is unsupported") from error
    if not isinstance(shape_value, list) or any(
        not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape_value
    ):
        raise NativeBundleError(f"native bundle {label} shape is malformed")
    return TensorType(tuple(shape_value), dtype)


def _target_identity() -> dict[str, object]:
    return {
        "os_name": os.name,
        "sys_platform": sys.platform,
        "machine": platform.machine(),
        "pointer_bits": ctypes.sizeof(ctypes.c_void_p) * 8,
    }


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    try:
        path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise NativeBundleError(f"failed to write native bundle manifest: {error}") from error


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NativeBundleError(f"failed to read native bundle manifest: {error}") from error
    if not isinstance(value, dict):
        raise NativeBundleError("native bundle manifest must be a JSON object")
    return value


def _verify_hashed_file(path: Path, expected_hash: object, *, label: str) -> None:
    if not isinstance(expected_hash, str) or not _is_sha256(expected_hash):
        raise NativeBundleError(f"native bundle {label} hash is malformed")
    if not path.is_file():
        raise NativeBundleError(f"native bundle {label} file is missing")
    if _sha256_file(path) != expected_hash:
        raise NativeBundleError(f"native bundle {label} hash does not match manifest")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise NativeBundleError(f"failed to hash native bundle file: {error}") from error
    return digest.hexdigest()


def _prepare_outputs(
    output_types: tuple[TensorType, ...],
    output: NativeOutput,
    runtime_inputs: Sequence[np.ndarray[Any, Any]],
) -> tuple[np.ndarray, ...]:
    output_count = len(output_types)
    if output_count == 1:
        if output is None or isinstance(output, np.ndarray):
            candidates: tuple[np.ndarray | None, ...] = (output,)
        else:
            raise TypeError("output must be a numpy.ndarray")
    else:
        if output is None:
            candidates = (None,) * output_count
        elif isinstance(output, np.ndarray) or not isinstance(output, Sequence):
            raise TypeError("multi-output bundle requires a sequence of numpy.ndarray outputs")
        else:
            candidates = tuple(output)
            if len(candidates) != output_count:
                raise ValueError(
                    f"multi-output bundle requires {output_count} output arrays, got {len(candidates)}"
                )

    outputs = tuple(
        _prepare_output_array(
            output_type,
            candidate,
            runtime_inputs,
            label="output" if output_count == 1 else f"output {index}",
        )
        for index, (output_type, candidate) in enumerate(
            zip(output_types, candidates, strict=True)
        )
    )
    for left_index, left in enumerate(outputs):
        for right_index in range(left_index + 1, len(outputs)):
            if np.shares_memory(left, outputs[right_index]):
                raise ValueError(f"outputs {left_index} and {right_index} must not overlap")
    return outputs


def _prepare_output_array(
    output_type: TensorType,
    output: np.ndarray | None,
    runtime_inputs: Sequence[np.ndarray[Any, Any]],
    *,
    label: str,
) -> np.ndarray:
    expected_dtype = np.dtype(output_type.dtype.to_numpy())
    if output is None:
        return np.empty(output_type.shape, dtype=expected_dtype)
    if not isinstance(output, np.ndarray):
        raise TypeError(f"{label} must be a numpy.ndarray")
    if tuple(output.shape) != output_type.shape:
        raise ValueError(
            f"{label} shape {tuple(output.shape)} does not match expected {output_type.shape}"
        )
    if output.dtype != expected_dtype:
        raise ValueError(f"{label} dtype {output.dtype} does not match expected {expected_dtype}")
    if not output.flags.c_contiguous:
        raise ValueError(f"{label} must be C-contiguous")
    if not output.flags.writeable:
        raise ValueError(f"{label} must be writable")
    if not output.flags.aligned:
        raise ValueError(f"{label} must be aligned for its dtype")
    for index, runtime_input in enumerate(runtime_inputs):
        if np.shares_memory(output, runtime_input):
            raise ValueError(f"{label} must not overlap runtime input {index}")
    return output


__all__ = [
    "NativeBundleError",
    "NativeBundleExecutable",
    "compile_native_bundle",
    "load_native_bundle",
]
