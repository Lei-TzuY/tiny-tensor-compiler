from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .compiler import compile_dynamic_module, compile_module
from .ir import DType, Module
from .runtime import ExecutionResult, execute_reference
from .serialization import (
    IRSerializationError,
    deserialize_module,
    module_sha256,
    serialize_module,
)
from .symbolic import has_symbolic_shapes

_FORMAT_NAME = "tiny-tensor-repro"
_FORMAT_VERSION = 1

_CANONICAL_DTYPES = {
    DType.INT32: np.dtype("<i4"),
    DType.INT64: np.dtype("<i8"),
    DType.FLOAT32: np.dtype("<f4"),
    DType.FLOAT64: np.dtype("<f8"),
}


class ReproCaseError(ValueError):
    """Raised when a deterministic repro artifact is malformed or unverifiable."""


class ReproMismatchError(AssertionError):
    """Raised when replay output does not match the captured reference bits."""


@dataclass(frozen=True)
class ReproCase:
    """Verified high-level module plus exact runtime inputs and reference outputs."""

    module: Module
    inputs: tuple[np.ndarray, ...]
    expected_outputs: tuple[np.ndarray, ...]
    module_digest: str


def capture_repro_case(module: Module, inputs: Sequence[Any] = ()) -> str:
    """Capture one canonical deterministic high-level IR replay artifact."""
    if not isinstance(module, Module):
        raise TypeError("capture_repro_case requires a Module")

    provided = tuple(inputs)
    expected = execute_reference(module, inputs=provided)
    input_arrays = tuple(_freeze_array(np.asarray(value)) for value in provided)
    expected_outputs = tuple(_freeze_array(value) for value in _result_tuple(expected))
    case = ReproCase(
        module=module,
        inputs=input_arrays,
        expected_outputs=expected_outputs,
        module_digest=module_sha256(module),
    )
    return _serialize_case(case)


def load_repro_case(document: str) -> ReproCase:
    """Decode and fail-closed validate a version-1 deterministic repro artifact."""
    payload = _parse_document(document)
    _require_exact_keys(
        payload,
        {
            "expected_outputs",
            "format",
            "inputs",
            "module",
            "module_sha256",
            "version",
        },
        "repro document",
    )
    if payload["format"] != _FORMAT_NAME:
        raise ReproCaseError(f"unsupported repro format: {payload['format']!r}")
    version = payload["version"]
    if not isinstance(version, int) or isinstance(version, bool) or version != _FORMAT_VERSION:
        raise ReproCaseError(f"unsupported repro version: {version!r}")

    module_document = payload["module"]
    if not isinstance(module_document, str):
        raise ReproCaseError("repro module must be a canonical tensor-IR JSON string")
    try:
        module = deserialize_module(module_document)
    except IRSerializationError as exc:
        raise ReproCaseError(f"repro module is invalid: {exc}") from exc
    if serialize_module(module) != module_document:
        raise ReproCaseError("repro module snapshot is not canonical")

    digest = payload["module_sha256"]
    if not _is_sha256(digest):
        raise ReproCaseError("module SHA-256 must be 64 lowercase hexadecimal characters")
    actual_digest = module_sha256(module)
    if digest != actual_digest:
        raise ReproCaseError(
            f"module SHA-256 mismatch: expected {digest}, reconstructed {actual_digest}"
        )

    inputs = _decode_array_list(payload["inputs"], "inputs")
    expected_outputs = _decode_array_list(payload["expected_outputs"], "expected_outputs")
    input_count = sum(op.opcode == "input" for op in module.function.ops)
    if len(inputs) != input_count:
        raise ReproCaseError(
            f"repro input count mismatch: module declares {input_count}, artifact stores {len(inputs)}"
        )
    return_op = module.function.ops[-1]
    expected_count = len(return_op.operands)
    if len(expected_outputs) != expected_count:
        raise ReproCaseError(
            "repro expected-output count mismatch: "
            f"module returns {expected_count}, artifact stores {len(expected_outputs)}"
        )

    return ReproCase(
        module=module,
        inputs=inputs,
        expected_outputs=expected_outputs,
        module_digest=digest,
    )


def repro_case_sha256(document: str) -> str:
    """Return the content fingerprint of the canonicalized verified repro artifact."""
    case = load_repro_case(document)
    canonical = _serialize_case(case)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def replay_repro_case(
    document: str,
    *,
    backend: str = "reference",
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    parallel: bool = False,
) -> ExecutionResult:
    """Replay a captured case and require exact shape/dtype/raw-bit output equality."""
    case = load_repro_case(document)
    if backend == "reference":
        if compiler is not None or cache_dir is not None or parallel:
            raise ValueError("compiler, cache_dir, and parallel are native-backend options")
        actual = execute_reference(case.module, inputs=case.inputs)
    elif backend == "native":
        if has_symbolic_shapes(case.module):
            executable = compile_dynamic_module(
                case.module,
                compiler=compiler,
                cache_dir=cache_dir,
                parallel=parallel,
            )
        else:
            executable = compile_module(
                case.module,
                compiler=compiler,
                cache_dir=cache_dir,
                parallel=parallel,
            )
        actual = executable(inputs=case.inputs)
    else:
        raise ValueError("backend must be 'reference' or 'native'")

    _require_expected_bits(actual, case.expected_outputs)
    return actual


def _serialize_case(case: ReproCase) -> str:
    module_document = serialize_module(case.module)
    digest = module_sha256(case.module)
    if case.module_digest != digest:
        raise ReproCaseError(
            f"module SHA-256 mismatch: case stores {case.module_digest}, module is {digest}"
        )
    payload = {
        "format": _FORMAT_NAME,
        "version": _FORMAT_VERSION,
        "module": module_document,
        "module_sha256": digest,
        "inputs": [_encode_array(value) for value in case.inputs],
        "expected_outputs": [_encode_array(value) for value in case.expected_outputs],
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse_document(document: str) -> Mapping[str, Any]:
    if not isinstance(document, str):
        raise TypeError("repro document must be a JSON string")
    try:
        payload = json.loads(document, object_pairs_hook=_object_without_duplicates)
    except json.JSONDecodeError as exc:
        raise ReproCaseError(f"invalid repro JSON: {exc.msg}") from exc
    return _require_mapping(payload, "repro document")


def _encode_array(value: np.ndarray[Any, Any]) -> dict[str, Any]:
    array = np.asarray(value)
    try:
        dtype = DType.from_numpy(array.dtype)
    except TypeError as exc:
        raise ReproCaseError(str(exc)) from exc
    canonical_dtype = _CANONICAL_DTYPES[dtype]
    canonical = np.array(array, dtype=canonical_dtype, order="C", copy=True).reshape(array.shape)
    return {
        "dtype": dtype.value,
        "shape": list(array.shape),
        "data": base64.b64encode(canonical.tobytes(order="C")).decode("ascii"),
    }


def _decode_array_list(raw: Any, context: str) -> tuple[np.ndarray, ...]:
    if not isinstance(raw, list):
        raise ReproCaseError(f"{context} must be a list")
    return tuple(_decode_array(item, f"{context} item #{index}") for index, item in enumerate(raw))


def _decode_array(raw: Any, context: str) -> np.ndarray:
    record = _require_mapping(raw, context)
    _require_exact_keys(record, {"dtype", "shape", "data"}, context)
    dtype_text = record["dtype"]
    if not isinstance(dtype_text, str):
        raise ReproCaseError(f"{context} dtype must be a string")
    try:
        dtype = DType(dtype_text)
    except ValueError as exc:
        raise ReproCaseError(f"unsupported {context} dtype: {dtype_text!r}") from exc

    raw_shape = record["shape"]
    if not isinstance(raw_shape, list):
        raise ReproCaseError(f"{context} shape must be a list")
    shape = []
    for axis, raw_dim in enumerate(raw_shape):
        if not isinstance(raw_dim, int) or isinstance(raw_dim, bool):
            raise ReproCaseError(f"{context} shape axis {axis} must be an integer")
        if raw_dim < 0:
            raise ReproCaseError(f"{context} dimensions must be non-negative")
        shape.append(raw_dim)

    data = record["data"]
    if not isinstance(data, str):
        raise ReproCaseError(f"{context} data must be base64 text")
    try:
        payload = base64.b64decode(data.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise ReproCaseError(f"{context} data is not canonical base64") from exc

    canonical_dtype = _CANONICAL_DTYPES[dtype]
    expected_bytes = math.prod(shape) * canonical_dtype.itemsize
    if len(payload) != expected_bytes:
        raise ReproCaseError(
            f"{context} byte length mismatch: expected {expected_bytes}, found {len(payload)}"
        )
    decoded = np.frombuffer(payload, dtype=canonical_dtype).reshape(tuple(shape))
    return _freeze_array(np.array(decoded, dtype=dtype.to_numpy(), order="C", copy=True))


def _freeze_array(value: np.ndarray[Any, Any]) -> np.ndarray:
    array = np.array(value, copy=True, order="C").reshape(value.shape)
    array.setflags(write=False)
    return array


def _result_tuple(result: ExecutionResult) -> tuple[np.ndarray, ...]:
    return result if isinstance(result, tuple) else (result,)


def _require_expected_bits(
    actual: ExecutionResult,
    expected_outputs: tuple[np.ndarray, ...],
) -> None:
    actual_outputs = _result_tuple(actual)
    if len(actual_outputs) != len(expected_outputs):
        raise ReproMismatchError(
            f"output count mismatch: expected {len(expected_outputs)}, found {len(actual_outputs)}"
        )

    for index, (actual_output, expected) in enumerate(
        zip(actual_outputs, expected_outputs, strict=True)
    ):
        actual_array = np.asarray(actual_output)
        if actual_array.shape != expected.shape:
            raise ReproMismatchError(
                f"output #{index} shape mismatch: expected {expected.shape}, found {actual_array.shape}"
            )
        if actual_array.dtype != expected.dtype:
            raise ReproMismatchError(
                f"output #{index} dtype mismatch: expected {expected.dtype}, found {actual_array.dtype}"
            )
        actual_bytes = _canonical_bytes(actual_array)
        expected_bytes = _canonical_bytes(expected)
        if actual_bytes != expected_bytes:
            raise ReproMismatchError(
                f"output #{index} raw bytes mismatch: expected sha256 "
                f"{hashlib.sha256(expected_bytes).hexdigest()}, found "
                f"{hashlib.sha256(actual_bytes).hexdigest()}"
            )


def _canonical_bytes(value: np.ndarray[Any, Any]) -> bytes:
    array = np.asarray(value)
    try:
        dtype = DType.from_numpy(array.dtype)
    except TypeError as exc:
        raise ReproCaseError(str(exc)) from exc
    canonical = np.array(
        array,
        dtype=_CANONICAL_DTYPES[dtype],
        order="C",
        copy=True,
    ).reshape(array.shape)
    return canonical.tobytes(order="C")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_mapping(raw: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ReproCaseError(f"{context} must be a JSON object")
    if any(not isinstance(key, str) for key in raw):
        raise ReproCaseError(f"{context} keys must be strings")
    return raw


def _require_exact_keys(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(raw)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ReproCaseError(
            f"{context} has unexpected keys: missing={missing}, extra={extra}"
        )


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReproCaseError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result
