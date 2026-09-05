from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .ir import (
    AffineDim,
    DType,
    Function,
    LinearDim,
    Module,
    ShapeDim,
    SymbolicDim,
    TensorType,
)
from .verifier import VerificationError, verify

_FORMAT_NAME = "tiny-tensor-ir"
_FORMAT_VERSION = 1

_CANONICAL_DTYPES = {
    DType.INT32: np.dtype("<i4"),
    DType.INT64: np.dtype("<i8"),
    DType.FLOAT32: np.dtype("<f4"),
    DType.FLOAT64: np.dtype("<f8"),
}


class IRSerializationError(ValueError):
    """Raised when a serialized tensor-IR document is malformed or unverifiable."""


def serialize_module(module: Module) -> str:
    """Return one canonical versioned JSON representation of verified tensor IR."""
    if not isinstance(module, Module):
        raise TypeError("serialize_module requires a Module")
    verify(module)

    function = module.function
    payload = {
        "format": _FORMAT_NAME,
        "version": _FORMAT_VERSION,
        "function": {
            "name": function.name,
            "ops": [
                {
                    "opcode": op.opcode,
                    "operands": [operand.id for operand in op.operands],
                    "results": [
                        {"id": result.id, "type": _encode_tensor_type(result.type)}
                        for result in op.results
                    ],
                    "attrs": _encode_attrs(op.attrs),
                }
                for op in function.ops
            ],
        },
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def deserialize_module(document: str) -> Module:
    """Rebuild and reverify a module from a version-1 tensor-IR JSON document."""
    if not isinstance(document, str):
        raise TypeError("deserialize_module requires a JSON string")

    try:
        payload = json.loads(document, object_pairs_hook=_object_without_duplicates)
    except json.JSONDecodeError as exc:
        raise IRSerializationError(f"invalid tensor IR JSON: {exc.msg}") from exc

    root = _require_mapping(payload, "document")
    _require_exact_keys(root, {"format", "version", "function"}, "document")
    if root["format"] != _FORMAT_NAME:
        raise IRSerializationError(f"unsupported tensor IR format: {root['format']!r}")
    version = root["version"]
    if not isinstance(version, int) or isinstance(version, bool) or version != _FORMAT_VERSION:
        raise IRSerializationError(f"unsupported tensor IR version: {version!r}")

    function_record = _require_mapping(root["function"], "function")
    _require_exact_keys(function_record, {"name", "ops"}, "function")
    name = function_record["name"]
    if not isinstance(name, str):
        raise IRSerializationError("function name must be a string")
    operations = function_record["ops"]
    if not isinstance(operations, list):
        raise IRSerializationError("function ops must be a list")

    function = Function(name)
    values = {}
    symbols: dict[str, SymbolicDim] = {}

    for op_index, raw_record in enumerate(operations):
        record = _require_mapping(raw_record, f"op #{op_index}")
        _require_exact_keys(
            record,
            {"opcode", "operands", "results", "attrs"},
            f"op #{op_index}",
        )

        opcode = record["opcode"]
        if not isinstance(opcode, str) or not opcode:
            raise IRSerializationError(f"op #{op_index} opcode must be a non-empty string")

        operand_ids = _decode_value_ids(record["operands"], f"op #{op_index} operands")
        operands = []
        for value_id in operand_ids:
            try:
                operands.append(values[value_id])
            except KeyError as exc:
                raise IRSerializationError(
                    f"op #{op_index} references undefined value %{value_id}"
                ) from exc

        raw_results = record["results"]
        if not isinstance(raw_results, list):
            raise IRSerializationError(f"op #{op_index} results must be a list")
        declared_ids: list[int] = []
        result_types: list[TensorType] = []
        for result_index, raw_result in enumerate(raw_results):
            result_record = _require_mapping(
                raw_result,
                f"op #{op_index} result #{result_index}",
            )
            _require_exact_keys(
                result_record,
                {"id", "type"},
                f"op #{op_index} result #{result_index}",
            )
            value_id = _decode_value_id(
                result_record["id"],
                f"op #{op_index} result #{result_index} id",
            )
            if value_id in values or value_id in declared_ids:
                raise IRSerializationError(f"value %{value_id} is defined more than once")
            declared_ids.append(value_id)
            result_types.append(_decode_tensor_type(result_record["type"], symbols))

        attrs = _decode_attrs(record["attrs"])
        op = function.add_op(opcode, operands=operands, result_types=result_types, attrs=attrs)
        actual_ids = [result.id for result in op.results]
        if actual_ids != declared_ids:
            raise IRSerializationError(
                f"op #{op_index} result ids are not canonical: "
                f"expected {actual_ids}, found {declared_ids}"
            )
        for result in op.results:
            values[result.id] = result

    module = Module(function)
    try:
        verify(module)
    except VerificationError as exc:
        raise IRSerializationError(f"serialized tensor IR failed verification: {exc}") from exc
    return module


def module_sha256(module: Module) -> str:
    """Return the SHA-256 digest of the module's canonical UTF-8 serialization."""
    return hashlib.sha256(serialize_module(module).encode("utf-8")).hexdigest()


def _encode_tensor_type(type_: TensorType) -> dict[str, Any]:
    return {
        "dtype": type_.dtype.value,
        "shape": [_encode_shape_dim(dim) for dim in type_.shape],
    }


def _decode_tensor_type(raw: Any, symbols: dict[str, SymbolicDim]) -> TensorType:
    record = _require_mapping(raw, "tensor type")
    _require_exact_keys(record, {"dtype", "shape"}, "tensor type")
    dtype_text = record["dtype"]
    if not isinstance(dtype_text, str):
        raise IRSerializationError("tensor dtype must be a string")
    try:
        dtype = DType(dtype_text)
    except ValueError as exc:
        raise IRSerializationError(f"unsupported tensor dtype: {dtype_text!r}") from exc
    raw_shape = record["shape"]
    if not isinstance(raw_shape, list):
        raise IRSerializationError("tensor shape must be a list")
    shape = tuple(_decode_shape_dim(dim, symbols) for dim in raw_shape)
    try:
        return TensorType(shape, dtype)
    except (TypeError, ValueError) as exc:
        raise IRSerializationError(str(exc)) from exc


def _encode_shape_dim(dim: ShapeDim) -> Any:
    if isinstance(dim, bool):
        raise IRSerializationError("boolean tensor dimensions are not serializable")
    if isinstance(dim, int):
        return dim
    if isinstance(dim, SymbolicDim):
        return {"kind": "symbol", "name": dim.name}
    if isinstance(dim, AffineDim):
        return {
            "kind": "affine",
            "symbol": dim.symbol.name,
            "scale": dim.scale,
            "offset": dim.offset,
        }
    if isinstance(dim, LinearDim):
        return {
            "kind": "linear",
            "terms": [
                {"symbol": symbol.name, "coefficient": coefficient}
                for symbol, coefficient in dim.terms
            ],
            "offset": dim.offset,
        }
    raise IRSerializationError(f"unsupported tensor dimension: {type(dim).__name__}")


def _decode_shape_dim(raw: Any, symbols: dict[str, SymbolicDim]) -> ShapeDim:
    if isinstance(raw, bool):
        raise IRSerializationError("boolean tensor dimensions are invalid")
    if isinstance(raw, int):
        if raw < 0:
            raise IRSerializationError("tensor dimensions must be non-negative")
        return raw

    record = _require_mapping(raw, "symbolic tensor dimension")
    kind = record.get("kind")
    if kind == "symbol":
        _require_exact_keys(record, {"kind", "name"}, "symbolic tensor dimension")
        return _symbol(record["name"], symbols)
    if kind == "affine":
        _require_exact_keys(
            record,
            {"kind", "symbol", "scale", "offset"},
            "affine tensor dimension",
        )
        symbol = _symbol(record["symbol"], symbols)
        scale = _require_plain_int(record["scale"], "affine scale")
        offset = _require_plain_int(record["offset"], "affine offset")
        try:
            return AffineDim(symbol, scale=scale, offset=offset)
        except (TypeError, ValueError) as exc:
            raise IRSerializationError(str(exc)) from exc
    if kind == "linear":
        _require_exact_keys(
            record,
            {"kind", "terms", "offset"},
            "linear tensor dimension",
        )
        raw_terms = record["terms"]
        if not isinstance(raw_terms, list):
            raise IRSerializationError("linear terms must be a list")
        terms = []
        for term_index, raw_term in enumerate(raw_terms):
            term = _require_mapping(raw_term, f"linear term #{term_index}")
            _require_exact_keys(
                term,
                {"symbol", "coefficient"},
                f"linear term #{term_index}",
            )
            terms.append(
                (
                    _symbol(term["symbol"], symbols),
                    _require_plain_int(
                        term["coefficient"],
                        f"linear term #{term_index} coefficient",
                    ),
                )
            )
        offset = _require_plain_int(record["offset"], "linear offset")
        try:
            return LinearDim(tuple(terms), offset=offset)
        except (TypeError, ValueError) as exc:
            raise IRSerializationError(str(exc)) from exc
    raise IRSerializationError(f"unsupported symbolic tensor dimension kind: {kind!r}")


def _symbol(raw_name: Any, symbols: dict[str, SymbolicDim]) -> SymbolicDim:
    if not isinstance(raw_name, str):
        raise IRSerializationError("symbolic dimension name must be a string")
    try:
        return symbols.setdefault(raw_name, SymbolicDim(raw_name))
    except ValueError as exc:
        raise IRSerializationError(str(exc)) from exc


def _encode_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    encoded = {}
    for key in sorted(attrs):
        if not isinstance(key, str):
            raise IRSerializationError("operation attribute names must be strings")
        encoded[key] = _encode_attr(attrs[key])
    return encoded


def _decode_attrs(raw: Any) -> dict[str, Any]:
    record = _require_mapping(raw, "operation attrs")
    return {key: _decode_attr(value) for key, value in record.items()}


def _encode_attr(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IRSerializationError("non-finite scalar float attributes are unsupported")
        return {"kind": "float", "hex": value.hex()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        return _encode_attr(float(value))
    if isinstance(value, np.ndarray):
        return _encode_array(value)
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_encode_attr(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_encode_attr(item) for item in value]}
    if isinstance(value, Mapping):
        items = []
        for key in sorted(value):
            if not isinstance(key, str):
                raise IRSerializationError("nested attribute mapping keys must be strings")
            items.append([key, _encode_attr(value[key])])
        return {"kind": "mapping", "items": items}
    raise IRSerializationError(f"unsupported operation attribute type: {type(value).__name__}")


def _decode_attr(raw: Any) -> Any:
    if raw is None or isinstance(raw, (bool, int, str)):
        return raw
    if isinstance(raw, float):
        raise IRSerializationError("bare JSON float attributes are not canonical")
    if not isinstance(raw, Mapping):
        raise IRSerializationError("operation attribute has an unsupported JSON shape")

    kind = raw.get("kind")
    if kind == "float":
        _require_exact_keys(raw, {"kind", "hex"}, "float attribute")
        text = raw["hex"]
        if not isinstance(text, str):
            raise IRSerializationError("float attribute hex value must be a string")
        try:
            value = float.fromhex(text)
        except ValueError as exc:
            raise IRSerializationError(f"invalid float attribute: {text!r}") from exc
        if not math.isfinite(value):
            raise IRSerializationError("non-finite scalar float attributes are unsupported")
        return value
    if kind in {"tuple", "list"}:
        _require_exact_keys(raw, {"kind", "items"}, f"{kind} attribute")
        items = raw["items"]
        if not isinstance(items, list):
            raise IRSerializationError(f"{kind} attribute items must be a list")
        decoded = [_decode_attr(item) for item in items]
        return tuple(decoded) if kind == "tuple" else decoded
    if kind == "mapping":
        _require_exact_keys(raw, {"kind", "items"}, "mapping attribute")
        items = raw["items"]
        if not isinstance(items, list):
            raise IRSerializationError("mapping attribute items must be a list")
        result = {}
        for item_index, item in enumerate(items):
            if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str):
                raise IRSerializationError(
                    f"mapping attribute item #{item_index} must be [string, value]"
                )
            key = item[0]
            if key in result:
                raise IRSerializationError(f"duplicate nested attribute key: {key!r}")
            result[key] = _decode_attr(item[1])
        return result
    if kind == "ndarray":
        return _decode_array(raw)
    raise IRSerializationError(f"unsupported operation attribute kind: {kind!r}")


def _encode_array(value: np.ndarray[Any, Any]) -> dict[str, Any]:
    array = np.asarray(value)
    try:
        dtype = DType.from_numpy(array.dtype)
    except TypeError as exc:
        raise IRSerializationError(str(exc)) from exc
    canonical_dtype = _CANONICAL_DTYPES[dtype]
    canonical = np.ascontiguousarray(array, dtype=canonical_dtype)
    return {
        "kind": "ndarray",
        "dtype": dtype.value,
        "shape": list(canonical.shape),
        "data": base64.b64encode(canonical.tobytes(order="C")).decode("ascii"),
    }


def _decode_array(raw: Mapping[str, Any]) -> np.ndarray[Any, Any]:
    _require_exact_keys(raw, {"kind", "dtype", "shape", "data"}, "ndarray attribute")
    dtype_text = raw["dtype"]
    if not isinstance(dtype_text, str):
        raise IRSerializationError("ndarray dtype must be a string")
    try:
        dtype = DType(dtype_text)
    except ValueError as exc:
        raise IRSerializationError(f"unsupported ndarray dtype: {dtype_text!r}") from exc

    raw_shape = raw["shape"]
    if not isinstance(raw_shape, list):
        raise IRSerializationError("ndarray shape must be a list")
    shape = []
    for axis, dim in enumerate(raw_shape):
        value = _require_plain_int(dim, f"ndarray shape axis {axis}")
        if value < 0:
            raise IRSerializationError("ndarray dimensions must be non-negative")
        shape.append(value)

    data = raw["data"]
    if not isinstance(data, str):
        raise IRSerializationError("ndarray data must be base64 text")
    try:
        payload = base64.b64decode(data.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise IRSerializationError("ndarray data is not canonical base64") from exc

    canonical_dtype = _CANONICAL_DTYPES[dtype]
    count = math.prod(shape)
    expected_bytes = count * canonical_dtype.itemsize
    if len(payload) != expected_bytes:
        raise IRSerializationError(
            f"ndarray byte length mismatch: expected {expected_bytes}, found {len(payload)}"
        )
    array = np.frombuffer(payload, dtype=canonical_dtype).reshape(tuple(shape))
    return np.array(array, dtype=dtype.to_numpy(), order="C", copy=True)


def _decode_value_ids(raw: Any, context: str) -> list[int]:
    if not isinstance(raw, list):
        raise IRSerializationError(f"{context} must be a list")
    return [_decode_value_id(value, f"{context} item #{index}") for index, value in enumerate(raw)]


def _decode_value_id(raw: Any, context: str) -> int:
    value = _require_plain_int(raw, context)
    if value < 0:
        raise IRSerializationError(f"{context} must be non-negative")
    return value


def _require_plain_int(raw: Any, context: str) -> int:
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise IRSerializationError(f"{context} must be an integer")
    return raw


def _require_mapping(raw: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise IRSerializationError(f"{context} must be a JSON object")
    if any(not isinstance(key, str) for key in raw):
        raise IRSerializationError(f"{context} keys must be strings")
    return raw


def _require_exact_keys(raw: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(raw)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise IRSerializationError(
            f"{context} has unexpected keys: missing={missing}, extra={extra}"
        )


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise IRSerializationError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result
