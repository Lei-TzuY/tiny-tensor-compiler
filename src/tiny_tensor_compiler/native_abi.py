from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from .ir import TensorType
from .loop_ir import LoopProgram

NATIVE_ABI_SYMBOL = "tiny_tensor_abi_sha256"


def native_abi_sha256(program: LoopProgram) -> str:
    """Return the canonical concrete input/output ABI identity for one LoopProgram."""
    return tensor_abi_sha256(program.input_types, _return_types(program))


def tensor_abi_sha256(
    input_types: Sequence[TensorType],
    output_types: Sequence[TensorType],
) -> str:
    """Hash the ordered concrete tensor ABI using the native-bundle canonical form."""
    encoded_inputs = [_encode_tensor_type(type_) for type_ in input_types]
    encoded_outputs = [_encode_tensor_type(type_) for type_ in output_types]
    payload = json.dumps(
        {"inputs": encoded_inputs, "outputs": encoded_outputs},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def append_native_abi_export(source: str, abi_sha256: str) -> str:
    """Append the ABI identity export without changing public generated-C semantics."""
    if len(abi_sha256) != 64 or any(character not in "0123456789abcdef" for character in abi_sha256):
        raise ValueError("native ABI fingerprint must be lowercase SHA-256 hex")
    return (
        source
        + "\nTINY_TENSOR_EXPORT const char *tiny_tensor_abi_sha256(void) {\n"
        + f'    return "{abi_sha256}";\n'
        + "}\n"
    )


def _return_types(program: LoopProgram) -> tuple[TensorType, ...]:
    types = program.value_types
    try:
        return tuple(types[slot] for slot in program.return_slots)
    except KeyError as error:
        raise RuntimeError("verified loop IR return value unexpectedly has no type") from error


def _encode_tensor_type(type_: TensorType) -> dict[str, object]:
    if not type_.is_static:
        raise ValueError("native ABI identity requires fully concrete tensor shapes")
    return {
        "dtype": type_.dtype.value,
        "shape": [int(dim) for dim in type_.shape],
    }
