# Canonical tensor IR snapshots

The tensor compiler can persist its verified high-level tensor IR as a deterministic, versioned JSON document. This format is intended for reproducible compiler inputs, regression snapshots, content-addressed caching, and cross-process replay of the typed tensor program before lowering.

## Public API

The serialization API lives in `tiny_tensor_compiler.serialization`:

```python
from tiny_tensor_compiler.serialization import (
    deserialize_module,
    module_sha256,
    serialize_module,
)

text = serialize_module(module)
digest = module_sha256(module)
restored = deserialize_module(text)
```

`serialize_module()` verifies the source `Module` before producing bytes. `deserialize_module()` reconstructs a fresh object graph and runs the ordinary tensor-IR verifier again before returning it.

## Version 1 document

Version 1 uses the top-level identity:

```json
{"format":"tiny-tensor-ir","version":1,...}
```

The document records, in program order:

- the function name;
- every operation opcode;
- operand SSA value ids;
- result SSA value ids and complete `TensorType` values;
- operation attributes.

SSA result ids are canonical. Deserialization requires each operand to reference a value that has already been defined and requires the reconstructed `Function` allocator to reproduce the declared result ids. Forward references, duplicate definitions, duplicate JSON object keys, unknown top-level fields, and unsupported format versions are rejected.

## Tensor types and symbolic dimensions

Tensor dtypes are encoded by the compiler's stable dtype spelling (`i32`, `i64`, `f32`, `f64`). Shapes preserve the full current tensor-IR dimension surface:

- concrete non-negative integer extents;
- `SymbolicDim` names;
- one-variable `AffineDim` scale/offset terms;
- canonical multi-symbol `LinearDim` terms.

During decode, identical symbolic names are interned to one `SymbolicDim` object so relations among repeated symbolic terms are preserved. The ordinary verifier and dynamic-specialization machinery remain responsible for semantic legality and runtime solving.

## Constants and exact numeric bits

NumPy constant attributes are encoded independently of the host machine's native byte order:

- dtype is normalized to canonical little-endian fixed-width storage;
- shape is recorded explicitly, including rank-zero shape `[]`;
- row-major raw bytes are base64 encoded.

This preserves exact floating-point payload bits, including signed zero, infinities, and NaN payloads. Those values never rely on non-standard JSON `NaN` or `Infinity` tokens. Scalar constants remain true zero-dimensional arrays; canonicalization must not promote them to shape `(1,)`.

The decoder validates the expected byte count from dtype and shape before reconstructing an owned C-order NumPy array.

## Canonical bytes and fingerprints

`serialize_module()` emits UTF-8 JSON with sorted object keys, no insignificant whitespace, and no non-standard JSON numeric constants. Independently constructed but structurally identical verified modules therefore serialize to the same text.

`module_sha256()` hashes exactly those canonical UTF-8 bytes and returns a 64-character lowercase SHA-256 digest. The digest is a content fingerprint for the serialized tensor IR; it is not a publisher signature, release authorization, or replacement for the repository's native-bundle trust model.

## Verification and fail-closed decoding

The decoder treats the serialized document as untrusted structured data. It checks the versioned schema, rejects malformed encodings, rebuilds only tensor-IR objects, and invokes `verify()` on the reconstructed module. A document that cannot produce verified tensor IR fails with `IRSerializationError`.

Deserialization does not execute generated code, load a native library, or deserialize Python objects through `pickle`. This is nevertheless not a sandbox or a general security boundary: callers that subsequently compile or execute the returned verified module are explicitly choosing to run that tensor program.

Unknown future versions are rejected rather than interpreted approximately. New tensor operations can reuse the generic opcode/operand/result/attribute representation only when the installed verifier understands their semantics; otherwise post-decode verification rejects the document.

## Scope boundary

Version 1 serializes the high-level typed tensor `Module` only. It deliberately does not persist:

- Buffer IR or memory plans;
- Loop IR, fusion decisions, or scheduling choices;
- native compiler commands or generated shared libraries;
- persistent native cache entries or bundle archives;
- release-checkpoint, publisher-attestation, threshold-policy, or rollback trust state.

Those layers are derived from the reverified tensor module and may evolve independently. Keeping the snapshot above physical lowering means a saved program can be replayed through the current verifier, optimizer, layout analysis, code generator, and native runtime instead of freezing stale backend decisions into the interchange format.

## Regression evidence

The version-1 regression suite exercises canonical round trips across the current difficult tensor-IR surfaces, including composed storage aliases (`view`, `reverse`, `transpose`, `slice`), writable `copy_into` effects, multi-output returns, symbolic/affine/linear dimensions, dynamic specialization, native execution, exact floating-point constant bits, rank-zero constants, malformed base64, forward SSA references, duplicate JSON keys, and unsupported versions.
