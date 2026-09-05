# Deterministic IR repro cases

The repro-case layer packages one verified high-level tensor program together with the exact runtime inputs that trigger it and the exact reference outputs observed at capture time.

Its purpose is deterministic debugging and differential replay. It is deliberately separate from native bundle distribution, release authorization, and backend-specific lowering state.

## Version 1 artifact

`capture_repro_case(module, inputs=...)` returns canonical JSON with exactly these fields:

- `format`: `tiny-tensor-repro`;
- `version`: `1`;
- `module`: the canonical version-1 tensor-IR snapshot produced by `serialize_module()`;
- `module_sha256`: the SHA-256 fingerprint of that canonical module snapshot;
- `inputs`: exact runtime arrays encoded as explicit dtype, shape, and canonical little-endian bytes;
- `expected_outputs`: exact reference outputs encoded by the same array format.

The artifact contains no timestamp, hostname, compiler path, native library, cache path, thread count, or other machine-local metadata. Equivalent module/input cases therefore produce byte-for-byte identical canonical JSON.

`repro_case_sha256(document)` canonicalizes and validates the full case before hashing it, so insignificant input JSON whitespace or key ordering does not create a different semantic case fingerprint.

Both SHA-256 values are content identities and integrity checks only. They are **not** publisher signatures, release authorization, threshold-policy evidence, rollback protection, or authenticity claims. Trusted native-bundle distribution remains the responsibility of the existing bundle trust/release subsystems.

## Capture and replay

```python
import numpy as np

from tiny_tensor_compiler import GraphBuilder
from tiny_tensor_compiler.repro import capture_repro_case, replay_repro_case

builder = GraphBuilder()
x = builder.input((2, 4), dtype="int32")
y = (x + 1).relu().reverse(axis=1)
module = builder.finish(y)

inputs = [np.arange(8, dtype=np.int32).reshape(2, 4)]
document = capture_repro_case(module, inputs=inputs)

reference = replay_repro_case(document, backend="reference")
native = replay_repro_case(document, backend="native")
```

Capture executes the ordinary verified reference interpreter first. Invalid input count, shape, dtype, or unresolved dynamic-shape constraints therefore fail instead of producing a malformed case.

Replay decodes and reverifies the tensor-IR snapshot, checks the module fingerprint, reconstructs exact input/expected arrays, executes the selected backend, and then requires output count, shape, dtype, and canonical raw bytes to match the captured reference outputs exactly. A numerical result that differs at the bit level raises `ReproMismatchError` with expected and actual output-content fingerprints.

Runtime-symbolic modules are supported because the concrete symbolic binding is recovered from the captured runtime inputs during replay. The artifact does not store a backend specialization or generated C; native replay recompiles or reuses the ordinary backend from the high-level verified program.

## Replay-time backend configuration

Backend execution policy is intentionally outside the artifact:

- `backend="reference"` performs pure tensor-IR reference replay;
- `backend="native"` uses the ordinary concrete or dynamic native compiler path;
- `compiler=`, `cache_dir=`, and `parallel=` are native replay-time options.

Keeping these settings outside the artifact lets the same case reproduce behavior across compilers, platforms, cache states, or serial/parallel execution without changing the captured high-level program and data.

## Fail-closed validation

Version 1 rejects rather than approximates:

- unknown format versions or unexpected schema fields;
- duplicate JSON object keys;
- non-canonical or unverifiable embedded tensor-IR snapshots;
- module SHA-256 mismatches;
- unsupported array dtypes;
- negative or malformed dimensions;
- invalid base64 or byte-length mismatches;
- stored input/output counts inconsistent with the module signature;
- unknown replay backends.

Array storage is canonical fixed-width little-endian data, including rank-zero arrays. Floating-point raw bits are preserved, including signed zero, infinities, and NaN payloads.

## Deliberate boundary

This phase provides deterministic capture, content addressing, and reference/native differential replay. It does not provide a randomized program generator, fuzz scheduler, failing-case minimizer, corpus database, sandbox, or distributed execution service.

A later testing phase can build randomized differential generation and deterministic shrinking on top of this stable case format without expanding the version-1 artifact into backend or release metadata.
