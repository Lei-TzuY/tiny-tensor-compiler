from __future__ import annotations

import json
from pathlib import Path

import pytest

from tiny_tensor_compiler import GraphBuilder, SymbolicDim


def test_dynamic_bundle_set_rejects_repeated_linear_symbol_that_drops_a_declared_symbol(
    tmp_path: Path,
) -> None:
    from tiny_tensor_compiler.native_bundle_set import (
        NativeBundleSetError,
        compile_dynamic_bundle_set,
        load_dynamic_bundle_set,
    )

    batch = SymbolicDim("B")
    width = SymbolicDim("W")
    builder = GraphBuilder()
    x = builder.input((batch + width,), dtype="float32")
    module = builder.finish(x.relu())
    bundle = tmp_path / "family.ttcset"
    compile_dynamic_bundle_set(module, ({batch: 1, width: 2},), bundle)

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"][0]["shape"][0]["terms"] = [["B", 1], ["B", 2]]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(NativeBundleSetError, match="linear terms|every declared symbol"):
        load_dynamic_bundle_set(bundle)
