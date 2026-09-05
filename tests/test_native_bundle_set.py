from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tiny_tensor_compiler import GraphBuilder, SymbolicDim


def _dynamic_module():
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    x = builder.input((batch, 3), dtype="float32")
    module = builder.finish((x.relu(), x + x))
    return module, batch


def _input(batch: int) -> np.ndarray:
    values = np.arange(batch * 3, dtype=np.float32).reshape(batch, 3)
    return values - np.float32(2)


def _assert_outputs(result, x: np.ndarray) -> None:
    actual_relu, actual_add = result
    np.testing.assert_array_equal(actual_relu, np.maximum(x, np.float32(0)))
    np.testing.assert_array_equal(actual_add, x + x)


def test_dynamic_bundle_set_dispatches_without_compiler(tmp_path: Path, monkeypatch) -> None:
    from tiny_tensor_compiler import native_bundle, native_bundle_set

    module, batch = _dynamic_module()
    bundle = tmp_path / "family.ttcset"
    native_bundle_set.compile_dynamic_bundle_set(
        module,
        ({batch: 2}, {batch: 5}),
        bundle,
    )

    monkeypatch.setattr(
        native_bundle,
        "_compiler_command",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("compiler lookup")),
    )
    executable = native_bundle_set.load_dynamic_bundle_set(bundle)
    try:
        assert executable.available_bindings == ((("B", 2),), (("B", 5),))
        assert executable.loaded_bindings == ()

        x2 = _input(2)
        _assert_outputs(executable(inputs=[x2]), x2)
        assert executable.loaded_bindings == ((("B", 2),),)
        _assert_outputs(executable(inputs=[x2]), x2)
        assert executable.loaded_bindings == ((("B", 2),),)

        x5 = _input(5)
        _assert_outputs(executable(inputs=[x5]), x5)
        assert executable.loaded_bindings == ((("B", 2),), (("B", 5),))
    finally:
        executable.close()


def test_dynamic_bundle_set_explicit_specialization_and_preallocated_outputs(
    tmp_path: Path,
) -> None:
    from tiny_tensor_compiler.native_bundle_set import (
        NativeBundleSetError,
        compile_dynamic_bundle_set,
        load_dynamic_bundle_set,
    )

    module, batch = _dynamic_module()
    bundle = tmp_path / "family.ttcset"
    compile_dynamic_bundle_set(module, ({batch: 2}, {batch: 5}), bundle)
    executable = load_dynamic_bundle_set(bundle)
    try:
        child = executable.specialize({"B": 5})
        x = _input(5)
        out0 = np.empty((5, 3), dtype=np.float32)
        out1 = np.empty((5, 3), dtype=np.float32)
        result = child(inputs=[x], out=(out0, out1))
        assert result[0] is out0
        assert result[1] is out1
        _assert_outputs(result, x)

        with pytest.raises(NativeBundleSetError, match="not packaged"):
            executable.specialize({"B": 3})
    finally:
        executable.close()


def test_dynamic_bundle_set_manifest_binds_symbolic_contract_to_child_abis(
    tmp_path: Path,
) -> None:
    from tiny_tensor_compiler.native_bundle_set import compile_dynamic_bundle_set

    module, batch = _dynamic_module()
    bundle = tmp_path / "family.ttcset"
    compile_dynamic_bundle_set(module, ({batch: 5}, {batch: 2}), bundle)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema"] == "native-bundle-set-v1"
    assert manifest["symbols"] == ["B"]
    assert manifest["inputs"] == [
        {
            "dtype": "f32",
            "shape": [
                {"kind": "symbol", "name": "B"},
                {"kind": "static", "value": 3},
            ],
        }
    ]
    assert [variant["bindings"] for variant in manifest["variants"]] == [
        {"B": 2},
        {"B": 5},
    ]
    for index, variant in enumerate(manifest["variants"]):
        child_manifest = bundle / variant["path"] / "manifest.json"
        digest = hashlib.sha256(child_manifest.read_bytes()).hexdigest()
        assert variant["path"] == f"variants/{index:04d}"
        assert variant["manifest_sha256"] == digest
        child = json.loads(child_manifest.read_text(encoding="utf-8"))
        assert variant["abi_sha256"] == child["abi_sha256"]


def test_dynamic_bundle_set_rejects_unbundled_runtime_shape(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_set import (
        NativeBundleSetError,
        compile_dynamic_bundle_set,
        load_dynamic_bundle_set,
    )

    module, batch = _dynamic_module()
    bundle = tmp_path / "family.ttcset"
    compile_dynamic_bundle_set(module, ({batch: 2}, {batch: 5}), bundle)
    executable = load_dynamic_bundle_set(bundle)
    try:
        with pytest.raises(NativeBundleSetError, match="does not match any packaged"):
            executable(inputs=[_input(3)])
    finally:
        executable.close()


def test_dynamic_bundle_set_rejects_duplicate_binding_before_build(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_set import compile_dynamic_bundle_set

    module, batch = _dynamic_module()
    bundle = tmp_path / "family.ttcset"
    with pytest.raises(ValueError, match="duplicate binding"):
        compile_dynamic_bundle_set(module, ({batch: 2}, {"B": 2}), bundle)
    assert not bundle.exists()


def test_dynamic_bundle_set_rejects_ambiguous_concrete_input_abi(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_set import compile_dynamic_bundle_set

    batch = SymbolicDim("B")
    width = SymbolicDim("W")
    builder = GraphBuilder()
    x = builder.input((batch + width,), dtype="float32")
    module = builder.finish(x.relu())
    bundle = tmp_path / "family.ttcset"

    with pytest.raises(ValueError, match="same concrete runtime input ABI"):
        compile_dynamic_bundle_set(
            module,
            ({batch: 1, width: 2}, {batch: 2, width: 1}),
            bundle,
        )
    assert not bundle.exists()


def test_dynamic_bundle_set_rejects_rewritten_binding_contract(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_set import (
        NativeBundleSetError,
        compile_dynamic_bundle_set,
        load_dynamic_bundle_set,
    )

    module, batch = _dynamic_module()
    bundle = tmp_path / "family.ttcset"
    compile_dynamic_bundle_set(module, ({batch: 2}, {batch: 5}), bundle)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["variants"][0]["bindings"]["B"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(NativeBundleSetError, match="binding does not match child input ABI"):
        load_dynamic_bundle_set(bundle)


def test_dynamic_bundle_set_rejects_child_manifest_substitution(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_set import (
        NativeBundleSetError,
        compile_dynamic_bundle_set,
        load_dynamic_bundle_set,
    )

    module, batch = _dynamic_module()
    bundle = tmp_path / "family.ttcset"
    compile_dynamic_bundle_set(module, ({batch: 2}, {batch: 5}), bundle)
    first = bundle / "variants" / "0000" / "manifest.json"
    second = bundle / "variants" / "0001" / "manifest.json"
    first.write_bytes(second.read_bytes())

    with pytest.raises(NativeBundleSetError, match="child manifest hash does not match"):
        load_dynamic_bundle_set(bundle)


def test_dynamic_bundle_set_outer_publication_is_atomic_on_child_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tiny_tensor_compiler import native_bundle_set

    module, batch = _dynamic_module()
    bundle = tmp_path / "family.ttcset"

    def fail_child(_program, destination, compiler=None):
        del compiler
        child = Path(destination)
        child.mkdir(parents=True)
        (child / "partial").write_text("partial", encoding="utf-8")
        raise RuntimeError("synthetic child compile failure")

    monkeypatch.setattr(native_bundle_set, "compile_native_bundle", fail_child)
    with pytest.raises(RuntimeError, match="synthetic child"):
        native_bundle_set.compile_dynamic_bundle_set(module, ({batch: 2},), bundle)

    assert not bundle.exists()
    assert not tuple(tmp_path.glob(".family.ttcset.build-*"))


def test_dynamic_bundle_set_destination_must_not_exist(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_set import compile_dynamic_bundle_set

    module, batch = _dynamic_module()
    bundle = tmp_path / "family.ttcset"
    bundle.mkdir()
    with pytest.raises(FileExistsError):
        compile_dynamic_bundle_set(module, ({batch: 2},), bundle)
