from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tiny_tensor_compiler import GraphBuilder, fuse_elementwise, lower_to_cpu, lower_to_loops


def _program():
    builder = GraphBuilder()
    x = builder.input((2, 3), dtype="float32")
    y = (x + 1).relu()
    z = x * 2
    module = builder.finish((y, z))
    return fuse_elementwise(lower_to_loops(lower_to_cpu(module)))


def _input() -> np.ndarray:
    return np.array([[-2.0, 0.0, 3.0], [4.0, -5.0, 1.5]], dtype=np.float32)


def test_native_bundle_round_trips_without_compiler(tmp_path: Path, monkeypatch) -> None:
    from tiny_tensor_compiler import native_bundle

    bundle = tmp_path / "program.ttcbundle"
    native_bundle.compile_native_bundle(_program(), bundle)

    monkeypatch.setattr(
        native_bundle,
        "_compiler_command",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("compiler lookup")),
    )
    executable = native_bundle.load_native_bundle(bundle)
    try:
        x = _input()
        actual_y, actual_z = executable(inputs=[x])
        np.testing.assert_array_equal(actual_y, np.maximum(x + np.float32(1), np.float32(0)))
        np.testing.assert_array_equal(actual_z, x * np.float32(2))
    finally:
        executable.close()


def test_native_bundle_manifest_is_self_describing(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle import compile_native_bundle

    bundle = tmp_path / "program.ttcbundle"
    compile_native_bundle(_program(), bundle)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema"] == "native-bundle-v1"
    assert manifest["inputs"] == [{"dtype": "f32", "shape": [2, 3]}]
    assert manifest["outputs"] == [
        {"dtype": "f32", "shape": [2, 3]},
        {"dtype": "f32", "shape": [2, 3]},
    ]
    assert len(manifest["library_sha256"]) == 64
    assert len(manifest["source_sha256"]) == 64
    assert (bundle / manifest["library"]).is_file()


def test_loaded_bundle_preserves_preallocated_multi_output_contract(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle import compile_native_bundle, load_native_bundle

    bundle = tmp_path / "program.ttcbundle"
    compile_native_bundle(_program(), bundle)
    executable = load_native_bundle(bundle)
    try:
        x = _input()
        out0 = np.empty((2, 3), dtype=np.float32)
        out1 = np.empty((2, 3), dtype=np.float32)
        result = executable(inputs=[x], out=(out0, out1))
        assert result[0] is out0
        assert result[1] is out1
        np.testing.assert_array_equal(out0, np.maximum(x + np.float32(1), np.float32(0)))
        np.testing.assert_array_equal(out1, x * np.float32(2))
    finally:
        executable.close()


def test_native_bundle_rejects_substituted_library(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle import (
        NativeBundleError,
        compile_native_bundle,
        load_native_bundle,
    )

    bundle = tmp_path / "program.ttcbundle"
    compile_native_bundle(_program(), bundle)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    library = bundle / manifest["library"]
    library.write_bytes(library.read_bytes() + b"tampered")

    with pytest.raises(NativeBundleError, match="hash"):
        load_native_bundle(bundle)


def test_native_bundle_rejects_wrong_target(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle import (
        NativeBundleError,
        compile_native_bundle,
        load_native_bundle,
    )

    bundle = tmp_path / "program.ttcbundle"
    compile_native_bundle(_program(), bundle)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["target"]["machine"] = "definitely-not-this-machine"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(NativeBundleError, match="target"):
        load_native_bundle(bundle)


def test_native_bundle_rejects_malformed_abi_type(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle import (
        NativeBundleError,
        compile_native_bundle,
        load_native_bundle,
    )

    bundle = tmp_path / "program.ttcbundle"
    compile_native_bundle(_program(), bundle)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"][0]["dtype"] = "u8"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(NativeBundleError, match="ABI"):
        load_native_bundle(bundle)


def test_failed_bundle_build_never_publishes_partial_destination(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tiny_tensor_compiler import native_bundle

    bundle = tmp_path / "program.ttcbundle"

    def fail_compile(*_args, **_kwargs):
        raise RuntimeError("synthetic compile failure")

    monkeypatch.setattr(native_bundle, "_compile_source", fail_compile)
    with pytest.raises(RuntimeError, match="synthetic"):
        native_bundle.compile_native_bundle(_program(), bundle)

    assert not bundle.exists()
    assert not tuple(tmp_path.glob(".program.ttcbundle.build-*"))


def test_bundle_destination_must_not_already_exist(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle import compile_native_bundle

    bundle = tmp_path / "program.ttcbundle"
    bundle.mkdir()
    with pytest.raises(FileExistsError):
        compile_native_bundle(_program(), bundle)
