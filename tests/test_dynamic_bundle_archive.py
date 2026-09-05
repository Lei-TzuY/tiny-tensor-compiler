from __future__ import annotations

import json
import stat
import warnings
import zipfile
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
    relu, added = result
    np.testing.assert_array_equal(relu, np.maximum(x, np.float32(0)))
    np.testing.assert_array_equal(added, x + x)


def _compile_family(tmp_path: Path):
    from tiny_tensor_compiler.native_bundle_set import compile_dynamic_bundle_set

    module, batch = _dynamic_module()
    bundle = tmp_path / "family.ttcset"
    compile_dynamic_bundle_set(module, ({batch: 2}, {batch: 5}), bundle)
    return bundle


def test_dynamic_bundle_archive_is_deterministic_and_compiler_free(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tiny_tensor_compiler import native_bundle, native_bundle_archive

    bundle = _compile_family(tmp_path)
    first = tmp_path / "family-a.ttca"
    second = tmp_path / "family-b.ttca"
    native_bundle_archive.pack_dynamic_bundle_set_archive(bundle, first)
    native_bundle_archive.pack_dynamic_bundle_set_archive(bundle, second)

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as packed:
        assert packed.namelist()[0] == "archive.json"
        assert packed.namelist()[1:] == sorted(packed.namelist()[1:])
        assert all(info.compress_type == zipfile.ZIP_STORED for info in packed.infolist())
        descriptor = json.loads(packed.read("archive.json"))
        assert descriptor == {
            "kind": "dynamic-bundle-set",
            "root": "bundle",
            "schema": "native-bundle-archive-v1",
        }

    monkeypatch.setattr(
        native_bundle,
        "_compiler_command",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("compiler lookup")),
    )
    executable = native_bundle_archive.load_dynamic_bundle_set_archive(first)
    try:
        assert executable.available_bindings == ((("B", 2),), (("B", 5),))
        x2 = _input(2)
        _assert_outputs(executable(inputs=[x2]), x2)
        child = executable.specialize({"B": 5})
        x5 = _input(5)
        out0 = np.empty_like(x5)
        out1 = np.empty_like(x5)
        result = child(inputs=[x5], out=(out0, out1))
        assert result[0] is out0
        assert result[1] is out1
        _assert_outputs(result, x5)
    finally:
        executable.close()

    assert executable.closed
    with pytest.raises(RuntimeError, match="archive executable is closed"):
        executable(inputs=[_input(2)])


def test_dynamic_bundle_archive_rejects_tampered_child_library(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_archive import (
        NativeBundleArchiveError,
        load_dynamic_bundle_set_archive,
        pack_dynamic_bundle_set_archive,
    )

    bundle = _compile_family(tmp_path)
    archive = tmp_path / "family.ttca"
    pack_dynamic_bundle_set_archive(bundle, archive)

    child_manifest = json.loads(
        (bundle / "variants" / "0000" / "manifest.json").read_text(encoding="utf-8")
    )
    library_entry = f"bundle/variants/0000/{child_manifest['library']}"
    rewritten = tmp_path / "tampered.ttca"
    with zipfile.ZipFile(archive, "r") as source, zipfile.ZipFile(
        rewritten, "w", compression=zipfile.ZIP_STORED
    ) as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == library_entry:
                data += b"tamper"
            target.writestr(info, data)

    with pytest.raises(NativeBundleArchiveError, match="payload failed bundle verification"):
        load_dynamic_bundle_set_archive(rewritten)


def test_dynamic_bundle_archive_rejects_unsafe_paths_before_extraction(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_archive import (
        NativeBundleArchiveError,
        load_dynamic_bundle_set_archive,
    )

    archive = tmp_path / "escape.ttca"
    outside = tmp_path / "escape.txt"
    descriptor = json.dumps(
        {
            "kind": "dynamic-bundle-set",
            "root": "bundle",
            "schema": "native-bundle-archive-v1",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as packed:
        packed.writestr("archive.json", descriptor)
        packed.writestr("bundle/../escape.txt", b"escape")

    with pytest.raises(NativeBundleArchiveError, match="escapes its payload root"):
        load_dynamic_bundle_set_archive(archive)
    assert not outside.exists()


def test_dynamic_bundle_archive_rejects_duplicate_names(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_archive import (
        NativeBundleArchiveError,
        load_dynamic_bundle_set_archive,
    )

    archive = tmp_path / "duplicate.ttca"
    descriptor = json.dumps(
        {
            "kind": "dynamic-bundle-set",
            "root": "bundle",
            "schema": "native-bundle-archive-v1",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as packed:
            packed.writestr("archive.json", descriptor)
            packed.writestr("archive.json", descriptor)
            packed.writestr("bundle/manifest.json", b"{}")

    with pytest.raises(NativeBundleArchiveError, match="duplicate entry names"):
        load_dynamic_bundle_set_archive(archive)


def test_dynamic_bundle_archive_rejects_symlink_entries(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_archive import (
        NativeBundleArchiveError,
        load_dynamic_bundle_set_archive,
    )

    archive = tmp_path / "symlink.ttca"
    descriptor = json.dumps(
        {
            "kind": "dynamic-bundle-set",
            "root": "bundle",
            "schema": "native-bundle-archive-v1",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    link = zipfile.ZipInfo("bundle/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    link.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as packed:
        packed.writestr("archive.json", descriptor)
        packed.writestr(link, b"../outside")

    with pytest.raises(NativeBundleArchiveError, match="symbolic links"):
        load_dynamic_bundle_set_archive(archive)


def test_dynamic_bundle_archive_rejects_invalid_transport_schema(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_archive import (
        NativeBundleArchiveError,
        load_dynamic_bundle_set_archive,
    )

    archive = tmp_path / "schema.ttca"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as packed:
        packed.writestr(
            "archive.json",
            json.dumps(
                {
                    "kind": "dynamic-bundle-set",
                    "root": "bundle",
                    "schema": "native-bundle-archive-v2",
                }
            ),
        )
        packed.writestr("bundle/manifest.json", b"{}")

    with pytest.raises(NativeBundleArchiveError, match="unsupported.*schema"):
        load_dynamic_bundle_set_archive(archive)


def test_dynamic_bundle_archive_pack_rejects_corrupt_source_atomically(
    tmp_path: Path,
) -> None:
    from tiny_tensor_compiler.native_bundle_archive import (
        NativeBundleArchiveError,
        pack_dynamic_bundle_set_archive,
    )

    bundle = _compile_family(tmp_path)
    child_manifest = json.loads(
        (bundle / "variants" / "0000" / "manifest.json").read_text(encoding="utf-8")
    )
    library = bundle / "variants" / "0000" / child_manifest["library"]
    library.write_bytes(library.read_bytes() + b"tamper")
    archive = tmp_path / "family.ttca"

    with pytest.raises(NativeBundleArchiveError, match="source bundle set failed verification"):
        pack_dynamic_bundle_set_archive(bundle, archive)
    assert not archive.exists()
    assert not tuple(tmp_path.glob(".family.ttca.build-*.tmp"))


def test_dynamic_bundle_archive_destination_must_not_exist(tmp_path: Path) -> None:
    from tiny_tensor_compiler.native_bundle_archive import pack_dynamic_bundle_set_archive

    bundle = _compile_family(tmp_path)
    archive = tmp_path / "family.ttca"
    archive.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        pack_dynamic_bundle_set_archive(bundle, archive)
