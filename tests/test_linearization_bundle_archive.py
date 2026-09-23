from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tiny_tensor_compiler import GraphBuilder, SymbolicDim


def _dynamic_quartic_module():
    batch = SymbolicDim("B")
    builder = GraphBuilder("archived-linearization")
    x = builder.input((batch, 3), dtype="float64")
    squared = x * x
    return batch, builder.finish((squared * squared).sum())


def _input(batch: int) -> np.ndarray:
    return (
        np.arange(batch * 3, dtype=np.float64).reshape(batch, 3) * 0.25
        - 1.5
    )


def _compile_family(tmp_path: Path):
    from tiny_tensor_compiler.native_linearization_bundle_set import (
        compile_dynamic_linearization_bundle_set,
    )

    batch, module = _dynamic_quartic_module()
    bundle = tmp_path / "linearizations.ttclin"
    compile_dynamic_linearization_bundle_set(
        module,
        ({batch: 2}, {batch: 5}),
        bundle,
        wrt=(0,),
    )
    return bundle


def test_linearization_bundle_archive_is_deterministic_and_compiler_free(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tiny_tensor_compiler import native_bundle, native_bundle_archive

    bundle = _compile_family(tmp_path)
    first = tmp_path / "linearizations-a.ttcla"
    second = tmp_path / "linearizations-b.ttcla"
    native_bundle_archive.pack_dynamic_linearization_bundle_set_archive(
        bundle,
        first,
    )
    native_bundle_archive.pack_dynamic_linearization_bundle_set_archive(
        bundle,
        second,
    )

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as packed:
        assert packed.namelist()[0] == "archive.json"
        assert packed.namelist()[1:] == sorted(packed.namelist()[1:])
        assert all(
            info.compress_type == zipfile.ZIP_STORED
            for info in packed.infolist()
        )
        assert json.loads(packed.read("archive.json")) == {
            "kind": "retained-linearization-bundle-set",
            "root": "bundle",
            "schema": "native-bundle-archive-v1",
        }

    monkeypatch.setattr(
        native_bundle,
        "_compiler_command",
        lambda *_args, **_kwargs: (
            _ for _ in ()
        ).throw(AssertionError("compiler lookup")),
    )

    executable = native_bundle_archive.load_dynamic_linearization_bundle_set_archive(
        first
    )
    try:
        assert executable.available_bindings == (
            (("B", 2),),
            (("B", 5),),
        )
        assert executable.loaded_bindings == ()

        values = _input(2)
        frozen = values.copy()
        state = executable.linearize((values,))
        values[:] = 999.0
        tangent = np.arange(6, dtype=np.float64).reshape(2, 3) * -0.5 + 2.0
        cotangent = np.array(0.75, dtype=np.float64)

        np.testing.assert_allclose(
            state.primal,
            np.array(np.sum(frozen**4), dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            state.pushforward((tangent,)),
            np.array(np.sum(4.0 * frozen**3 * tangent), dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            state.pullback(cotangent),
            4.0 * frozen**3 * cotangent,
            rtol=0.0,
            atol=0.0,
        )
        assert executable.loaded_bindings == ((("B", 2),),)

        child = executable.specialize({"B": 5})
        state5 = child.linearize((_input(5),))
        np.testing.assert_allclose(
            state5.primal,
            np.array(np.sum(_input(5) ** 4), dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )
    finally:
        executable.close()

    assert executable.closed
    with pytest.raises(RuntimeError, match="archive executable is closed"):
        executable.linearize((_input(2),))


def test_linearization_bundle_archive_rejects_tampered_child_library(
    tmp_path: Path,
) -> None:
    from tiny_tensor_compiler.native_bundle_archive import (
        NativeBundleArchiveError,
        load_dynamic_linearization_bundle_set_archive,
        pack_dynamic_linearization_bundle_set_archive,
    )

    bundle = _compile_family(tmp_path)
    archive = tmp_path / "linearizations.ttcla"
    pack_dynamic_linearization_bundle_set_archive(bundle, archive)

    child_manifest = json.loads(
        (
            bundle
            / "variants"
            / "0000"
            / "primal_tape"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    library_entry = (
        "bundle/variants/0000/primal_tape/"
        + child_manifest["library"]
    )
    rewritten = tmp_path / "tampered.ttcla"
    with zipfile.ZipFile(archive, "r") as source, zipfile.ZipFile(
        rewritten,
        "w",
        compression=zipfile.ZIP_STORED,
    ) as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == library_entry:
                data += b"tamper"
            target.writestr(info, data)

    with pytest.raises(
        NativeBundleArchiveError,
        match="archive payload failed linearization bundle verification",
    ):
        load_dynamic_linearization_bundle_set_archive(rewritten)


def test_linearization_bundle_archive_rejects_wrong_payload_kind(
    tmp_path: Path,
) -> None:
    from tiny_tensor_compiler.native_bundle_archive import (
        NativeBundleArchiveError,
        load_dynamic_linearization_bundle_set_archive,
        pack_dynamic_linearization_bundle_set_archive,
    )

    bundle = _compile_family(tmp_path)
    archive = tmp_path / "linearizations.ttcla"
    pack_dynamic_linearization_bundle_set_archive(bundle, archive)

    rewritten = tmp_path / "wrong-kind.ttcla"
    with zipfile.ZipFile(archive, "r") as source, zipfile.ZipFile(
        rewritten,
        "w",
        compression=zipfile.ZIP_STORED,
    ) as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == "archive.json":
                descriptor = json.loads(data)
                descriptor["kind"] = "dynamic-bundle-set"
                data = (
                    json.dumps(
                        descriptor,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode()
            target.writestr(info, data)

    with pytest.raises(
        NativeBundleArchiveError,
        match="unsupported native bundle archive payload kind",
    ):
        load_dynamic_linearization_bundle_set_archive(rewritten)
