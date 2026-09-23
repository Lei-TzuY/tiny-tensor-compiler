import json
from pathlib import Path

import numpy as np
import pytest

from tiny_tensor_compiler.frontend import GraphBuilder
from tiny_tensor_compiler.ir import SymbolicDim
from tiny_tensor_compiler.native_linearization_bundle_set import (
    NativeLinearizationBundleSetError,
    compile_dynamic_linearization_bundle_set,
    load_dynamic_linearization_bundle_set,
)


def _dynamic_quartic_module():
    batch = SymbolicDim("B")
    builder = GraphBuilder("packaged-linearization")
    x = builder.input((batch, 3), dtype="float64")
    squared = x * x
    return batch, builder.finish((squared * squared).sum())


def _input(batch: int) -> np.ndarray:
    return (
        np.arange(batch * 3, dtype=np.float64).reshape(batch, 3) * 0.25
        - 1.5
    )


def test_linearization_bundle_set_executes_compiler_free_retained_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    batch, module = _dynamic_quartic_module()
    bundle = tmp_path / "linearizations.ttclin"
    compile_dynamic_linearization_bundle_set(
        module,
        ({batch: 2}, {batch: 5}),
        bundle,
        wrt=(0,),
    )

    import tiny_tensor_compiler.native_linearization_bundle_set as bundle_module

    def fail_compile(*_args, **_kwargs):
        raise AssertionError("deployment-side linearization unexpectedly invoked compiler")

    monkeypatch.setattr(bundle_module, "compile_native_bundle", fail_compile)

    executable = load_dynamic_linearization_bundle_set(bundle)
    try:
        assert executable.available_bindings == (
            (("B", 2),),
            (("B", 5),),
        )
        assert executable.loaded_bindings == ()

        values = _input(2)
        frozen = values.copy()
        state = executable.linearize((values,))
        assert executable.loaded_bindings == ((("B", 2),),)
        np.testing.assert_allclose(
            state.primal,
            np.array(np.sum(frozen**4), dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )

        values[:] = 1234.0
        tangent = np.arange(6, dtype=np.float64).reshape(2, 3) * -0.5 + 2.0
        cotangent = np.array(0.75, dtype=np.float64)
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
        assert state.pushforward_query_count == 1
        assert state.pullback_query_count == 1

        second = executable.specialize({"B": 5})
        values5 = _input(5)
        state5 = second.linearize((values5,))
        np.testing.assert_allclose(
            state5.primal,
            np.array(np.sum(values5**4), dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )
        assert executable.loaded_bindings == ((("B", 2),), (("B", 5),))
    finally:
        executable.close()


def test_linearization_bundle_set_rejects_unbundled_runtime_shape(tmp_path: Path) -> None:
    batch, module = _dynamic_quartic_module()
    bundle = tmp_path / "linearizations.ttclin"
    compile_dynamic_linearization_bundle_set(module, ({batch: 2},), bundle, wrt=(0,))
    executable = load_dynamic_linearization_bundle_set(bundle)
    try:
        with pytest.raises(
            NativeLinearizationBundleSetError,
            match="does not match any packaged",
        ):
            executable.linearize((_input(3),))
    finally:
        executable.close()


def test_linearization_bundle_set_rejects_component_role_substitution(
    tmp_path: Path,
) -> None:
    batch, module = _dynamic_quartic_module()
    bundle = tmp_path / "linearizations.ttclin"
    compile_dynamic_linearization_bundle_set(module, ({batch: 2},), bundle, wrt=(0,))

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    components = manifest["variants"][0]["components"]
    components["pushforward"], components["pullback"] = (
        components["pullback"],
        components["pushforward"],
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        NativeLinearizationBundleSetError,
        match="component ABI contract",
    ):
        load_dynamic_linearization_bundle_set(bundle)
