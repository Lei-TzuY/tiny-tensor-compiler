import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    SymbolicDim,
    common_subexpression_eliminate,
    compile_dynamic_module,
    compile_module,
    dead_code_eliminate,
    execute_loop,
    execute_reference,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
)
from tiny_tensor_compiler.inference import TypeInferenceError
from tiny_tensor_compiler.repro import capture_repro_case, replay_repro_case
from tiny_tensor_compiler.serialization import deserialize_module, serialize_module


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def test_argmax_builds_index_typed_ir_and_normalizes_axis() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="float32")
    full = value.argmax()
    axis = value.argmax(axis=-2)
    kept = value.argmax(axis=2, keepdims=True)
    module = builder.finish((full, axis, kept))

    assert full.type.shape == ()
    assert full.type.dtype.value == "i64"
    assert axis.type.shape == (2, 4)
    assert axis.type.dtype.value == "i64"
    assert kept.type.shape == (2, 3, 1)
    assert [(op.opcode, op.attrs) for op in module.function.ops] == [
        ("input", {"index": 0}),
        ("argmax", {}),
        ("argmax", {"axis": 1}),
        ("argmax", {"axis": 2}),
        ("view", {}),
        ("return", {}),
    ]


def test_argmax_rejects_multi_axis_invalid_keepdims_and_known_empty_domains() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="float32")
    with pytest.raises(TypeInferenceError, match="integer or None"):
        value.argmax(axis=(0, 1))  # type: ignore[arg-type]
    with pytest.raises(TypeInferenceError, match="axis"):
        value.argmax(axis=True)  # type: ignore[arg-type]
    with pytest.raises(TypeInferenceError, match="keepdims"):
        value.argmax(axis=1, keepdims=1)  # type: ignore[arg-type]

    empty = builder.input((2, 0, 3), dtype="int32")
    with pytest.raises(TypeInferenceError, match="must not be empty"):
        empty.argmax()
    with pytest.raises(TypeInferenceError, match="must not be empty"):
        empty.argmax(axis=1)
    assert empty.argmax(axis=0).type.shape == (0, 3)


def test_argmax_reference_and_loop_match_first_tie_and_nan_policy() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 5), dtype="float32")
    module = builder.finish((value.argmax(), value.argmax(axis=1)))
    runtime = np.array(
        [
            [1.0, 9.0, 9.0, np.nan, np.nan],
            [4.0, -0.0, 0.0, 4.0, 3.0],
        ],
        dtype=np.float32,
    )

    reference = execute_reference(module, inputs=[runtime])
    loop = execute_loop(lower_to_loops(lower_to_cpu(module)), inputs=[runtime])
    expected = (np.asarray(np.argmax(runtime), dtype=np.int64), np.argmax(runtime, axis=1))
    assert isinstance(reference, tuple)
    assert isinstance(loop, tuple)
    for actual, wanted in zip(reference, expected, strict=True):
        np.testing.assert_array_equal(actual, wanted)
    for actual, wanted in zip(loop, expected, strict=True):
        np.testing.assert_array_equal(actual, wanted)


def test_argmax_generated_c_has_first_nan_selection_and_i64_output() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 4), dtype="float32")
    module = builder.finish(value.argmax(axis=1))
    source = generate_c(lower_to_loops(lower_to_cpu(module)))

    assert "int64_t *out" in source
    assert "float argmax_best" in source
    assert "int64_t argmax_index = 0;" in source
    assert "for (r = 1; r < 4; ++r)" in source
    assert "!isnan(argmax_best)" in source
    assert "isnan(argmax_candidate) || argmax_candidate > argmax_best" in source


def test_argmax_native_handles_strided_views_borrowed_inputs_multi_output_and_openmp() -> None:
    _default_compiler_or_skip()
    builder = GraphBuilder()
    value = builder.input((3, 4, 5), dtype="float32")
    viewed = value.transpose((2, 0, 1)).reverse(2)
    module = builder.finish((viewed.argmax(), viewed.argmax(axis=1, keepdims=True)))
    runtime = ((np.arange(60, dtype=np.float32).reshape(3, 4, 5) % 11) - 5).astype(
        np.float32
    )
    runtime[1, 2, 3] = np.nan

    native = compile_module(module, borrow_inputs=True, parallel=True)(inputs=[runtime])
    reference = execute_reference(module, inputs=[runtime])
    assert isinstance(native, tuple)
    assert isinstance(reference, tuple)
    for actual, expected in zip(native, reference, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_dynamic_argmax_specializes_nonempty_domains_and_rejects_zero_axis() -> None:
    _default_compiler_or_skip()
    width = SymbolicDim("W")
    builder = GraphBuilder()
    value = builder.input((2, width), dtype="int32")
    module = builder.finish(value.argmax(axis=1, keepdims=True))
    executable = compile_dynamic_module(module, borrow_inputs=True, parallel=True)

    for width_value in (1, 5, 2, 5):
        runtime = np.arange(2 * width_value, dtype=np.int32).reshape(2, width_value)
        actual = executable(inputs=[runtime])
        expected = np.argmax(runtime, axis=1, keepdims=True)
        np.testing.assert_array_equal(actual, expected)

    assert executable.cached_bindings == ((('W', 1),), (('W', 2),), (('W', 5),))
    with pytest.raises(ValueError, match="argmax reduction domain must not be empty"):
        executable(inputs=[np.empty((2, 0), dtype=np.int32)])


def test_argmax_participates_in_dce_exact_cse_serialization_and_repro() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="int32")
    value.argmax(axis=1)
    module = builder.finish(value.relu())
    assert dead_code_eliminate(module) == 1

    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="float64")
    first = value.argmax(axis=-1)
    second = value.argmax(axis=1)
    module = builder.finish((first, second))
    assert common_subexpression_eliminate(module) == 1
    assert len([op for op in module.function.ops if op.opcode == "argmax"]) == 1

    document = serialize_module(module)
    restored = deserialize_module(document)
    assert serialize_module(restored) == document

    runtime = np.array([[1.0, 7.0, 7.0], [np.nan, 3.0, np.nan]], dtype=np.float64)
    repro = capture_repro_case(restored, inputs=[runtime])
    reference = replay_repro_case(repro, backend="reference")
    _default_compiler_or_skip()
    native = replay_repro_case(repro, backend="native", parallel=True)
    assert isinstance(reference, tuple)
    assert isinstance(native, tuple)
    for actual, expected in zip(native, reference, strict=True):
        np.testing.assert_array_equal(actual, expected)
