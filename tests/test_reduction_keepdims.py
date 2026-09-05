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
    lower_to_cpu,
    lower_to_loops,
)
from tiny_tensor_compiler.inference import TypeInferenceError
from tiny_tensor_compiler.repro import capture_repro_case, load_repro_case, replay_repro_case
from tiny_tensor_compiler.serialization import deserialize_module, serialize_module


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def test_keepdims_is_a_canonical_reduction_then_verified_view() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="int32")
    summed = value.sum(axis=(2, 0), keepdims=True)
    product = value.prod(keepdims=True)
    module = builder.finish((summed, product))

    assert summed.type.shape == (1, 3, 1)
    assert product.type.shape == (1, 1, 1)
    ops = module.function.ops
    assert [(op.opcode, op.attrs) for op in ops] == [
        ("input", {"index": 0}),
        ("sum", {"axis": (0, 2)}),
        ("view", {}),
        ("prod", {}),
        ("view", {}),
        ("return", {}),
    ]
    assert ops[1].results[0].type.shape == (3,)
    assert ops[3].results[0].type.shape == ()


def test_keepdims_false_preserves_historical_reduction_ir() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="float32")
    module = builder.finish((value.sum(axis=(0, 2)), value.prod()))

    reductions = [op for op in module.function.ops if op.opcode in {"sum", "prod"}]
    assert [op.attrs for op in reductions] == [{"axis": (0, 2)}, {}]
    assert [op.results[0].type.shape for op in reductions] == [(3,), ()]
    assert all(op.opcode != "view" for op in module.function.ops)


def test_keepdims_requires_an_actual_bool() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="float32")
    for invalid in (1, 0, None, np.bool_(True), "true"):
        with pytest.raises(TypeInferenceError, match="keepdims"):
            value.sum(axis=1, keepdims=invalid)  # type: ignore[arg-type]
        with pytest.raises(TypeInferenceError, match="keepdims"):
            value.prod(axis=0, keepdims=invalid)  # type: ignore[arg-type]


def test_keepdims_reference_and_loop_preserve_logical_view_order_and_broadcasting() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="float32")
    viewed = value.transpose((2, 0, 1)).reverse(2)
    kept_sum = viewed.sum(axis=(0, 2), keepdims=True)
    kept_product = viewed.prod(axis=1, keepdims=True)
    module = builder.finish((kept_sum + viewed, kept_product))

    runtime = ((np.arange(24, dtype=np.float32).reshape(2, 3, 4) % 7) - 3) / 2.0
    logical = runtime.transpose(2, 0, 1)[:, :, ::-1]
    expected_sum = np.sum(logical, axis=(0, 2), keepdims=True, dtype=np.float32) + logical
    expected_product = np.prod(logical, axis=1, keepdims=True, dtype=np.float32)

    reference = execute_reference(module, inputs=[runtime])
    loop = execute_loop(lower_to_loops(lower_to_cpu(module)), inputs=[runtime])
    assert isinstance(reference, tuple)
    assert isinstance(loop, tuple)
    np.testing.assert_array_equal(reference[0], expected_sum)
    np.testing.assert_array_equal(reference[1], expected_product)
    np.testing.assert_array_equal(loop[0], expected_sum)
    np.testing.assert_array_equal(loop[1], expected_product)


def test_keepdims_native_composes_with_views_borrowed_inputs_multi_output_and_openmp() -> None:
    _default_compiler_or_skip()
    builder = GraphBuilder()
    value = builder.input((3, 4, 5), dtype="int32")
    viewed = value.transpose((2, 0, 1)).reverse(2)
    module = builder.finish(
        (
            viewed.sum(axis=(0, 2), keepdims=True),
            viewed.prod(axis=1, keepdims=True),
        )
    )
    runtime = (np.arange(60, dtype=np.int32).reshape(3, 4, 5) % 5) - 2

    native = compile_module(module, borrow_inputs=True, parallel=True)(inputs=[runtime])
    reference = execute_reference(module, inputs=[runtime])
    assert isinstance(native, tuple)
    assert isinstance(reference, tuple)
    for actual, expected in zip(native, reference, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_dynamic_keepdims_retains_unreduced_symbolic_axis_and_reuses_specialization() -> None:
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    value = builder.input((batch, 3, 4), dtype="float32")
    module = builder.finish(value.sum(axis=(1, 2), keepdims=True))
    executable = compile_dynamic_module(module, borrow_inputs=True, parallel=True)

    for batch_size in (0, 2, 5, 2):
        runtime = ((np.arange(batch_size * 12, dtype=np.float32) % 9) - 4).reshape(
            batch_size, 3, 4
        )
        actual = executable(inputs=[runtime])
        expected = np.sum(runtime, axis=(1, 2), keepdims=True, dtype=np.float32)
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (batch_size, 1, 1)

    assert executable.cached_batch_sizes == (0, 2, 5)


def test_keepdims_composition_participates_in_dce_and_exact_cse() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="int32")
    value.sum(axis=(0, 2), keepdims=True)
    module = builder.finish(value.relu())
    assert dead_code_eliminate(module) == 2

    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="int32")
    first = value.prod(axis=(2, 0), keepdims=True)
    second = value.prod(axis=(0, 2), keepdims=True)
    module = builder.finish((first, second))
    assert common_subexpression_eliminate(module) == 2
    assert len([op for op in module.function.ops if op.opcode == "prod"]) == 1
    assert len([op for op in module.function.ops if op.opcode == "view"]) == 1


def test_keepdims_round_trips_through_serialization_and_repro() -> None:
    _default_compiler_or_skip()
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="int32")
    module = builder.finish(
        (
            value.sum(axis=(2, 0), keepdims=True),
            value.reverse(2).prod(axis=1, keepdims=True),
        )
    )
    runtime = (np.arange(24, dtype=np.int32).reshape(2, 3, 4) % 5) - 2

    document = serialize_module(module)
    restored = deserialize_module(document)
    assert serialize_module(restored) == document
    assert [op.opcode for op in restored.function.ops].count("view") == 2

    repro = capture_repro_case(restored, inputs=[runtime])
    case = load_repro_case(repro)
    assert [op.opcode for op in case.module.function.ops].count("view") == 2
    reference = replay_repro_case(repro, backend="reference")
    native = replay_repro_case(repro, backend="native", parallel=True)
    assert isinstance(reference, tuple)
    assert isinstance(native, tuple)
    for actual, expected in zip(native, reference, strict=True):
        np.testing.assert_array_equal(actual, expected)
