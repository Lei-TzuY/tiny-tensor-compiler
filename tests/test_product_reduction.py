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
    execute_cpu,
    execute_loop,
    execute_reference,
    fuse_elementwise,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
)
from tiny_tensor_compiler.inference import TypeInferenceError
from tiny_tensor_compiler.reduction import ReductionOperator
from tiny_tensor_compiler.repro import capture_repro_case, load_repro_case, replay_repro_case
from tiny_tensor_compiler.serialization import deserialize_module, serialize_module


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _left_fold_product(value: np.ndarray) -> np.ndarray:
    accumulator = value.dtype.type(1)
    for index in np.ndindex(value.shape):
        accumulator = value.dtype.type(np.multiply(accumulator, value[index]))
    return np.array(accumulator, dtype=value.dtype)


def _left_fold_product_axis(value: np.ndarray, axis: int) -> np.ndarray:
    axis %= value.ndim
    output_shape = value.shape[:axis] + value.shape[axis + 1 :]
    output = np.empty(output_shape, dtype=value.dtype)
    for output_index in np.ndindex(output_shape):
        accumulator = value.dtype.type(1)
        for reduction_index in range(value.shape[axis]):
            input_index = output_index[:axis] + (reduction_index,) + output_index[axis:]
            accumulator = value.dtype.type(np.multiply(accumulator, value[input_index]))
        output[output_index] = accumulator
    return output


def test_product_is_typed_ir_for_full_and_single_axis_domains() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="int32")
    axis_product = value.prod(axis=1)
    negative_axis_product = value.prod(axis=-1)
    full_product = value.prod()
    module = builder.finish((axis_product, negative_axis_product, full_product))

    assert axis_product.type.shape == (2, 4)
    assert negative_axis_product.type.shape == (2, 3)
    assert full_product.type.shape == ()
    assert axis_product.type.dtype == value.type.dtype

    products = [op for op in module.function.ops if op.opcode == "prod"]
    assert [op.attrs for op in products] == [{"axis": 1}, {"axis": 2}, {}]


def test_product_rejects_bool_and_out_of_range_axes() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="float32")

    with pytest.raises(TypeInferenceError, match="prod axis"):
        value.prod(axis=True)
    with pytest.raises(TypeInferenceError, match="prod axis"):
        value.prod(axis=2)
    with pytest.raises(TypeInferenceError, match="prod axis"):
        value.prod(axis=-3)


def test_reference_product_is_ordered_same_dtype_and_uses_one_identity() -> None:
    builder = GraphBuilder()
    integer = builder.input((2, 4), dtype="int32")
    floating = builder.input((2, 3), dtype="float32")
    empty = builder.input((3, 0, 2), dtype="float64")
    module = builder.finish((integer.prod(), floating.prod(axis=0), empty.prod(axis=1)))

    integer_value = np.array(
        [[2**30, 4, -3, 5], [7, -8, 9, 10]],
        dtype=np.int32,
    )
    floating_value = np.array(
        [[1.0e20, 1.0e-20, 3.0], [1.0e-20, 1.0e20, -2.0]],
        dtype=np.float32,
    )
    empty_value = np.empty((3, 0, 2), dtype=np.float64)

    actual = execute_reference(module, inputs=[integer_value, floating_value, empty_value])
    assert isinstance(actual, tuple)
    np.testing.assert_array_equal(actual[0], _left_fold_product(integer_value))
    np.testing.assert_array_equal(actual[1], _left_fold_product_axis(floating_value, 0))
    np.testing.assert_array_equal(actual[2], np.ones((3, 2), dtype=np.float64))


def test_reduction_plan_is_shared_metadata_and_product_is_fusion_boundary() -> None:
    builder = GraphBuilder()
    value = builder.input((3, 4, 5), dtype="int64")
    reduced = value.prod(axis=1)
    module = builder.finish(reduced.relu())

    cpu = lower_to_cpu(module)
    loops = lower_to_loops(cpu)
    fused = fuse_elementwise(loops)

    buffer_product = next(kernel for kernel in cpu.instructions if kernel.opcode == "prod")
    loop_product = next(kernel for kernel in loops.kernels if kernel.opcode == "prod")
    assert buffer_product.reduction is not None
    assert loop_product.reduction is not None
    assert buffer_product.reduction.operator is ReductionOperator.PRODUCT
    assert loop_product.reduction.operator is ReductionOperator.PRODUCT
    assert buffer_product.reduction.axis == 1
    assert loop_product.reduction.axis == 1
    # Compatibility view retained for the existing sum-oriented lowering surface.
    assert buffer_product.reduction_axis == 1
    assert loop_product.reduction_axis == 1
    assert loop_product.iteration_shape == (3, 5)
    assert loop_product.input_maps == ()
    assert [kernel.opcode for kernel in fused.kernels] == ["prod", "relu"]


def test_product_reads_signed_stride_views_in_logical_order() -> None:
    builder = GraphBuilder()
    value = builder.input((3, 4, 5), dtype="float32")
    viewed = value.transpose((2, 0, 1)).reverse(2).slice(axis=1, start=0, stop=3, step=2)
    module = builder.finish(viewed.prod(axis=2))

    runtime = (np.arange(60, dtype=np.float32).reshape(3, 4, 5) - 20.0) / 10.0
    expected_view = runtime.transpose(2, 0, 1)[:, 0:3:2, ::-1]
    expected = _left_fold_product_axis(expected_view, 2)

    cpu = lower_to_cpu(module)
    np.testing.assert_array_equal(execute_reference(module, inputs=[runtime]), expected)
    np.testing.assert_array_equal(execute_cpu(cpu, inputs=[runtime]), expected)
    np.testing.assert_array_equal(execute_loop(lower_to_loops(cpu), inputs=[runtime]), expected)


def test_generated_c_uses_shared_reduction_shape_and_parallelizes_only_outputs() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="float32")
    loops = lower_to_loops(lower_to_cpu(builder.finish(value.prod(axis=1))))

    serial = generate_c(loops)
    parallel = generate_c(loops, parallel=True)
    for source in (serial, parallel):
        assert "float prod_value = 1.0f;" in source
        assert "for (int64_t r = 0; r < 3; ++r)" in source
        assert "prod_value = ((float)prod_value *" in source
    assert "#pragma omp parallel for schedule(static)" not in serial
    assert parallel.count("#pragma omp parallel for schedule(static)") == 1

    builder = GraphBuilder()
    vector = builder.input((7,), dtype="float32")
    scalar_loops = lower_to_loops(lower_to_cpu(builder.finish(vector.prod(axis=0))))
    scalar_parallel = generate_c(scalar_loops, parallel=True)
    assert "float prod_value = 1.0f;" in scalar_parallel
    assert "#pragma omp parallel for schedule(static)" not in scalar_parallel


def test_static_native_product_composes_with_views_borrowing_parallel_and_multi_output() -> None:
    _default_compiler_or_skip()
    builder = GraphBuilder()
    value = builder.input((4, 5, 6), dtype="int32")
    viewed = value.transpose((2, 0, 1)).reverse(2)
    module = builder.finish((viewed.prod(axis=1), viewed.prod(axis=2), viewed.prod()))
    runtime = (np.arange(120, dtype=np.int32).reshape(4, 5, 6) % 7) - 3

    executable = compile_module(module, borrow_inputs=True, parallel=True)
    actual = executable(inputs=[runtime])
    expected = execute_reference(module, inputs=[runtime])

    assert isinstance(actual, tuple)
    assert isinstance(expected, tuple)
    for result, wanted in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, wanted)


def test_dynamic_native_product_preserves_unreduced_symbolic_extent_and_cache() -> None:
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    value = builder.input((batch, 3, 4), dtype="float32")
    module = builder.finish(value.reverse(2).prod(axis=1))
    executable = compile_dynamic_module(module, borrow_inputs=True, parallel=True)

    for batch_size in (0, 2, 5, 2):
        runtime = ((np.arange(batch_size * 12, dtype=np.float32) % 7) - 3).reshape(
            batch_size, 3, 4
        )
        actual = executable(inputs=[runtime])
        expected = _left_fold_product_axis(runtime[:, :, ::-1], 1)
        np.testing.assert_array_equal(actual, expected)

    assert executable.cached_batch_sizes == (0, 2, 5)


def test_product_participates_in_dce_and_axis_sensitive_cse_without_merging_sum() -> None:
    builder = GraphBuilder()
    value = builder.input((3, 3), dtype="int32")
    value.prod(axis=0)
    module = builder.finish(value.relu())
    assert dead_code_eliminate(module) == 1
    assert all(op.opcode != "prod" for op in module.function.ops)

    builder = GraphBuilder()
    value = builder.input((3, 3), dtype="int32")
    axis0_a = value.prod(axis=0)
    axis0_b = value.prod(axis=0)
    axis1 = value.prod(axis=1)
    sum_axis0 = value.sum(axis=0)
    module = builder.finish((axis0_a, axis0_b, axis1, sum_axis0))
    assert common_subexpression_eliminate(module) == 1
    products = [op for op in module.function.ops if op.opcode == "prod"]
    sums = [op for op in module.function.ops if op.opcode == "sum"]
    assert [op.attrs for op in products] == [{"axis": 0}, {"axis": 1}]
    assert [op.attrs for op in sums] == [{"axis": 0}]


def test_product_round_trips_through_canonical_ir_and_repro_replay() -> None:
    _default_compiler_or_skip()
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="int32")
    module = builder.finish((value.prod(axis=1), value.reverse(2).prod()))
    runtime = (np.arange(24, dtype=np.int32).reshape(2, 3, 4) % 5) - 2

    document = serialize_module(module)
    restored = deserialize_module(document)
    assert serialize_module(restored) == document
    assert [op.attrs for op in restored.function.ops if op.opcode == "prod"] == [
        {"axis": 1},
        {},
    ]

    repro = capture_repro_case(restored, inputs=[runtime])
    case = load_repro_case(repro)
    assert [op.attrs for op in case.module.function.ops if op.opcode == "prod"] == [
        {"axis": 1},
        {},
    ]
    reference = replay_repro_case(repro, backend="reference")
    native = replay_repro_case(repro, backend="native", parallel=True)
    assert isinstance(reference, tuple)
    assert isinstance(native, tuple)
    for actual, expected in zip(native, reference, strict=True):
        np.testing.assert_array_equal(actual, expected)
