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


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _left_fold_axis(value: np.ndarray, axis: int) -> np.ndarray:
    axis = axis % value.ndim
    output_shape = value.shape[:axis] + value.shape[axis + 1 :]
    output = np.empty(output_shape, dtype=value.dtype)
    for output_index in np.ndindex(output_shape):
        accumulator = value.dtype.type(0)
        for reduction_index in range(value.shape[axis]):
            input_index = output_index[:axis] + (reduction_index,) + output_index[axis:]
            accumulator = value.dtype.type(np.add(accumulator, value[input_index]))
        output[output_index] = accumulator
    return output


def test_axis_sum_is_typed_ir_and_normalizes_negative_axis() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="int32")
    axis_sum = value.sum(axis=1)
    negative_axis_sum = value.sum(axis=-1)
    full_sum = value.sum()
    module = builder.finish((axis_sum, negative_axis_sum, full_sum))

    assert axis_sum.type.shape == (2, 4)
    assert negative_axis_sum.type.shape == (2, 3)
    assert full_sum.type.shape == ()
    assert axis_sum.type.dtype == value.type.dtype

    sums = [op for op in module.function.ops if op.opcode == "sum"]
    assert sums[0].attrs == {"axis": 1}
    assert sums[1].attrs == {"axis": 2}
    assert sums[2].attrs == {}


def test_axis_sum_rejects_bool_and_out_of_range_axes() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="float32")

    with pytest.raises(TypeInferenceError, match="axis"):
        value.sum(axis=True)
    with pytest.raises(TypeInferenceError, match="axis"):
        value.sum(axis=2)
    with pytest.raises(TypeInferenceError, match="axis"):
        value.sum(axis=-3)


def test_reference_axis_sum_is_deterministic_left_fold_and_empty_identity() -> None:
    builder = GraphBuilder()
    integer = builder.input((2, 4), dtype="int32")
    floating = builder.input((2, 3), dtype="float32")
    empty = builder.input((3, 0, 2), dtype="float64")
    module = builder.finish((integer.sum(axis=1), floating.sum(axis=0), empty.sum(axis=1)))

    integer_value = np.array([[2**31 - 1, 1, -3, 5], [7, -8, 9, 10]], dtype=np.int32)
    floating_value = np.array(
        [[1.0e20, 1.0, -1.0e20], [-1.0e20, 3.0, 1.0e20]],
        dtype=np.float32,
    )
    empty_value = np.empty((3, 0, 2), dtype=np.float64)

    actual = execute_reference(module, inputs=[integer_value, floating_value, empty_value])
    assert isinstance(actual, tuple)
    np.testing.assert_array_equal(actual[0], _left_fold_axis(integer_value, 1))
    np.testing.assert_array_equal(actual[1], _left_fold_axis(floating_value, 0))
    np.testing.assert_array_equal(actual[2], np.zeros((3, 2), dtype=np.float64))


def test_axis_sum_metadata_survives_lowering_and_remains_fusion_boundary() -> None:
    builder = GraphBuilder()
    value = builder.input((3, 4, 5), dtype="int64")
    reduced = value.sum(axis=1)
    module = builder.finish(reduced.relu())

    cpu = lower_to_cpu(module)
    loops = lower_to_loops(cpu)
    fused = fuse_elementwise(loops)

    buffer_sum = next(kernel for kernel in cpu.instructions if kernel.opcode == "sum")
    loop_sum = next(kernel for kernel in loops.kernels if kernel.opcode == "sum")
    assert buffer_sum.reduction_axis == 1
    assert loop_sum.reduction_axis == 1
    assert loop_sum.iteration_shape == (3, 5)
    assert loop_sum.input_maps == ()
    assert loop_sum.output != loop_sum.inputs[0]
    assert [kernel.opcode for kernel in fused.kernels] == ["sum", "relu"]


def test_axis_sum_reads_signed_stride_views_in_logical_axis_order() -> None:
    builder = GraphBuilder()
    value = builder.input((3, 4, 5), dtype="float32")
    viewed = value.transpose((2, 0, 1)).reverse(2).slice(axis=1, start=0, stop=3, step=2)
    module = builder.finish(viewed.sum(axis=2))

    runtime = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    expected_view = runtime.transpose(2, 0, 1)[:, 0:3:2, ::-1]
    expected = _left_fold_axis(expected_view, 2)

    cpu = lower_to_cpu(module)
    np.testing.assert_array_equal(execute_reference(module, inputs=[runtime]), expected)
    np.testing.assert_array_equal(execute_cpu(cpu, inputs=[runtime]), expected)
    np.testing.assert_array_equal(execute_loop(lower_to_loops(cpu), inputs=[runtime]), expected)


def test_generated_c_parallelizes_only_independent_axis_sum_outputs() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="float32")
    loops = lower_to_loops(lower_to_cpu(builder.finish(value.sum(axis=1))))

    serial = generate_c(loops)
    parallel = generate_c(loops, parallel=True)

    assert "for (int64_t i0 = 0; i0 < 2; ++i0)" in serial
    assert "for (int64_t i1 = 0; i1 < 4; ++i1)" in serial
    assert "for (int64_t r = 0; r < 3; ++r)" in serial
    assert "#pragma omp parallel for schedule(static)" not in serial
    assert "#pragma omp parallel for schedule(static)" in parallel
    assert parallel.count("#pragma omp parallel for schedule(static)") == 1
    assert "for (int64_t r = 0; r < 3; ++r)" in parallel

    builder = GraphBuilder()
    vector = builder.input((7,), dtype="float32")
    scalar_loops = lower_to_loops(lower_to_cpu(builder.finish(vector.sum(axis=0))))
    scalar_parallel = generate_c(scalar_loops, parallel=True)
    assert "#pragma omp parallel for schedule(static)" not in scalar_parallel


def test_static_native_axis_sum_composes_with_views_borrowing_parallel_and_multi_output() -> None:
    _default_compiler_or_skip()
    builder = GraphBuilder()
    value = builder.input((4, 5, 6), dtype="int32")
    viewed = value.transpose((2, 0, 1)).reverse(2)
    first = viewed.sum(axis=1)
    second = viewed.sum(axis=2)
    module = builder.finish((first, second))
    runtime = np.arange(120, dtype=np.int32).reshape(4, 5, 6) - 60

    executable = compile_module(module, borrow_inputs=True, parallel=True)
    actual = executable(inputs=[runtime])
    expected = execute_reference(module, inputs=[runtime])

    assert isinstance(actual, tuple)
    assert isinstance(expected, tuple)
    for result, wanted in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, wanted)


def test_dynamic_native_axis_sum_preserves_unreduced_symbolic_extent_and_cache() -> None:
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    value = builder.input((batch, 3, 4), dtype="float32")
    module = builder.finish(value.reverse(2).sum(axis=1))
    executable = compile_dynamic_module(module, borrow_inputs=True, parallel=True)

    for batch_size in (0, 2, 5, 2):
        runtime = np.arange(batch_size * 12, dtype=np.float32).reshape(batch_size, 3, 4)
        actual = executable(inputs=[runtime])
        expected = _left_fold_axis(runtime[:, :, ::-1], 1)
        np.testing.assert_array_equal(actual, expected)

    assert executable.cached_batch_sizes == (0, 2, 5)


def test_axis_sum_participates_in_dce_and_axis_sensitive_cse() -> None:
    builder = GraphBuilder()
    value = builder.input((3, 3), dtype="int32")
    value.sum(axis=0)
    module = builder.finish(value.relu())
    assert dead_code_eliminate(module) == 1
    assert all(op.opcode != "sum" for op in module.function.ops)

    builder = GraphBuilder()
    value = builder.input((3, 3), dtype="int32")
    axis0_a = value.sum(axis=0)
    axis0_b = value.sum(axis=0)
    axis1 = value.sum(axis=1)
    module = builder.finish((axis0_a, axis0_b, axis1))
    assert common_subexpression_eliminate(module) == 1
    sums = [op for op in module.function.ops if op.opcode == "sum"]
    assert [op.attrs for op in sums] == [{"axis": 0}, {"axis": 1}]
