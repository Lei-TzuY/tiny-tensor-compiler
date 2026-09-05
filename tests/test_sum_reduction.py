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


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _left_fold_sum(value: np.ndarray) -> np.ndarray:
    dtype = value.dtype
    acc = dtype.type(0)
    for index in np.ndindex(value.shape):
        acc = dtype.type(np.add(acc, value[index]))
    return np.array(acc, dtype=dtype)


def test_sum_is_typed_scalar_ir_and_preserves_dtype():
    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="int32")
    reduced = value.sum()
    module = builder.finish(reduced)

    assert reduced.type.shape == ()
    assert reduced.type.dtype == value.type.dtype
    assert "%1 = sum %0" in module.dump()


def test_reference_sum_is_c_order_left_fold_with_fixed_width_and_empty_identity():
    builder = GraphBuilder()
    integer = builder.input((4,), dtype="int32")
    floating = builder.input((3,), dtype="float32")
    empty = builder.input((0, 7), dtype="float64")
    module = builder.finish((integer.sum(), floating.sum(), empty.sum()))

    integer_value = np.array([2**31 - 1, 1, -3, 5], dtype=np.int32)
    floating_value = np.array([1.0e20, -1.0e20, 3.0], dtype=np.float32)
    empty_value = np.empty((0, 7), dtype=np.float64)
    actual = execute_reference(module, inputs=[integer_value, floating_value, empty_value])

    assert isinstance(actual, tuple)
    np.testing.assert_array_equal(actual[0], _left_fold_sum(integer_value))
    np.testing.assert_array_equal(actual[1], _left_fold_sum(floating_value))
    np.testing.assert_array_equal(actual[2], np.array(0.0, dtype=np.float64))


def test_sum_lowers_to_scalar_reduction_kernel_and_is_fusion_boundary():
    builder = GraphBuilder()
    value = builder.input((3, 4), dtype="int64")
    reduced = value.sum()
    module = builder.finish(reduced.relu())

    cpu = lower_to_cpu(module)
    loops = lower_to_loops(cpu)
    fused = fuse_elementwise(loops)

    reduction = next(kernel for kernel in loops.kernels if kernel.opcode == "sum")
    assert reduction.iteration_shape == ()
    assert reduction.input_maps == ()
    assert reduction.output != reduction.inputs[0]
    assert [kernel.opcode for kernel in fused.kernels] == ["sum", "relu"]


def test_cpu_and_loop_sum_read_logical_c_order_from_signed_stride_views():
    builder = GraphBuilder()
    value = builder.input((3, 4), dtype="float32")
    viewed = value.transpose((1, 0)).reverse(0).slice(axis=1, start=0, stop=3, step=2)
    module = builder.finish(viewed.sum())
    runtime = np.array(
        [[1.0e20, 1.0, 2.0, 3.0], [-1.0e20, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0]],
        dtype=np.float32,
    )
    expected_view = runtime.transpose(1, 0)[::-1, :][:, 0:3:2]
    expected = _left_fold_sum(expected_view)

    cpu = lower_to_cpu(module)
    np.testing.assert_array_equal(execute_cpu(cpu, inputs=[runtime]), expected)
    np.testing.assert_array_equal(execute_loop(lower_to_loops(cpu), inputs=[runtime]), expected)
    np.testing.assert_array_equal(execute_reference(module, inputs=[runtime]), expected)


def test_generated_c_keeps_sum_serial_even_when_parallel_mode_is_enabled():
    builder = GraphBuilder()
    value = builder.input((4, 8), dtype="float32")
    loops = lower_to_loops(lower_to_cpu(builder.finish(value.sum())))

    serial = generate_c(loops)
    parallel = generate_c(loops, parallel=True)

    for source in (serial, parallel):
        assert "float sum_value = 0.0f;" in source
        assert "for (int64_t n = 0; n < 32; ++n)" in source
        assert "sum_value = ((float)sum_value + (float)p0[n]);" in source
        assert "p1[0] = sum_value;" in source
    assert "#pragma omp parallel for schedule(static)" not in parallel


def test_static_native_sum_composes_with_borrowed_views_parallel_and_multi_output():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    value = builder.input((6, 8), dtype="int32")
    viewed = value.transpose((1, 0)).reverse(0).slice(axis=1, start=1, stop=6, step=2)
    reduced = viewed.sum()
    module = builder.finish((reduced, reduced.relu()))
    runtime = (np.arange(48, dtype=np.int32).reshape(6, 8) - 20)

    executable = compile_module(module, borrow_inputs=True, parallel=True)
    actual = executable(inputs=[runtime])
    expected = execute_reference(module, inputs=[runtime])

    assert isinstance(actual, tuple)
    assert isinstance(expected, tuple)
    for result, wanted in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, wanted)


def test_dynamic_native_sum_specializes_symbolic_extent_and_reuses_cache():
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    value = builder.input((batch, 4), dtype="float32")
    module = builder.finish(value.reverse(1).sum())
    executable = compile_dynamic_module(module, borrow_inputs=True, parallel=True)

    for batch_size in (0, 3, 7, 3):
        runtime = np.arange(batch_size * 4, dtype=np.float32).reshape(batch_size, 4)
        actual = executable(inputs=[runtime])
        expected = _left_fold_sum(runtime[:, ::-1])
        np.testing.assert_array_equal(actual, expected)

    assert executable.cached_batch_sizes == (0, 3, 7)


def test_sum_participates_in_pure_dce_and_exact_cse():
    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="int32")
    value.sum()
    module = builder.finish(value.relu())
    assert dead_code_eliminate(module) == 1
    assert all(op.opcode != "sum" for op in module.function.ops)

    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="int32")
    lhs = value.sum()
    rhs = value.sum()
    module = builder.finish(lhs + rhs)
    assert common_subexpression_eliminate(module) == 1
    assert sum(op.opcode == "sum" for op in module.function.ops) == 1
