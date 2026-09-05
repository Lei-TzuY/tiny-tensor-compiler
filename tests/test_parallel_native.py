import os
import shutil

import numpy as np
import pytest

import tiny_tensor_compiler.native as native_module
from tiny_tensor_compiler import (
    GraphBuilder,
    SymbolicDim,
    compile_dynamic_module,
    compile_module,
    execute_reference,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
)


@pytest.fixture(autouse=True)
def _clear_native_artifact_cache():
    native_module.clear_native_cache()
    yield
    native_module.clear_native_cache()


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _same_shape_float_module(shape=(64,)):
    builder = GraphBuilder()
    lhs = builder.input(shape, dtype="float32")
    rhs = builder.input(shape, dtype="float32")
    return builder.finish((lhs + rhs).relu())


def test_parallel_codegen_is_opt_in_for_verified_non_scalar_kernel():
    module = _same_shape_float_module()
    loops = lower_to_loops(lower_to_cpu(module))

    serial = generate_c(loops)
    parallel = generate_c(loops, parallel=True)

    assert "#pragma omp parallel for schedule(static)" not in serial
    assert "#pragma omp parallel for schedule(static)" in parallel
    assert "TINY_TENSOR_VECTORIZE_LOOP\n        for" in serial
    assert "TINY_TENSOR_VECTORIZE_LOOP\n        for" not in parallel


def test_parallel_codegen_schedules_outer_loop_for_broadcast_kernel():
    builder = GraphBuilder()
    lhs = builder.input((4, 1), dtype="float32")
    rhs = builder.input((1, 3), dtype="float32")
    loops = lower_to_loops(lower_to_cpu(builder.finish(lhs + rhs)))

    source = generate_c(loops, parallel=True)

    marker = "#pragma omp parallel for schedule(static)"
    assert source.count(marker) == 1
    assert f"{marker}\n        for (int64_t i0 = 0; i0 < 4; ++i0)" in source
    assert "for (int64_t i1 = 0; i1 < 3; ++i1)" in source


def test_parallel_codegen_keeps_scalar_and_zero_extent_kernels_serial():
    builder = GraphBuilder()
    scalar = builder.input((), dtype="float32")
    empty = builder.input((0,), dtype="float32")
    loops = lower_to_loops(lower_to_cpu(builder.finish((scalar.relu(), empty.relu()))))

    source = generate_c(loops, parallel=True)

    assert "#pragma omp parallel for schedule(static)" not in source


def test_parallel_compiler_mode_adds_platform_openmp_flag():
    command = ["cl"] if os.name == "nt" else ["cc"]

    enabled = native_module._enable_openmp(command)

    expected = "/openmp" if os.name == "nt" else "-fopenmp"
    assert enabled[-1] == expected
    assert native_module._enable_openmp(enabled) == enabled


def test_compile_module_parallel_native_matches_reference():
    _default_compiler_or_skip()
    module = _same_shape_float_module((257,))
    lhs = np.linspace(-10.0, 10.0, 257, dtype=np.float32)
    rhs = np.linspace(3.0, -4.0, 257, dtype=np.float32)

    executable = compile_module(module, parallel=True)
    actual = executable(inputs=[lhs, rhs])
    expected = execute_reference(module, inputs=[lhs, rhs])

    np.testing.assert_array_equal(actual, expected)


def test_parallel_native_preserves_broadcast_multi_output_and_borrowed_inputs():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    lhs = builder.input((17, 1), dtype="float32")
    rhs = builder.input((1, 5), dtype="float32")
    summed = lhs + rhs
    module = builder.finish((summed, summed.relu()))
    lhs_value = np.arange(17, dtype=np.float32).reshape(17, 1)
    rhs_value = np.linspace(-4.0, 2.0, 5, dtype=np.float32).reshape(1, 5)

    executable = compile_module(module, borrow_inputs=True, parallel=True)
    actual = executable(inputs=[lhs_value, rhs_value])
    expected = execute_reference(module, inputs=[lhs_value, rhs_value])

    assert isinstance(actual, tuple)
    assert isinstance(expected, tuple)
    for actual_output, expected_output in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_output, expected_output)


def test_dynamic_parallel_specializations_reuse_openmp_native_path():
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    lhs = builder.input((batch, 4), dtype="float32")
    rhs = builder.input((batch, 4), dtype="float32")
    module = builder.finish(lhs + rhs)
    executable = compile_dynamic_module(module, parallel=True)

    for batch_size in (2, 5, 2):
        lhs_value = np.arange(batch_size * 4, dtype=np.float32).reshape(batch_size, 4)
        rhs_value = np.full((batch_size, 4), 3.5, dtype=np.float32)
        actual = executable(inputs=[lhs_value, rhs_value])
        expected = execute_reference(module, inputs=[lhs_value, rhs_value])
        np.testing.assert_array_equal(actual, expected)

    assert executable.cached_batch_sizes == (2, 5)
