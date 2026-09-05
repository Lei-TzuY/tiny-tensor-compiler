import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    SymbolicDim,
    TypeInferenceError,
    compile_dynamic_module,
    compile_module,
    execute_cpu,
    execute_loop,
    execute_reference,
    fuse_elementwise,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
    plan_memory,
)


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def test_static_reshape_is_typed_ir_and_reference_uses_c_order():
    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="float32")
    reshaped = value.reshape((3, 2))
    module = builder.finish(reshaped)

    assert reshaped.type.shape == (3, 2)
    assert reshaped.type.dtype == value.type.dtype
    assert "%1 = reshape %0" in module.dump()

    runtime = np.arange(6, dtype=np.float32).reshape(2, 3)
    actual = execute_reference(module, inputs=[runtime])
    expected = np.reshape(runtime, (3, 2), order="C")
    np.testing.assert_array_equal(actual, expected)
    assert not np.shares_memory(actual, runtime)


def test_reshape_rejects_static_element_count_mismatch_and_negative_extent():
    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="int32")

    with pytest.raises(TypeInferenceError, match="element count"):
        value.reshape((4, 2))
    with pytest.raises(TypeInferenceError, match="invalid tensor shape|non-negative"):
        value.reshape((2, -1))


def test_symbolic_reshape_requires_exact_element_count_identity():
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    value = builder.input((batch, 4), dtype="float32")

    reshaped = value.reshape((2, 2 * batch))
    assert reshaped.type.shape == (2, 2 * batch)

    with pytest.raises(TypeInferenceError, match="element count"):
        value.reshape((batch, 5))


def test_symbolic_reshape_accepts_reordered_existing_symbols_and_rejects_new_symbol():
    batch = SymbolicDim("B")
    width = SymbolicDim("W")
    other = SymbolicDim("X")
    builder = GraphBuilder()
    value = builder.input((batch + 1, width + 1), dtype="int32")

    reshaped = value.reshape((width + 1, batch + 1))
    assert reshaped.type.shape == (width + 1, batch + 1)

    with pytest.raises(TypeInferenceError, match="new symbolic dimension"):
        value.reshape((other + 1, batch + 1))


def test_reshape_lowers_to_distinct_copy_kernel_and_is_fusion_boundary():
    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="int32")
    reshaped = value.reshape((3, 2))
    module = builder.finish(reshaped.relu())

    cpu = lower_to_cpu(module)
    plan = plan_memory(cpu)
    loops = lower_to_loops(cpu)
    fused = fuse_elementwise(loops)

    reshape_kernel = next(kernel for kernel in loops.kernels if kernel.opcode == "reshape")
    assert reshape_kernel.iteration_shape == (3, 2)
    assert reshape_kernel.input_maps == ()
    assert reshape_kernel.output != reshape_kernel.inputs[0]
    assert plan.physical_count == 3
    assert [kernel.opcode for kernel in fused.kernels] == ["reshape", "relu"]


def test_cpu_and_loop_execution_match_c_order_reshape_including_scalar_and_zero_extent():
    builder = GraphBuilder()
    matrix = builder.input((2, 3), dtype="int64")
    scalar = builder.input((1,), dtype="int64")
    empty = builder.input((0, 4), dtype="int64")
    module = builder.finish(
        (
            matrix.reshape((3, 2)),
            scalar.reshape(()),
            empty.reshape((2, 0)),
        )
    )
    matrix_value = np.arange(6, dtype=np.int64).reshape(2, 3)
    scalar_value = np.array([17], dtype=np.int64)
    empty_value = np.empty((0, 4), dtype=np.int64)
    inputs = [matrix_value, scalar_value, empty_value]

    cpu = lower_to_cpu(module)
    cpu_actual = execute_cpu(cpu, inputs=inputs)
    loop_actual = execute_loop(lower_to_loops(cpu), inputs=inputs)
    expected = execute_reference(module, inputs=inputs)

    assert isinstance(cpu_actual, tuple)
    assert isinstance(loop_actual, tuple)
    assert isinstance(expected, tuple)
    for actual_set in (cpu_actual, loop_actual):
        for actual, wanted in zip(actual_set, expected, strict=True):
            np.testing.assert_array_equal(actual, wanted)


def test_generated_c_uses_flat_copy_and_parallel_mode_schedules_it():
    builder = GraphBuilder()
    value = builder.input((4, 8), dtype="float32")
    loops = lower_to_loops(lower_to_cpu(builder.finish(value.reshape((8, 4)))))

    serial = generate_c(loops)
    parallel = generate_c(loops, parallel=True)

    assert "p1[n] = p0[n];" in serial
    assert "#pragma omp parallel for schedule(static)" not in serial
    assert "#pragma omp parallel for schedule(static)" in parallel
    assert "p1[n] = p0[n];" in parallel


def test_static_native_reshape_composes_with_borrowing_parallel_and_multi_output():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    value = builder.input((9, 4), dtype="float32")
    reshaped = value.reshape((6, 6))
    module = builder.finish((reshaped, reshaped.relu()))
    runtime = np.linspace(-20.0, 15.0, 36, dtype=np.float32).reshape(9, 4)

    executable = compile_module(module, borrow_inputs=True, parallel=True)
    actual = executable(inputs=[runtime])
    expected = execute_reference(module, inputs=[runtime])

    assert isinstance(actual, tuple)
    assert isinstance(expected, tuple)
    for result, wanted in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, wanted)
    assert not np.shares_memory(actual[0], runtime)


def test_dynamic_native_symbolic_reshape_specializes_and_reuses_cache():
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    value = builder.input((batch, 4), dtype="int32")
    module = builder.finish(value.reshape((2, 2 * batch)))
    executable = compile_dynamic_module(module, borrow_inputs=True, parallel=True)

    for batch_size in (2, 5, 2):
        runtime = np.arange(batch_size * 4, dtype=np.int32).reshape(batch_size, 4)
        actual = executable(inputs=[runtime])
        expected = np.reshape(runtime, (2, 2 * batch_size), order="C")
        np.testing.assert_array_equal(actual, expected)

    assert executable.cached_batch_sizes == (2, 5)
