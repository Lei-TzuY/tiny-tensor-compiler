import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    SymbolicDim,
    TypeInferenceError,
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


def test_static_concatenate_is_typed_variadic_ir_and_matches_reference():
    builder = GraphBuilder()
    left = builder.input((2, 2), dtype="float32")
    middle = builder.input((2, 1), dtype="float32")
    right = builder.input((2, 3), dtype="float32")
    joined = builder.concatenate((left, middle, right), axis=1)
    module = builder.finish(joined)

    assert joined.type.shape == (2, 6)
    assert joined.type.dtype == left.type.dtype
    assert "concat" in module.dump()

    inputs = [
        np.arange(4, dtype=np.float32).reshape(2, 2),
        np.array([[10.0], [11.0]], dtype=np.float32),
        np.arange(20, 26, dtype=np.float32).reshape(2, 3),
    ]
    actual = execute_reference(module, inputs=inputs)
    expected = np.concatenate(inputs, axis=1)
    np.testing.assert_array_equal(actual, expected)


def test_concatenate_normalizes_negative_axis_and_rejects_invalid_contracts():
    builder = GraphBuilder()
    lhs = builder.input((2, 3), dtype="int32")
    rhs = builder.input((2, 4), dtype="int32")
    joined = builder.concatenate((lhs, rhs), axis=-1)
    assert joined.type.shape == (2, 7)
    assert joined.value.producer is not None
    assert joined.value.producer.attrs == {"axis": 1}

    with pytest.raises(TypeInferenceError, match="at least two"):
        builder.concatenate((lhs,), axis=1)
    with pytest.raises(TypeInferenceError, match="axis"):
        builder.concatenate((lhs, rhs), axis=2)

    bad_shape = builder.input((3, 4), dtype="int32")
    with pytest.raises(TypeInferenceError, match="non-concatenated dimensions"):
        builder.concatenate((lhs, bad_shape), axis=1)

    bad_dtype = builder.input((2, 4), dtype="float32")
    with pytest.raises(TypeInferenceError, match="same exact dtype"):
        builder.concatenate((lhs, bad_dtype), axis=1)

    scalar_a = builder.input((), dtype="int32")
    scalar_b = builder.input((), dtype="int32")
    with pytest.raises(TypeInferenceError, match="rank"):
        builder.concatenate((scalar_a, scalar_b), axis=0)


def test_symbolic_concatenate_axis_sum_specializes_exactly():
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    left = builder.input((2, batch), dtype="int32")
    right = builder.input((2, 2 * batch + 1), dtype="int32")
    joined = builder.concatenate((left, right), axis=1)
    module = builder.finish(joined)

    assert str(joined.type.shape[1]) == "3*B+1"

    for batch_size in (0, 3):
        lhs = np.arange(2 * batch_size, dtype=np.int32).reshape(2, batch_size)
        rhs = np.arange(2 * (2 * batch_size + 1), dtype=np.int32).reshape(
            2, 2 * batch_size + 1
        )
        actual = execute_reference(module, inputs=[lhs, rhs])
        np.testing.assert_array_equal(actual, np.concatenate((lhs, rhs), axis=1))


def test_concatenate_lowering_reads_logical_view_layouts_and_owns_output():
    builder = GraphBuilder()
    source = builder.input((2, 4), dtype="int32")
    reversed_source = source.reverse(1)
    sliced = source.slice(axis=1, start=0, stop=4, step=2)
    joined = builder.concatenate((reversed_source, sliced), axis=1)
    module = builder.finish(joined.relu())

    cpu = lower_to_cpu(module)
    loops = lower_to_loops(cpu)
    fused = fuse_elementwise(loops)
    concat = next(kernel for kernel in loops.kernels if kernel.opcode == "concat")

    assert concat.iteration_shape == (2, 6)
    assert concat.input_maps == ()
    assert concat.concat_axis == 1
    assert concat.output not in concat.inputs
    assert [kernel.opcode for kernel in fused.kernels] == ["concat", "relu"]

    runtime = np.arange(8, dtype=np.int32).reshape(2, 4)
    expected = execute_reference(module, inputs=[runtime])
    np.testing.assert_array_equal(execute_cpu(cpu, inputs=[runtime]), expected)
    np.testing.assert_array_equal(execute_loop(loops, inputs=[runtime]), expected)


def test_generated_c_concatenate_uses_logical_layouts_and_parallel_mode_is_safe_fallback():
    builder = GraphBuilder()
    source = builder.input((2, 4), dtype="int32")
    joined = builder.concatenate(
        (source.reverse(1), source.slice(axis=1, start=0, stop=4, step=2)),
        axis=1,
    )
    loops = lower_to_loops(lower_to_cpu(builder.finish(joined)))

    serial = generate_c(loops)
    parallel = generate_c(loops, parallel=True)

    assert "concat" in serial
    assert "#pragma omp parallel for schedule(static)" not in serial
    assert "#pragma omp parallel for schedule(static)" not in parallel
    assert "p" in serial and "[" in serial


def test_native_concatenate_composes_with_borrowing_and_multi_output():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    lhs = builder.input((3, 2), dtype="float32")
    rhs = builder.input((3, 3), dtype="float32")
    joined = builder.concatenate((lhs, rhs), axis=1)
    module = builder.finish((joined, joined.relu()))
    inputs = [
        np.arange(6, dtype=np.float32).reshape(3, 2) - 4.0,
        np.arange(9, dtype=np.float32).reshape(3, 3) - 2.0,
    ]

    executable = compile_module(module, borrow_inputs=True, parallel=True)
    actual = executable(inputs=inputs)
    expected = execute_reference(module, inputs=inputs)

    assert isinstance(actual, tuple)
    assert isinstance(expected, tuple)
    for result, wanted in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, wanted)
    assert not np.shares_memory(actual[0], inputs[0])
    assert not np.shares_memory(actual[0], inputs[1])


def test_dynamic_native_concatenate_specializes_and_reuses_cache():
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    lhs = builder.input((2, batch), dtype="int32")
    rhs = builder.input((2, batch + 1), dtype="int32")
    module = builder.finish(builder.concatenate((lhs, rhs), axis=1))
    executable = compile_dynamic_module(module, borrow_inputs=True, parallel=True)

    for batch_size in (1, 4, 1):
        left = np.arange(2 * batch_size, dtype=np.int32).reshape(2, batch_size)
        right = np.arange(2 * (batch_size + 1), dtype=np.int32).reshape(
            2, batch_size + 1
        )
        actual = executable(inputs=[left, right])
        np.testing.assert_array_equal(actual, np.concatenate((left, right), axis=1))

    assert executable.cached_batch_sizes == (1, 4)


def test_concatenate_is_pure_for_dce_and_exact_cse():
    builder = GraphBuilder()
    lhs = builder.input((2, 2), dtype="int32")
    rhs = builder.input((2, 1), dtype="int32")
    first = builder.concatenate((lhs, rhs), axis=1)
    duplicate = builder.concatenate((lhs, rhs), axis=1)
    unused = builder.concatenate((rhs, rhs), axis=1)
    module = builder.finish((first, duplicate))

    assert unused.value.producer is not None
    assert common_subexpression_eliminate(module) == 1
    assert dead_code_eliminate(module) == 1
    concat_ops = [op for op in module.function.ops if op.opcode == "concat"]
    assert len(concat_ops) == 1
