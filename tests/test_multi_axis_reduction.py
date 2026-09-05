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
from tiny_tensor_compiler.reduction import ReductionOperator
from tiny_tensor_compiler.repro import capture_repro_case, load_repro_case, replay_repro_case
from tiny_tensor_compiler.serialization import deserialize_module, serialize_module


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _left_fold_axes(
    value: np.ndarray,
    axes: tuple[int, ...],
    operator: ReductionOperator,
) -> np.ndarray:
    axes = tuple(sorted(axis % value.ndim for axis in axes))
    axis_set = set(axes)
    output_shape = tuple(dim for axis, dim in enumerate(value.shape) if axis not in axis_set)
    reduction_shape = tuple(value.shape[axis] for axis in axes)
    output = np.empty(output_shape, dtype=value.dtype)

    for output_index in np.ndindex(output_shape):
        accumulator = operator.identity(value.dtype)
        for reduction_index in np.ndindex(reduction_shape):
            source_index: list[int] = []
            output_position = 0
            reduction_position = 0
            for axis in range(value.ndim):
                if axis in axis_set:
                    source_index.append(reduction_index[reduction_position])
                    reduction_position += 1
                else:
                    source_index.append(output_index[output_position])
                    output_position += 1
            accumulator = operator.combine(value.dtype, accumulator, value[tuple(source_index)])
        output[output_index] = accumulator
    return output


def test_multi_axis_reductions_canonicalize_axis_collections_in_typed_ir() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3, 4, 5), dtype="int32")
    summed = value.sum(axis=(2, -4))
    product = value.prod(axis={3, 1})
    module = builder.finish((summed, product))

    assert summed.type.shape == (3, 5)
    assert product.type.shape == (2, 4)
    assert summed.type.dtype == value.type.dtype
    reductions = [op for op in module.function.ops if op.opcode in {"sum", "prod"}]
    assert [op.attrs for op in reductions] == [
        {"axis": (0, 2)},
        {"axis": (1, 3)},
    ]


def test_multi_axis_reductions_reject_empty_duplicate_bool_and_out_of_range_axes() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="float32")

    for axes in ((), (1, -2), (True, 2), (0, 3), (0, -4)):
        with pytest.raises(TypeInferenceError, match="axis"):
            value.sum(axis=axes)
    with pytest.raises(TypeInferenceError, match="axis"):
        value.prod(axis="01")


def test_reduction_plan_carries_canonical_multi_axis_domain_through_loop_ir() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3, 4, 5), dtype="int64")
    module = builder.finish(value.prod(axis=(3, 1)))

    cpu = lower_to_cpu(module)
    loops = lower_to_loops(cpu)
    buffer_product = next(kernel for kernel in cpu.instructions if kernel.opcode == "prod")
    loop_product = next(kernel for kernel in loops.kernels if kernel.opcode == "prod")

    assert buffer_product.reduction is not None
    assert loop_product.reduction is not None
    assert buffer_product.reduction.axis == (1, 3)
    assert loop_product.reduction.axis == (1, 3)
    assert buffer_product.reduction_axis == (1, 3)
    assert loop_product.reduction_axis == (1, 3)
    assert loop_product.iteration_shape == (2, 4)
    assert loop_product.input_maps == ()


def test_reference_and_loop_multi_axis_reductions_use_deterministic_logical_order() -> None:
    builder = GraphBuilder()
    value = builder.input((3, 4, 5), dtype="float32")
    viewed = value.transpose((2, 0, 1)).reverse(2)
    module = builder.finish((viewed.sum(axis=(0, 2)), viewed.prod(axis=(2, 0))))

    runtime = ((np.arange(60, dtype=np.float32).reshape(3, 4, 5) % 11) - 5) / 3.0
    logical = runtime.transpose(2, 0, 1)[:, :, ::-1]
    expected_sum = _left_fold_axes(logical, (0, 2), ReductionOperator.SUM)
    expected_prod = _left_fold_axes(logical, (0, 2), ReductionOperator.PRODUCT)

    reference = execute_reference(module, inputs=[runtime])
    loop = execute_loop(lower_to_loops(lower_to_cpu(module)), inputs=[runtime])
    assert isinstance(reference, tuple)
    assert isinstance(loop, tuple)
    np.testing.assert_array_equal(reference[0], expected_sum)
    np.testing.assert_array_equal(reference[1], expected_prod)
    np.testing.assert_array_equal(loop[0], expected_sum)
    np.testing.assert_array_equal(loop[1], expected_prod)


def test_multi_axis_empty_domain_uses_each_operator_identity() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 0, 3, 0), dtype="float64")
    module = builder.finish((value.sum(axis=(1, 3)), value.prod(axis=(3, 1))))
    runtime = np.empty((2, 0, 3, 0), dtype=np.float64)

    actual = execute_reference(module, inputs=[runtime])
    assert isinstance(actual, tuple)
    np.testing.assert_array_equal(actual[0], np.zeros((2, 3), dtype=np.float64))
    np.testing.assert_array_equal(actual[1], np.ones((2, 3), dtype=np.float64))


def test_generated_c_nests_reduction_axes_and_parallelizes_only_output_domain() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3, 4, 5), dtype="float32")
    loops = lower_to_loops(lower_to_cpu(builder.finish(value.prod(axis=(1, 3)))))

    serial = generate_c(loops)
    parallel = generate_c(loops, parallel=True)
    for source in (serial, parallel):
        assert "float prod_value = 1.0f;" in source
        assert "for (int64_t r0 = 0; r0 < 3; ++r0)" in source
        assert "for (int64_t r1 = 0; r1 < 5; ++r1)" in source
        assert "prod_value = ((float)prod_value *" in source
    assert "#pragma omp parallel for schedule(static)" not in serial
    assert parallel.count("#pragma omp parallel for schedule(static)") == 1

    builder = GraphBuilder()
    scalar = builder.input((2, 3), dtype="int32").sum(axis=(0, 1))
    scalar_parallel = generate_c(lower_to_loops(lower_to_cpu(builder.finish(scalar))), parallel=True)
    assert "#pragma omp parallel for schedule(static)" not in scalar_parallel


def test_static_native_multi_axis_reductions_compose_with_views_borrowing_and_parallel() -> None:
    _default_compiler_or_skip()
    builder = GraphBuilder()
    value = builder.input((4, 5, 6), dtype="int32")
    viewed = value.transpose((2, 0, 1)).reverse(2)
    module = builder.finish((viewed.sum(axis=(0, 2)), viewed.prod(axis=(2, 0))))
    runtime = (np.arange(120, dtype=np.int32).reshape(4, 5, 6) % 5) - 2

    actual = compile_module(module, borrow_inputs=True, parallel=True)(inputs=[runtime])
    expected = execute_reference(module, inputs=[runtime])
    assert isinstance(actual, tuple)
    assert isinstance(expected, tuple)
    for result, wanted in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, wanted)


def test_dynamic_multi_axis_reduction_preserves_unreduced_symbolic_extent_and_cache() -> None:
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    value = builder.input((batch, 3, 4, 2), dtype="float32")
    module = builder.finish(value.reverse(2).sum(axis=(3, 1)))
    executable = compile_dynamic_module(module, borrow_inputs=True, parallel=True)

    for batch_size in (0, 2, 5, 2):
        runtime = ((np.arange(batch_size * 24, dtype=np.float32) % 9) - 4).reshape(
            batch_size, 3, 4, 2
        )
        logical = runtime[:, :, ::-1, :]
        actual = executable(inputs=[runtime])
        expected = _left_fold_axes(logical, (1, 3), ReductionOperator.SUM)
        np.testing.assert_array_equal(actual, expected)

    assert executable.cached_batch_sizes == (0, 2, 5)


def test_multi_axis_reductions_participate_in_dce_and_axis_sensitive_cse() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="int32")
    value.sum(axis=(0, 2))
    module = builder.finish(value.relu())
    assert dead_code_eliminate(module) == 1

    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="int32")
    a = value.prod(axis=(2, 0))
    b = value.prod(axis=(0, 2))
    c = value.prod(axis=(0, 1))
    d = value.sum(axis=(0, 2))
    module = builder.finish((a, b, c, d))
    assert common_subexpression_eliminate(module) == 1
    products = [op for op in module.function.ops if op.opcode == "prod"]
    sums = [op for op in module.function.ops if op.opcode == "sum"]
    assert [op.attrs for op in products] == [{"axis": (0, 2)}, {"axis": (0, 1)}]
    assert [op.attrs for op in sums] == [{"axis": (0, 2)}]


def test_multi_axis_reductions_round_trip_through_serialization_and_repro() -> None:
    _default_compiler_or_skip()
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="int32")
    module = builder.finish((value.sum(axis=(2, 0)), value.reverse(2).prod(axis=(0, 1))))
    runtime = (np.arange(24, dtype=np.int32).reshape(2, 3, 4) % 5) - 2

    document = serialize_module(module)
    restored = deserialize_module(document)
    assert serialize_module(restored) == document
    assert [op.attrs for op in restored.function.ops if op.opcode in {"sum", "prod"}] == [
        {"axis": (0, 2)},
        {"axis": (0, 1)},
    ]

    repro = capture_repro_case(restored, inputs=[runtime])
    case = load_repro_case(repro)
    assert [op.attrs for op in case.module.function.ops if op.opcode in {"sum", "prod"}] == [
        {"axis": (0, 2)},
        {"axis": (0, 1)},
    ]
    reference = replay_repro_case(repro, backend="reference")
    native = replay_repro_case(repro, backend="native", parallel=True)
    assert isinstance(reference, tuple)
    assert isinstance(native, tuple)
    for actual, expected in zip(native, reference, strict=True):
        np.testing.assert_array_equal(actual, expected)
