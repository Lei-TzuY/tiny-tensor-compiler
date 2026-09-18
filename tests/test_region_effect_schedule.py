import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    compile_module,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
)
from tiny_tensor_compiler.effect_schedule import plan_parallel_effect_groups


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _same_root_disjoint_copy_module():
    builder = GraphBuilder()
    base = builder.input((2, 6), dtype="int32")
    even_patch = builder.input((2, 3), dtype="int32")
    odd_patch = builder.input((2, 3), dtype="int32")
    owned = base.relu()
    generation1 = owned.copy_into(
        owned.slice(axis=1, start=0, stop=6, step=2),
        even_patch,
    )
    generation2 = generation1.copy_into(
        generation1.slice(axis=1, start=1, stop=6, step=2),
        odd_patch,
    )
    return builder.finish((generation2, base))


def _same_root_overlapping_copy_module():
    builder = GraphBuilder()
    base = builder.input((2, 6), dtype="int32")
    first_patch = builder.input((2, 4), dtype="int32")
    second_patch = builder.input((2, 4), dtype="int32")
    owned = base.relu()
    generation1 = owned.copy_into(
        owned.slice(axis=1, start=0, stop=4),
        first_patch,
    )
    generation2 = generation1.copy_into(
        generation1.slice(axis=1, start=2, stop=6),
        second_patch,
    )
    return builder.finish(generation2)


def _same_root_unproven_row_stride_module():
    builder = GraphBuilder()
    base = builder.input((4, 4), dtype="int32")
    even_patch = builder.input((2, 4), dtype="int32")
    odd_patch = builder.input((2, 4), dtype="int32")
    owned = base.relu()
    generation1 = owned.copy_into(
        owned.slice(axis=0, start=0, stop=4, step=2),
        even_patch,
    )
    generation2 = generation1.copy_into(
        generation1.slice(axis=0, start=1, stop=4, step=2),
        odd_patch,
    )
    return builder.finish(generation2)


def _same_root_disjoint_binary_module():
    builder = GraphBuilder()
    base = builder.input((2, 6), dtype="int32")
    even_delta = builder.input((2, 3), dtype="int32")
    odd_scale = builder.input((2, 3), dtype="int32")
    owned = base.relu()
    generation1 = owned.add_into(
        owned.slice(axis=1, start=0, stop=6, step=2),
        even_delta,
    )
    generation2 = generation1.mul_into(
        generation1.slice(axis=1, start=1, stop=6, step=2),
        odd_scale,
    )
    return builder.finish(generation2)


def _partial_then_full_root_module():
    builder = GraphBuilder()
    base = builder.input((2, 6), dtype="int32")
    patch = builder.input((2, 3), dtype="int32")
    delta = builder.input((2, 6), dtype="int32")
    owned = base.relu()
    generation1 = owned.copy_into(
        owned.slice(axis=1, start=0, stop=6, step=2),
        patch,
    )
    generation2 = generation1.add_inplace(delta)
    return builder.finish(generation2)


def _pure_kernel_barrier_module():
    builder = GraphBuilder()
    base = builder.input((2, 6), dtype="int32")
    even_patch = builder.input((2, 3), dtype="int32")
    odd_patch = builder.input((2, 3), dtype="int32")
    owned = base.relu()
    generation1 = owned.copy_into(
        owned.slice(axis=1, start=0, stop=6, step=2),
        even_patch,
    )
    snapshot = generation1 + 1
    generation2 = generation1.copy_into(
        generation1.slice(axis=1, start=1, stop=6, step=2),
        odd_patch,
    )
    return builder.finish((generation2, snapshot))


def _independent_pure_kernel_crossing_module():
    builder = GraphBuilder()
    left = builder.input((6,), dtype="int32")
    right = builder.input((6,), dtype="int32")
    left_delta = builder.input((6,), dtype="int32")
    right_delta = builder.input((6,), dtype="int32")
    side = builder.input((6,), dtype="int32")
    left_owned = left.relu()
    right_owned = right.relu()
    first = left_owned.add_inplace(left_delta)
    snapshot = side.relu()
    second = right_owned.add_inplace(right_delta)
    return builder.finish((first, second, snapshot, left, right))


def _pure_kernel_producer_barrier_module():
    builder = GraphBuilder()
    left = builder.input((6,), dtype="int32")
    right = builder.input((6,), dtype="int32")
    left_delta = builder.input((6,), dtype="int32")
    side = builder.input((6,), dtype="int32")
    left_owned = left.relu()
    right_owned = right.relu()
    first = left_owned.add_inplace(left_delta)
    produced = side.relu()
    second = right_owned.add_inplace(produced)
    return builder.finish((first, second, left, right))


def _multi_effect_groups(module):
    loops = lower_to_loops(lower_to_cpu(module))
    return loops, [
        group for group in plan_parallel_effect_groups(loops) if len(group.effects) > 1
    ]


def test_same_root_interleaved_columns_cross_fresh_view_into_one_group():
    loops, multi = _multi_effect_groups(_same_root_disjoint_copy_module())
    assert len(multi) == 1
    group = multi[0]
    assert len(group.effects) == 2
    assert loops.storage_root(group.effects[0].root) == loops.storage_root(group.effects[1].root)
    assert group.operation_indices[1] > group.operation_indices[0] + 1
    assert group.transparent_indices
    assert generate_c(loops, parallel=True).count("#pragma omp section") == 2


def test_same_root_disjoint_copy_sections_match_native_and_preserve_borrowed_input():
    _default_compiler_or_skip()
    module = _same_root_disjoint_copy_module()
    base = np.arange(12, dtype=np.int32).reshape(2, 6) - 4
    even_patch = 100 + np.arange(6, dtype=np.int32).reshape(2, 3)
    odd_patch = 200 + np.arange(6, dtype=np.int32).reshape(2, 3)
    expected = np.maximum(base, 0)
    expected[:, 0:6:2] = even_patch
    expected[:, 1:6:2] = odd_patch

    actual, returned = compile_module(module, borrow_inputs=True, parallel=True)(
        inputs=[base, even_patch, odd_patch]
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(returned, base)
    np.testing.assert_array_equal(base, np.arange(12, dtype=np.int32).reshape(2, 6) - 4)


def test_overlapping_same_root_regions_remain_serial():
    loops, multi = _multi_effect_groups(_same_root_overlapping_copy_module())
    assert multi == []
    assert "#pragma omp parallel sections" not in generate_c(loops, parallel=True)


def test_disjoint_multidimensional_regions_fail_closed_when_exact_set_is_unproven():
    loops, multi = _multi_effect_groups(_same_root_unproven_row_stride_module())
    assert multi == []
    assert "#pragma omp parallel sections" not in generate_c(loops, parallel=True)


def test_same_root_disjoint_binary_effects_use_region_dependence():
    _default_compiler_or_skip()
    module = _same_root_disjoint_binary_module()
    loops, multi = _multi_effect_groups(module)
    assert len(multi) == 1
    assert generate_c(loops, parallel=True).count("#pragma omp section") == 2

    base = np.arange(12, dtype=np.int32).reshape(2, 6) - 3
    even_delta = 10 + np.arange(6, dtype=np.int32).reshape(2, 3)
    odd_scale = np.array([[2, 3, 4], [5, 6, 7]], dtype=np.int32)
    expected = np.maximum(base, 0)
    expected[:, 0:6:2] += even_delta
    expected[:, 1:6:2] *= odd_scale
    actual = compile_module(module, parallel=True)(inputs=[base, even_delta, odd_scale])
    np.testing.assert_array_equal(actual, expected)


def test_full_root_inplace_effect_conflicts_with_partial_same_root_write():
    loops, multi = _multi_effect_groups(_partial_then_full_root_module())
    assert multi == []
    assert "#pragma omp parallel sections" not in generate_c(loops, parallel=True)


def test_effect_may_cross_independent_pure_kernel_and_native_result_stays_exact():
    _default_compiler_or_skip()
    module = _independent_pure_kernel_crossing_module()
    loops, multi = _multi_effect_groups(module)
    assert len(multi) == 1
    group = multi[0]
    assert len(group.effects) == 2
    assert group.operation_indices[1] > group.operation_indices[0] + 1
    assert generate_c(loops, parallel=True).count("#pragma omp section") == 2

    left = np.array([-3, -1, 0, 2, 4, 7], dtype=np.int32)
    right = np.array([-5, 1, 3, -2, 8, 0], dtype=np.int32)
    left_delta = np.arange(6, dtype=np.int32) + 10
    right_delta = np.arange(6, dtype=np.int32) + 20
    side = np.array([-9, -2, 0, 1, 5, 11], dtype=np.int32)
    actual_left, actual_right, actual_side, returned_left, returned_right = compile_module(
        module, parallel=True
    )(inputs=[left, right, left_delta, right_delta, side])
    np.testing.assert_array_equal(actual_left, np.maximum(left, 0) + left_delta)
    np.testing.assert_array_equal(actual_right, np.maximum(right, 0) + right_delta)
    np.testing.assert_array_equal(actual_side, np.maximum(side, 0))
    np.testing.assert_array_equal(returned_left, left)
    np.testing.assert_array_equal(returned_right, right)


def test_pure_kernel_read_hazard_remains_a_scheduling_barrier():
    loops, multi = _multi_effect_groups(_pure_kernel_barrier_module())
    assert multi == []
    assert "#pragma omp parallel sections" not in generate_c(loops, parallel=True)


def test_pure_kernel_producer_dependency_remains_a_scheduling_barrier():
    loops, multi = _multi_effect_groups(_pure_kernel_producer_barrier_module())
    assert multi == []
    assert "#pragma omp parallel sections" not in generate_c(loops, parallel=True)
