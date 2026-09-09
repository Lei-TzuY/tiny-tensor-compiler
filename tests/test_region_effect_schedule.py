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


def test_pure_kernel_remains_a_hard_scheduling_barrier():
    loops, multi = _multi_effect_groups(_pure_kernel_barrier_module())
    assert multi == []
    assert "#pragma omp parallel sections" not in generate_c(loops, parallel=True)
