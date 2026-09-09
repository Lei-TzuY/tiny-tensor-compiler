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
from tiny_tensor_compiler.loop_ir import LoopCopyInto, LoopInplaceBinary


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _independent_partial_effect_module():
    builder = GraphBuilder()
    base_a = builder.input((2, 4), dtype="int32")
    base_b = builder.input((2, 4), dtype="int32")
    shared = builder.input((2, 2), dtype="int32")
    root_a = base_a.relu()
    root_b = base_b.relu()
    target_a = root_a.slice(axis=1, start=0, stop=4, step=2)
    target_b = root_b.slice(axis=1, start=0, stop=4, step=2)
    out_a = root_a.add_into(target_a, shared)
    out_b = root_b.mul_into(target_b, shared)
    return builder.finish((out_a, out_b, base_a, base_b))


def _mixed_independent_effect_module():
    builder = GraphBuilder()
    copy_base = builder.input((2, 4), dtype="int32")
    inplace_base = builder.input((2, 4), dtype="int32")
    patch = builder.input((2, 2), dtype="int32")
    delta = builder.input((2, 4), dtype="int32")
    copy_root = copy_base.relu()
    inplace_root = inplace_base.relu()
    copy_target = copy_root.slice(axis=1, start=0, stop=4, step=2)
    copied = copy_root.copy_into(copy_target, patch)
    updated = inplace_root.add_inplace(delta)
    return builder.finish((copied, updated, copy_base, inplace_base))


def _broadcast_independent_effect_module():
    builder = GraphBuilder()
    base_a = builder.input((2, 4), dtype="int32")
    base_b = builder.input((2, 4), dtype="int32")
    copy_source = builder.input((2,), dtype="int32")
    binary_source = builder.input((), dtype="int32")
    root_a = base_a.relu()
    root_b = base_b.relu()
    target_a = root_a.slice(axis=1, start=0, stop=4, step=2)
    target_b = root_b.slice(axis=1, start=1, stop=4, step=2)
    copied = root_a.copy_into(target_a, copy_source)
    updated = root_b.add_into(target_b, binary_source)
    return builder.finish((copied, updated, base_a, base_b))


def _cross_root_raw_module():
    builder = GraphBuilder()
    base_a = builder.input((2, 4), dtype="int32")
    base_b = builder.input((2, 4), dtype="int32")
    patch = builder.input((2, 2), dtype="int32")
    root_a = base_a.relu()
    root_b = base_b.relu()
    target_a = root_a.slice(axis=1, start=0, stop=4, step=2)
    target_b = root_b.slice(axis=1, start=0, stop=4, step=1)
    written_a = root_a.copy_into(target_a, patch)
    written_b = root_b.copy_into(target_b, written_a)
    return builder.finish((written_a, written_b))


def test_effect_schedule_groups_independent_roots_with_shared_read_source():
    loops = lower_to_loops(lower_to_cpu(_independent_partial_effect_module()))
    multi = [group for group in plan_parallel_effect_groups(loops) if len(group.effects) > 1]
    assert len(multi) == 1
    group = multi[0]
    assert len(group.effects) == 2
    assert group.operation_indices[1] == group.operation_indices[0] + 1
    assert loops.storage_root(group.effects[0].root) != loops.storage_root(group.effects[1].root)
    assert loops.storage_root(group.effects[0].source) == loops.storage_root(group.effects[1].source)


def test_parallel_codegen_uses_sections_without_nested_effect_parallel_for():
    loops = lower_to_loops(lower_to_cpu(_independent_partial_effect_module()))
    parallel = generate_c(loops, parallel=True)
    assert parallel.count("#pragma omp parallel sections") == 1
    assert parallel.count("#pragma omp section") == 2
    region = parallel[parallel.index("#pragma omp parallel sections") :]
    region = region[: region.index(f"int32_t *p{loops.binary_intos[0].output}")]
    assert "#pragma omp parallel for" not in region


def test_independent_effect_sections_match_native_results_with_borrowed_inputs():
    _default_compiler_or_skip()
    module = _independent_partial_effect_module()
    base_a = np.array([[-3, 2, 5, 7], [1, -2, 4, 6]], dtype=np.int32)
    base_b = np.array([[2, -4, 3, 5], [-1, 8, 9, 2]], dtype=np.int32)
    shared = np.array([[10, 20], [30, 40]], dtype=np.int32)
    expected_a = np.maximum(base_a, 0)
    expected_a[:, 0:4:2] += shared
    expected_b = np.maximum(base_b, 0)
    expected_b[:, 0:4:2] *= shared
    actual_a, actual_b, returned_a, returned_b = compile_module(
        module, borrow_inputs=True, parallel=True
    )(inputs=[base_a, base_b, shared])
    np.testing.assert_array_equal(actual_a, expected_a)
    np.testing.assert_array_equal(actual_b, expected_b)
    np.testing.assert_array_equal(returned_a, base_a)
    np.testing.assert_array_equal(returned_b, base_b)


def test_mixed_copy_and_inplace_effects_share_sections():
    _default_compiler_or_skip()
    module = _mixed_independent_effect_module()
    loops = lower_to_loops(lower_to_cpu(module))
    multi = [group for group in plan_parallel_effect_groups(loops) if len(group.effects) > 1]
    assert len(multi) == 1
    assert isinstance(multi[0].effects[0], LoopCopyInto)
    assert isinstance(multi[0].effects[1], LoopInplaceBinary)
    assert generate_c(loops, parallel=True).count("#pragma omp section") == 2


def test_sections_preserve_current_broadcast_source_maps_in_native_execution():
    _default_compiler_or_skip()
    module = _broadcast_independent_effect_module()
    loops = lower_to_loops(lower_to_cpu(module))
    assert any(len(group.effects) == 2 for group in plan_parallel_effect_groups(loops))
    base_a = np.arange(8, dtype=np.int32).reshape(2, 4) - 3
    base_b = np.arange(8, dtype=np.int32).reshape(2, 4) - 2
    copy_source = np.array([17, 23], dtype=np.int32)
    binary_source = np.array(5, dtype=np.int32)
    expected_a = np.maximum(base_a, 0)
    expected_a[:, 0:4:2] = copy_source
    expected_b = np.maximum(base_b, 0)
    expected_b[:, 1:4:2] += binary_source
    actual_a, actual_b, returned_a, returned_b = compile_module(
        module, borrow_inputs=True, parallel=True
    )(inputs=[base_a, base_b, copy_source, binary_source])
    np.testing.assert_array_equal(actual_a, expected_a)
    np.testing.assert_array_equal(actual_b, expected_b)
    np.testing.assert_array_equal(returned_a, base_a)
    np.testing.assert_array_equal(returned_b, base_b)


def test_cross_root_raw_dependency_forces_a_barrier_level():
    _default_compiler_or_skip()
    module = _cross_root_raw_module()
    loops = lower_to_loops(lower_to_cpu(module))
    assert all(len(group.effects) == 1 for group in plan_parallel_effect_groups(loops))
    assert "#pragma omp parallel sections" not in generate_c(loops, parallel=True)
    base_a = np.arange(8, dtype=np.int32).reshape(2, 4) - 2
    base_b = np.arange(8, dtype=np.int32).reshape(2, 4) - 4
    patch = np.array([[31, 37], [41, 43]], dtype=np.int32)
    expected_a = np.maximum(base_a, 0)
    expected_a[:, 0:4:2] = patch
    expected_b = expected_a.copy()
    actual_a, actual_b = compile_module(module, parallel=True)(inputs=[base_a, base_b, patch])
    np.testing.assert_array_equal(actual_a, expected_a)
    np.testing.assert_array_equal(actual_b, expected_b)
