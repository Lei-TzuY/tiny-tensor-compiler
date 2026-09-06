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
    return builder.finish((out_a, out_b))


def _same_root_effect_chain_module():
    builder = GraphBuilder()
    base = builder.input((4,), dtype="int32")
    first_source = builder.input((4,), dtype="int32")
    second_source = builder.input((4,), dtype="int32")
    root = base.relu()
    first = root.add_inplace(first_source)
    second = first.mul_inplace(second_source)
    return builder.finish(second)


def test_effect_schedule_groups_independent_roots_with_shared_read_source():
    loops = lower_to_loops(lower_to_cpu(_independent_partial_effect_module()))
    groups = plan_parallel_effect_groups(loops)
    multi = [group for group in groups if len(group.effects) > 1]

    assert len(multi) == 1
    group = multi[0]
    assert len(group.effects) == 2
    assert group.operation_indices[1] == group.operation_indices[0] + 1
    assert loops.storage_root(group.effects[0].root) != loops.storage_root(group.effects[1].root)
    assert loops.storage_root(group.effects[0].source) == loops.storage_root(group.effects[1].source)


def test_parallel_codegen_uses_sections_without_nested_effect_parallel_for():
    loops = lower_to_loops(lower_to_cpu(_independent_partial_effect_module()))
    effects = loops.binary_intos
    assert len(effects) == 2

    serial = generate_c(loops, parallel=False)
    parallel = generate_c(loops, parallel=True)

    assert "#pragma omp parallel sections" not in serial
    assert parallel.count("#pragma omp parallel sections") == 1
    assert parallel.count("#pragma omp section") == 2

    start = parallel.index("#pragma omp parallel sections")
    first_alias = f"int32_t *p{effects[0].output} = p{effects[0].root};"
    second_alias = f"int32_t *p{effects[1].output} = p{effects[1].root};"
    first_alias_pos = parallel.index(first_alias)
    second_alias_pos = parallel.index(second_alias)
    region = parallel[start:first_alias_pos]

    assert "#pragma omp parallel for" not in region
    assert first_alias_pos < second_alias_pos


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

    actual_a, actual_b = compile_module(module, borrow_inputs=True, parallel=True)(
        inputs=[base_a, base_b, shared]
    )
    np.testing.assert_array_equal(actual_a, expected_a)
    np.testing.assert_array_equal(actual_b, expected_b)


def test_same_root_effect_chain_remains_sequential_and_has_no_sections():
    _default_compiler_or_skip()
    module = _same_root_effect_chain_module()
    loops = lower_to_loops(lower_to_cpu(module))
    groups = plan_parallel_effect_groups(loops)

    assert [len(group.effects) for group in groups if group.effects] == [1, 1]
    assert "#pragma omp parallel sections" not in generate_c(loops, parallel=True)

    base = np.array([-2, 3, 4, 5], dtype=np.int32)
    first_source = np.array([1, 2, 3, 4], dtype=np.int32)
    second_source = np.array([5, 6, 7, 8], dtype=np.int32)
    expected = (np.maximum(base, 0) + first_source) * second_source
    actual = compile_module(module, parallel=True)(
        inputs=[base, first_source, second_source]
    )
    np.testing.assert_array_equal(actual, expected)
