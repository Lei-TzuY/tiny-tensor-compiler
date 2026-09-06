import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    VerificationError,
    borrow_inputs,
    compile_dynamic_module,
    compile_module,
    execute_loop,
    execute_reference,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
    plan_memory,
    verify,
)
from tiny_tensor_compiler.serialization import deserialize_module, serialize_module


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _inplace_module(operator: str = "add"):
    builder = GraphBuilder()
    base = builder.input((2, 3), dtype="int32")
    source = builder.input((2, 3), dtype="int32")
    root = base.relu()
    updated = root.binary_inplace(source, operator=operator)
    return builder.finish(updated)


@pytest.mark.parametrize("operator", ["add", "mul"])
def test_inplace_binary_mutates_internal_root_and_matches_reference(operator: str):
    module = _inplace_module(operator)
    base = np.array([[-3, 2, 4], [5, -7, 1]], dtype=np.int32)
    source = np.array([[10, 3, -2], [4, 6, 8]], dtype=np.int32)

    actual = execute_reference(module, inputs=[base, source])
    owned = np.maximum(base, 0)
    expected = (
        np.add(owned, source, dtype=np.int32)
        if operator == "add"
        else np.multiply(owned, source, dtype=np.int32)
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(base, np.array([[-3, 2, 4], [5, -7, 1]], dtype=np.int32))


def test_inplace_binary_reuses_root_storage_without_new_physical_allocation():
    module = _inplace_module()
    cpu = lower_to_cpu(module)
    plan = plan_memory(cpu)
    loops = lower_to_loops(cpu)

    assert len(cpu.inplace_binaries) == 1
    effect = cpu.inplace_binaries[0]
    alias = plan.alias_for(effect.output)
    assert alias is not None
    assert alias.physical == plan.physical_for(effect.root)

    assert len(loops.inplace_binaries) == 1
    loop_effect = loops.inplace_binaries[0]
    assert loops.storage_root(loop_effect.output) == loops.storage_root(loop_effect.root)
    assert loop_effect.output not in {alloc.buffer for alloc in loops.allocations}


def test_inplace_binary_advances_generation_and_rejects_stale_handles():
    builder = GraphBuilder()
    base = builder.input((4,), dtype="int32")
    source = builder.input((4,), dtype="int32")
    root = base.relu()
    updated = root.binary_inplace(source, operator="add")
    module = builder.finish(updated.relu())
    verify(module)

    builder = GraphBuilder()
    base = builder.input((4,), dtype="int32")
    source = builder.input((4,), dtype="int32")
    root = base.relu()
    root.binary_inplace(source, operator="add")
    module = builder.finish(root.relu())
    with pytest.raises(VerificationError, match="stale tensor view/alias"):
        verify(module)


def test_inplace_binary_rejects_unsafe_roots_sources_and_types():
    builder = GraphBuilder()
    root = builder.input((4,), dtype="int32")
    source = builder.input((4,), dtype="int32")
    with pytest.raises(ValueError, match="internal computed storage"):
        root.binary_inplace(source, operator="add")

    builder = GraphBuilder()
    base = builder.input((4,), dtype="int32")
    root = base.relu()
    with pytest.raises(ValueError, match="different storage root"):
        root.binary_inplace(root.view((4,)), operator="add")

    builder = GraphBuilder()
    base = builder.input((4,), dtype="int32")
    source = builder.input((1,), dtype="int32")
    root = base.relu()
    with pytest.raises(ValueError, match="exactly match"):
        root.binary_inplace(source, operator="add")

    builder = GraphBuilder()
    base = builder.input((4,), dtype="int32")
    source = builder.input((4,), dtype="int32")
    root = base.relu()
    with pytest.raises(ValueError, match="operator"):
        root.binary_inplace(source, operator="sub")


def test_inplace_binary_loop_cpu_and_native_with_borrowed_source_and_parallel_mode():
    _default_compiler_or_skip()
    module = _inplace_module("mul")
    loops = lower_to_loops(lower_to_cpu(module))
    borrowed = borrow_inputs(loops)
    base = np.array([[-3, 2, 4], [5, -7, 1]], dtype=np.int32)
    source = np.array([[10, 3, -2], [4, 6, 8]], dtype=np.int32)
    expected = np.multiply(np.maximum(base, 0), source, dtype=np.int32)

    np.testing.assert_array_equal(execute_loop(borrowed, inputs=[base, source]), expected)
    actual = compile_module(module, borrow_inputs=True, parallel=True)(inputs=[base, source])
    np.testing.assert_array_equal(actual, expected)


def test_inplace_binary_generated_c_is_direct_serial_root_update():
    loops = lower_to_loops(lower_to_cpu(_inplace_module("add")))
    effect = loops.inplace_binaries[0]
    source = generate_c(loops, parallel=True)

    assert f"p{effect.root}[" in source
    assert f"p{effect.source}[" in source
    assert f"int32_t *p{effect.output} = p{effect.root};" in source
    effect_block = source[source.index(f"int32_t *p{effect.output} = p{effect.root};") - 500 :]
    assert "#pragma omp parallel for" not in effect_block.split(f"int32_t *p{effect.output}", 1)[0]


def test_inplace_binary_serialization_round_trip_preserves_effect():
    module = _inplace_module("mul")
    restored = deserialize_module(serialize_module(module))
    assert "binary_inplace" in restored.dump()
    verify(restored)

    base = np.arange(6, dtype=np.int32).reshape(2, 3) - 2
    source = np.full((2, 3), 3, dtype=np.int32)
    np.testing.assert_array_equal(
        execute_reference(restored, inputs=[base, source]),
        np.multiply(np.maximum(base, 0), source, dtype=np.int32),
    )


def test_dynamic_inplace_binary_specializes_and_reuses_cache():
    _default_compiler_or_skip()
    from tiny_tensor_compiler import SymbolicDim

    batch = SymbolicDim("B")
    builder = GraphBuilder()
    base = builder.input((batch, 3), dtype="int32")
    source = builder.input((batch, 3), dtype="int32")
    root = base.relu()
    updated = root.binary_inplace(source, operator="add")
    module = builder.finish(updated)
    executable = compile_dynamic_module(module, borrow_inputs=True, parallel=True)

    for size in (2, 0, 5, 2):
        base_value = np.arange(size * 3, dtype=np.int32).reshape(size, 3) - 2
        source_value = np.full((size, 3), 4, dtype=np.int32)
        actual = executable(inputs=[base_value, source_value])
        expected = np.add(np.maximum(base_value, 0), source_value, dtype=np.int32)
        np.testing.assert_array_equal(actual, expected)

    assert executable.cached_batch_sizes == (0, 2, 5)
