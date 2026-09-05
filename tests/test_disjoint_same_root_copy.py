import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    SymbolicDim,
    compile_dynamic_module,
    compile_module,
    execute_loop,
    execute_reference,
    lower_to_cpu,
    lower_to_loops,
    verify,
)


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _same_root_row_copy_module():
    builder = GraphBuilder()
    base = builder.input((2, 4), dtype="int32")
    owned = base.relu()
    source = owned.slice(axis=0, start=0, stop=1)
    target = owned.slice(axis=0, start=1, stop=2)
    updated = owned.copy_into(target, source)
    return builder.finish(updated)


def test_same_root_copy_is_desugared_through_explicit_snapshot():
    module = _same_root_row_copy_module()
    verify(module)

    copy_op = next(op for op in module.function.ops if op.opcode == "copy_into")
    snapshot = copy_op.operands[2]
    assert snapshot.producer is not None
    assert snapshot.producer.opcode == "reshape"
    assert snapshot.producer.operands[0].producer is not None
    assert snapshot.producer.operands[0].producer.opcode == "slice"

    # Canonical tensor IR still satisfies the historical low-level invariant:
    # copy_into reads its source from a distinct owning storage root.
    assert snapshot.producer.operands[0] is not snapshot


def test_same_root_copy_matches_reference_loop_and_native():
    _default_compiler_or_skip()
    module = _same_root_row_copy_module()
    base = np.array([[-3, 2, 5, -7], [11, 12, 13, 14]], dtype=np.int32)
    expected = np.maximum(base, 0)
    expected[1, :] = expected[0, :]

    reference = execute_reference(module, inputs=[base])
    loop = execute_loop(lower_to_loops(lower_to_cpu(module)), inputs=[base])
    native = compile_module(module, borrow_inputs=True, parallel=True)(inputs=[base])

    np.testing.assert_array_equal(reference, expected)
    np.testing.assert_array_equal(loop, expected)
    np.testing.assert_array_equal(native, expected)
    np.testing.assert_array_equal(
        base,
        np.array([[-3, 2, 5, -7], [11, 12, 13, 14]], dtype=np.int32),
    )


def test_reverse_source_snapshots_logical_order_across_native():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    base = builder.input((2, 4), dtype="int32")
    owned = base.relu()
    source = owned.slice(axis=0, start=0, stop=1).reverse(axis=1)
    target = owned.slice(axis=0, start=1, stop=2)
    updated = owned.copy_into(target, source)
    module = builder.finish(updated)

    copy_op = next(op for op in module.function.ops if op.opcode == "copy_into")
    snapshot = copy_op.operands[2]
    assert snapshot.producer is not None
    assert snapshot.producer.opcode == "reshape"
    assert snapshot.producer.operands[0].producer is not None
    assert snapshot.producer.operands[0].producer.opcode == "reverse"

    base_value = np.array([[-4, 2, 7, 1], [20, 21, 22, 23]], dtype=np.int32)
    expected = np.maximum(base_value, 0)
    expected[1, :] = expected[0, ::-1]

    reference = execute_reference(module, inputs=[base_value])
    loop = execute_loop(lower_to_loops(lower_to_cpu(module)), inputs=[base_value])
    native = compile_module(module, borrow_inputs=True, parallel=True)(inputs=[base_value])

    np.testing.assert_array_equal(reference, expected)
    np.testing.assert_array_equal(loop, expected)
    np.testing.assert_array_equal(native, expected)


def test_shifted_overlapping_same_root_copy_uses_prewrite_snapshot():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    base = builder.input((2, 4), dtype="int32")
    owned = base.relu()
    source = owned.slice(axis=1, start=0, stop=3)
    target = owned.slice(axis=1, start=1, stop=4)
    updated = owned.copy_into(target, source)
    module = builder.finish(updated)

    copy_op = next(op for op in module.function.ops if op.opcode == "copy_into")
    assert copy_op.operands[2].producer is not None
    assert copy_op.operands[2].producer.opcode == "reshape"

    base_value = np.array([[-4, 2, 7, 1], [8, 9, 10, 11]], dtype=np.int32)
    expected = np.maximum(base_value, 0)
    snapshot = np.array(expected[:, 0:3], copy=True)
    expected[:, 1:4] = snapshot

    reference = execute_reference(module, inputs=[base_value])
    loop = execute_loop(lower_to_loops(lower_to_cpu(module)), inputs=[base_value])
    native = compile_module(module, borrow_inputs=True, parallel=True)(inputs=[base_value])

    np.testing.assert_array_equal(reference, expected)
    np.testing.assert_array_equal(loop, expected)
    np.testing.assert_array_equal(native, expected)


def test_interleaved_same_root_regions_use_snapshot_semantics():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    base = builder.input((2, 4), dtype="int32")
    owned = base.relu()
    target = owned.slice(axis=1, start=0, stop=4, step=2)
    source = owned.slice(axis=1, start=1, stop=4, step=2)
    updated = owned.copy_into(target, source)
    module = builder.finish(updated)

    base_value = np.array([[-4, 2, 7, 1], [8, 9, 10, 11]], dtype=np.int32)
    expected = np.maximum(base_value, 0)
    snapshot = np.array(expected[:, 1:4:2], copy=True)
    expected[:, 0:4:2] = snapshot

    reference = execute_reference(module, inputs=[base_value])
    loop = execute_loop(lower_to_loops(lower_to_cpu(module)), inputs=[base_value])
    native = compile_module(module, borrow_inputs=True, parallel=True)(inputs=[base_value])

    np.testing.assert_array_equal(reference, expected)
    np.testing.assert_array_equal(loop, expected)
    np.testing.assert_array_equal(native, expected)


def test_symbolic_overlapping_same_root_copy_specializes_and_reuses_cache():
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    base = builder.input((batch, 4), dtype="int32")
    owned = base.relu()
    source = owned.slice(axis=1, start=0, stop=3)
    target = owned.slice(axis=1, start=1, stop=4)
    updated = owned.copy_into(target, source)
    module = builder.finish(updated)
    executable = compile_dynamic_module(module, borrow_inputs=True, parallel=True)

    for size in (2, 5, 0, 2):
        base_value = np.arange(size * 4, dtype=np.int32).reshape(size, 4) - 3
        expected = np.maximum(base_value, 0)
        snapshot = np.array(expected[:, 0:3], copy=True)
        expected[:, 1:4] = snapshot
        actual = executable(inputs=[base_value])
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            base_value,
            np.arange(size * 4, dtype=np.int32).reshape(size, 4) - 3,
        )

    assert executable.cached_batch_sizes == (0, 2, 5)