import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    SymbolicDim,
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


def _disjoint_row_copy_module():
    builder = GraphBuilder()
    base = builder.input((2, 4), dtype="int32")
    owned = base.relu()
    source = owned.slice(axis=0, start=0, stop=1)
    target = owned.slice(axis=0, start=1, stop=2)
    updated = owned.copy_into(target, source)
    return builder.finish(updated)


def test_disjoint_same_root_copy_is_desugared_through_explicit_snapshot():
    module = _disjoint_row_copy_module()
    verify(module)

    copy_op = next(op for op in module.function.ops if op.opcode == "copy_into")
    snapshot = copy_op.operands[2]
    assert snapshot.producer is not None
    assert snapshot.producer.opcode == "reshape"
    assert snapshot.producer.operands[0].producer is not None
    assert snapshot.producer.operands[0].producer.opcode == "slice"

    # The canonical tensor IR still satisfies the existing low-level invariant:
    # copy_into reads its source from a distinct owning storage root.
    assert copy_op.operands[2] is snapshot


def test_disjoint_same_root_copy_matches_reference_loop_and_native():
    _default_compiler_or_skip()
    module = _disjoint_row_copy_module()
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


def test_disjoint_reverse_source_snapshots_logical_order_across_native():
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


def test_interleaved_arithmetic_regions_are_proven_disjoint_and_execute_natively():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    base = builder.input((2, 4), dtype="int32")
    owned = base.relu()
    target = owned.slice(axis=1, start=0, stop=4, step=2)
    source = owned.slice(axis=1, start=1, stop=4, step=2)
    updated = owned.copy_into(target, source)
    module = builder.finish(updated)

    copy_op = next(op for op in module.function.ops if op.opcode == "copy_into")
    snapshot = copy_op.operands[2]
    assert snapshot.producer is not None
    assert snapshot.producer.opcode == "reshape"

    base_value = np.array([[-4, 2, 7, 1], [20, 21, 22, 23]], dtype=np.int32)
    expected = np.maximum(base_value, 0)
    expected[:, 0::2] = expected[:, 1::2]

    reference = execute_reference(module, inputs=[base_value])
    loop = execute_loop(lower_to_loops(lower_to_cpu(module)), inputs=[base_value])
    native = compile_module(module, borrow_inputs=True, parallel=True)(inputs=[base_value])

    np.testing.assert_array_equal(reference, expected)
    np.testing.assert_array_equal(loop, expected)
    np.testing.assert_array_equal(native, expected)


def test_overlapping_arithmetic_regions_are_rejected():
    builder = GraphBuilder()
    base = builder.input((2, 4), dtype="int32")
    owned = base.relu()
    target = owned.slice(axis=1, start=0, stop=4, step=2)
    source = owned.slice(axis=1, start=0, stop=4, step=2)

    with pytest.raises(ValueError, match="different storage root"):
        owned.copy_into(target, source)


def test_non_progression_interleaved_regions_remain_fail_closed():
    builder = GraphBuilder()
    base = builder.input((2, 5), dtype="int32")
    owned = base.relu()
    target = owned.slice(axis=1, start=0, stop=4, step=2)
    source = owned.slice(axis=1, start=1, stop=4, step=2)

    # Exact storage sets are {0, 2, 5, 7} and {1, 3, 6, 8}, but neither
    # region is one finite arithmetic progression. This phase deliberately
    # refuses to enumerate or solve a general multidimensional lattice.
    with pytest.raises(ValueError, match="different storage root"):
        owned.copy_into(target, source)


def test_symbolic_same_root_regions_require_a_concrete_disjointness_proof():
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    base = builder.input((batch, 4), dtype="int32")
    owned = base.relu()
    target = owned.slice(axis=1, start=0, stop=2)
    source = owned.slice(axis=1, start=2, stop=4)

    with pytest.raises(ValueError, match="different storage root"):
        owned.copy_into(target, source)
