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


def test_interleaved_same_root_regions_remain_fail_closed():
    builder = GraphBuilder()
    base = builder.input((2, 4), dtype="int32")
    owned = base.relu()
    target = owned.slice(axis=1, start=0, stop=4, step=2)
    source = owned.slice(axis=1, start=1, stop=4, step=2)

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
