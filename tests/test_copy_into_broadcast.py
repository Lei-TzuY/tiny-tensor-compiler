import os
import shutil
from dataclasses import replace

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    IndexMap,
    LoopCopyInto,
    LoopProgram,
    SymbolicDim,
    compile_dynamic_module,
    compile_module,
    execute_reference,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
)


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _broadcast_copy_module():
    builder = GraphBuilder()
    base = builder.input((2, 6), dtype="int32")
    source = builder.input((3,), dtype="int32")
    root = base.relu()
    target = root.slice(axis=1, start=0, stop=6, step=2)
    return builder.finish(root.copy_into(target, source))


def _replace_copy(loops: LoopProgram, replacement: LoopCopyInto) -> LoopProgram:
    return LoopProgram(
        tuple(
            replacement if isinstance(op, LoopCopyInto) else op
            for op in loops.operations
        )
    )


def test_copy_into_broadcasts_lower_rank_source_in_reference_execution():
    module = _broadcast_copy_module()
    base = np.arange(12, dtype=np.int32).reshape(2, 6) - 3
    source = np.array([10, 20, 30], dtype=np.int32)

    actual = execute_reference(module, inputs=[base, source])
    expected = np.maximum(base, 0)
    expected[:, 0:6:2] = source
    np.testing.assert_array_equal(actual, expected)


def test_copy_into_broadcasts_scalar_source_into_strided_target():
    builder = GraphBuilder()
    base = builder.input((2, 6), dtype="float32")
    source = builder.input((), dtype="float32")
    root = base.relu()
    target = root.slice(axis=1, start=1, stop=6, step=2)
    module = builder.finish(root.copy_into(target, source))

    base_value = np.arange(12, dtype=np.float32).reshape(2, 6) - 4
    source_value = np.array(2.5, dtype=np.float32)
    actual = execute_reference(module, inputs=[base_value, source_value])
    expected = np.maximum(base_value, 0)
    expected[:, 1:6:2] = source_value
    np.testing.assert_array_equal(actual, expected)


def test_copy_into_broadcast_lowering_and_generated_c_use_source_map():
    module = _broadcast_copy_module()
    loops = lower_to_loops(lower_to_cpu(module))
    effect = loops.copies[0]

    assert effect.source_map == IndexMap((1,))
    generated = generate_c(loops, parallel=True)
    assert f"p{effect.source}[i1]" in generated


def test_copy_into_broadcast_native_with_borrowed_inputs_and_parallel_module():
    _default_compiler_or_skip()
    module = _broadcast_copy_module()
    base = np.arange(12, dtype=np.int32).reshape(2, 6) - 3
    source = np.array([7, 11, 13], dtype=np.int32)

    actual = compile_module(module, borrow_inputs=True, parallel=True)(inputs=[base, source])
    expected = np.maximum(base, 0)
    expected[:, 0:6:2] = source
    np.testing.assert_array_equal(actual, expected)


def test_copy_into_broadcast_dynamic_specialization_reuses_binding_cache():
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    base = builder.input((batch, 6), dtype="int32")
    source = builder.input((3,), dtype="int32")
    root = base.relu()
    target = root.slice(axis=1, start=0, stop=6, step=2)
    module = builder.finish(root.copy_into(target, source))
    executable = compile_dynamic_module(module, borrow_inputs=True, parallel=True)

    source_value = np.array([3, 5, 7], dtype=np.int32)
    for size in (2, 0, 4, 2):
        base_value = np.arange(size * 6, dtype=np.int32).reshape(size, 6) - 2
        actual = executable(inputs=[base_value, source_value])
        expected = np.maximum(base_value, 0)
        expected[:, 0:6:2] = source_value
        np.testing.assert_array_equal(actual, expected)

    assert executable.cached_batch_sizes == (0, 2, 4)


def test_copy_into_same_root_broadcast_source_is_snapshotted_before_write():
    builder = GraphBuilder()
    base = builder.input((2, 3), dtype="int32")
    root = base.relu()
    source = root.slice(axis=0, start=0, stop=1).view((3,))
    module = builder.finish(root.copy_into(root, source))

    base_value = np.array([[-1, 2, 3], [4, 5, 6]], dtype=np.int32)
    actual = execute_reference(module, inputs=[base_value])
    first = np.maximum(base_value, 0)[0]
    expected = np.stack((first, first))
    np.testing.assert_array_equal(actual, expected)


def test_broadcast_copy_into_requires_explicit_canonical_source_map():
    loops = lower_to_loops(lower_to_cpu(_broadcast_copy_module()))
    effect = loops.copies[0]
    assert effect.source_map == IndexMap((1,))

    with pytest.raises(ValueError, match="requires an explicit source index map"):
        _replace_copy(loops, replace(effect, source_map=None))
    with pytest.raises(ValueError, match="source index map does not match broadcasting semantics"):
        _replace_copy(loops, replace(effect, source_map=IndexMap((0,))))


def test_exact_shape_copy_into_keeps_optional_source_map_compatibility():
    builder = GraphBuilder()
    base = builder.input((2, 3), dtype="int32")
    source = builder.input((2, 3), dtype="int32")
    root = base.relu()
    loops = lower_to_loops(lower_to_cpu(builder.finish(root.copy_into(root, source))))
    effect = loops.copies[0]
    assert effect.source_map is None
    _replace_copy(loops, replace(effect, source_map=None))


def test_copy_into_broadcast_rejects_dtype_change_and_nonbroadcastable_source():
    builder = GraphBuilder()
    base = builder.input((2, 6), dtype="int32")
    root = base.relu()
    target = root.slice(axis=1, start=0, stop=6, step=2)
    wrong_dtype = builder.input((3,), dtype="int64")
    with pytest.raises(ValueError, match="dtype"):
        root.copy_into(target, wrong_dtype)

    builder = GraphBuilder()
    base = builder.input((2, 6), dtype="int32")
    root = base.relu()
    target = root.slice(axis=1, start=0, stop=6, step=2)
    wrong_shape = builder.input((2, 2), dtype="int32")
    with pytest.raises(ValueError, match="broadcast"):
        root.copy_into(target, wrong_shape)
