import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    VerificationError,
    algebraic_simplify,
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
from tiny_tensor_compiler.ir import DType, TensorType


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _slice_update_module():
    builder = GraphBuilder()
    base = builder.input((3, 6), dtype="int32")
    patch = builder.input((3, 3), dtype="int32")
    owned = base + 0
    target = owned.slice(axis=1, start=1, stop=6, step=2)
    updated = owned.copy_into(target, patch)
    return builder.finish(updated)


def test_copy_into_updates_internal_slice_and_returns_fresh_root_reference():
    module = _slice_update_module()
    base = np.arange(18, dtype=np.int32).reshape(3, 6)
    patch = (100 + np.arange(9, dtype=np.int32)).reshape(3, 3)

    actual = execute_reference(module, inputs=[base, patch])
    expected = np.array(base, copy=True)
    expected[:, 1:6:2] = patch

    np.testing.assert_array_equal(actual, expected)
    assert "copy_into" in module.dump()
    np.testing.assert_array_equal(base, np.arange(18, dtype=np.int32).reshape(3, 6))


def test_copy_into_result_is_alias_without_new_physical_storage():
    module = _slice_update_module()
    cpu = lower_to_cpu(module)
    plan = plan_memory(cpu)
    loops = lower_to_loops(cpu)

    assert len(cpu.copies) == 1
    copy = cpu.copies[0]
    alias = plan.alias_for(copy.output)
    assert alias is not None
    assert alias.physical == plan.physical_for(copy.root)
    assert len(loops.copies) == 1
    loop_copy = loops.copies[0]
    assert loops.storage_root(loop_copy.output) == loop_copy.root
    assert loop_copy.output not in {alloc.buffer for alloc in loops.allocations}


def test_copy_into_handles_negative_stride_destination_across_cpu_and_native():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    base = builder.input((2, 4), dtype="float32")
    patch = builder.input((2, 4), dtype="float32")
    owned = base.relu()
    target = owned.reverse(axis=1)
    updated = owned.copy_into(target, patch)
    module = builder.finish(updated)
    loops = lower_to_loops(lower_to_cpu(module))
    borrowed = borrow_inputs(loops)

    base_value = np.array([[-3, -2, 1, 2], [4, -1, 5, -6]], dtype=np.float32)
    patch_value = np.arange(8, dtype=np.float32).reshape(2, 4) + 20
    expected = np.maximum(base_value, 0)
    expected[:, ::-1] = patch_value

    cpu_actual = execute_loop(borrowed, inputs=[base_value, patch_value])
    native_actual = compile_module(module, borrow_inputs=True, parallel=True)(
        inputs=[base_value, patch_value]
    )
    np.testing.assert_array_equal(cpu_actual, expected)
    np.testing.assert_array_equal(native_actual, expected)
    np.testing.assert_array_equal(
        base_value,
        np.array([[-3, -2, 1, 2], [4, -1, 5, -6]], dtype=np.float32),
    )


def test_generated_c_writes_root_through_target_layout_and_defines_fresh_alias():
    module = _slice_update_module()
    loops = lower_to_loops(lower_to_cpu(module))
    copy = loops.copies[0]
    source = generate_c(loops)

    assert "copy_into" not in source
    assert f"p{copy.root}[" in source
    assert f"const int32_t *p{copy.output} = p{copy.root};" in source
    assert "* 2" in source


def test_copy_into_rejects_input_or_const_storage_targets():
    builder = GraphBuilder()
    root = builder.input((2, 2), dtype="int32")
    patch = builder.input((2, 2), dtype="int32")
    with pytest.raises(ValueError, match="internal computed storage"):
        root.copy_into(root.view((2, 2)), patch)

    builder = GraphBuilder()
    root = builder.tensor(np.zeros((2, 2), dtype=np.int32))
    patch = builder.input((2, 2), dtype="int32")
    with pytest.raises(ValueError, match="internal computed storage"):
        root.copy_into(root.view((2, 2)), patch)


def test_copy_into_rejects_same_root_source_and_type_mismatch():
    builder = GraphBuilder()
    base = builder.input((2, 4), dtype="int32")
    owned = base.relu()
    target = owned.slice(axis=1, start=0, stop=4, step=2)
    same_root_source = owned.slice(axis=1, start=1, stop=4, step=2)
    with pytest.raises(ValueError, match="different storage root"):
        owned.copy_into(target, same_root_source)

    patch = builder.input((2, 3), dtype="int32")
    with pytest.raises(ValueError, match="exactly match"):
        owned.copy_into(target, patch)


def test_copy_into_is_terminal_effect_and_old_handles_cannot_be_used_after_write():
    builder = GraphBuilder()
    base = builder.input((2, 4), dtype="int32")
    patch = builder.input((2, 2), dtype="int32")
    owned = base.relu()
    target = owned.slice(axis=1, start=0, stop=4, step=2)
    owned.copy_into(target, patch)
    stale_use = owned.relu()
    module = builder.finish(stale_use)

    with pytest.raises(VerificationError, match="copy_into must be the final effect"):
        verify(module)

    builder = GraphBuilder()
    base = builder.input((2, 4), dtype="int32")
    patch = builder.input((2, 2), dtype="int32")
    owned = base.relu()
    target = owned.slice(axis=1, start=0, stop=4, step=2)
    updated = owned.copy_into(target, patch)
    module = builder.finish((updated, target))
    with pytest.raises(VerificationError, match="stale alias"):
        verify(module)


def test_copy_into_loop_verifier_rejects_same_storage_source():
    from tiny_tensor_compiler import (
        IndexMap,
        LoopAlloc,
        LoopCopyInto,
        LoopInput,
        LoopKernel,
        LoopProgram,
        LoopReturn,
        LoopView,
        StorageLayout,
    )

    type_ = TensorType((4,), DType.INT32)
    layout = StorageLayout.contiguous((4,))
    with pytest.raises(ValueError, match="different storage root"):
        LoopProgram(
            (
                LoopAlloc(0, type_),
                LoopAlloc(1, type_),
                LoopInput(1, 0),
                LoopKernel("relu", 0, (1,), (4,), (IndexMap((0,)),)),
                LoopView(2, 0, type_, layout),
                LoopCopyInto(3, 0, 2, 0, type_, layout),
                LoopReturn(3),
            )
        )


def test_effectful_module_blocks_reordering_simplification_but_remains_verifiable():
    module = _slice_update_module()
    assert algebraic_simplify(module) == 0
    verify(module)


def test_dynamic_copy_into_specializes_symbolic_unsliced_axis_and_reuses_cache():
    _default_compiler_or_skip()
    from tiny_tensor_compiler import SymbolicDim

    batch = SymbolicDim("B")
    builder = GraphBuilder()
    base = builder.input((batch, 6), dtype="int32")
    patch = builder.input((batch, 3), dtype="int32")
    owned = base.relu()
    target = owned.slice(axis=1, start=1, stop=6, step=2)
    updated = owned.copy_into(target, patch)
    module = builder.finish(updated)
    executable = compile_dynamic_module(module, borrow_inputs=True)

    for size in (2, 5, 0, 2):
        base_value = np.arange(size * 6, dtype=np.int32).reshape(size, 6) - 5
        patch_value = 50 + np.arange(size * 3, dtype=np.int32).reshape(size, 3)
        actual = executable(inputs=[base_value, patch_value])
        expected = np.maximum(base_value, 0)
        expected[:, 1:6:2] = patch_value
        np.testing.assert_array_equal(actual, expected)

    assert executable.cached_batch_sizes == (0, 2, 5)
