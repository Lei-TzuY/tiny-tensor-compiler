import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    IndexMap,
    LoopAlloc,
    LoopInput,
    LoopKernel,
    LoopProgram,
    LoopReturn,
    LoopView,
    SymbolicDim,
    borrow_inputs,
    common_subexpression_eliminate,
    compile_dynamic_module,
    compile_module,
    dead_code_eliminate,
    execute_loop,
    execute_reference,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
    plan_memory,
)
from tiny_tensor_compiler.ir import DType, TensorType


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def test_view_is_typed_ir_and_preserves_c_order_semantics():
    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="float32")
    viewed = value.view((3, 2))
    module = builder.finish(viewed)

    assert viewed.type.shape == (3, 2)
    assert viewed.type.dtype == value.type.dtype
    assert "%1 = view %0" in module.dump()

    runtime = np.arange(6, dtype=np.float32).reshape(2, 3)
    actual = execute_reference(module, inputs=[runtime])
    expected = np.reshape(runtime, (3, 2), order="C")
    np.testing.assert_array_equal(actual, expected)


def test_memory_plan_aliases_view_and_extends_source_lifetime_through_view_users():
    builder = GraphBuilder()
    source = builder.input((2, 3), dtype="int32")
    viewed = source.view((3, 2))
    other = builder.tensor(np.full((2, 3), 99, dtype=np.int32))
    module = builder.finish((other, viewed.relu()))

    cpu = lower_to_cpu(module)
    plan = plan_memory(cpu)

    assert plan.physical_for(1) == plan.physical_for(0)
    assert plan.physical_for(2) != plan.physical_for(0)
    assert plan.physical_count == 3

    runtime = np.arange(6, dtype=np.int32).reshape(2, 3) - 3
    actual = execute_loop(lower_to_loops(cpu), inputs=[runtime])
    expected = execute_reference(module, inputs=[runtime])
    assert isinstance(actual, tuple)
    assert isinstance(expected, tuple)
    for result, wanted in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, wanted)


def test_loop_view_has_no_storage_allocation_and_storage_root_alias_blocks_writes():
    builder = GraphBuilder()
    source = builder.input((2, 3), dtype="int32")
    module = builder.finish(source.view((3, 2)).relu())
    loops = lower_to_loops(lower_to_cpu(module))

    assert len(loops.views) == 1
    view = loops.views[0]
    assert view.output not in {alloc.buffer for alloc in loops.allocations}
    assert loops.storage_root(view.output) == loops.storage_root(view.source)
    assert len(loops.allocations) == 2

    type_ = TensorType((2, 3), DType.INT32)
    identity = IndexMap((0, 1))
    with pytest.raises(ValueError, match="storage alias"):
        LoopProgram(
            (
                LoopAlloc(0, type_),
                LoopInput(0, 0),
                LoopView(1, 0, type_),
                LoopKernel("relu", 0, (1,), (2, 3), (identity,)),
                LoopReturn(0),
            )
        )


def test_loop_verifier_rejects_stale_view_after_storage_root_is_rewritten():
    type_ = TensorType((2, 3), DType.INT32)
    identity = IndexMap((0, 1))

    with pytest.raises(ValueError, match="stale.*view|stale.*alias"):
        LoopProgram(
            (
                LoopAlloc(0, type_),
                LoopAlloc(1, type_),
                LoopInput(0, 0),
                LoopView(2, 0, type_),
                LoopInput(1, 1),
                LoopKernel("relu", 0, (1,), (2, 3), (identity,)),
                LoopReturn(2),
            )
        )


def test_generated_c_uses_pointer_alias_without_view_copy_storage():
    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="float32")
    loops = lower_to_loops(lower_to_cpu(builder.finish(value.view((3, 2)))))
    view = loops.views[0]

    source = generate_c(loops)
    assert f"const float *p{view.output} = p{view.source};" in source
    assert f"float p{view.output}[" not in source
    assert f"p{view.output}[n] = p{view.source}[n];" not in source


def test_borrowed_input_view_remains_zero_copy_and_handle_ids_do_not_collide():
    builder = GraphBuilder()
    value = builder.input((3, 4), dtype="float32")
    viewed = value.view((2, 6))
    module = builder.finish((viewed, viewed.relu()))
    loops = lower_to_loops(lower_to_cpu(module))
    borrowed = borrow_inputs(loops)

    allocation_ids = {alloc.buffer for alloc in borrowed.allocations}
    view_ids = {view.output for view in borrowed.views}
    assert allocation_ids.isdisjoint(view_ids)
    assert borrowed.storage_root(next(iter(view_ids))) in borrowed.borrowed_input_slots

    runtime = np.linspace(-5.0, 6.0, 12, dtype=np.float32).reshape(3, 4)
    actual = execute_loop(borrowed, inputs=[runtime])
    expected = execute_reference(module, inputs=[runtime])
    assert isinstance(actual, tuple)
    assert isinstance(expected, tuple)
    for result, wanted in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, wanted)


def test_native_view_composes_with_borrowed_input_multi_output_and_downstream_kernel():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    value = builder.input((4, 6), dtype="float32")
    viewed = value.view((3, 8))
    module = builder.finish((viewed, viewed.relu()))
    runtime = np.linspace(-12.0, 11.0, 24, dtype=np.float32).reshape(4, 6)

    executable = compile_module(module, borrow_inputs=True)
    actual = executable(inputs=[runtime])
    expected = execute_reference(module, inputs=[runtime])

    assert isinstance(actual, tuple)
    assert isinstance(expected, tuple)
    for result, wanted in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, wanted)


def test_dynamic_symbolic_view_specializes_and_reuses_native_cache():
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    value = builder.input((batch, 4), dtype="int32")
    module = builder.finish(value.view((2, 2 * batch)).relu())
    executable = compile_dynamic_module(module, borrow_inputs=True)

    for batch_size in (2, 5, 2):
        runtime = np.arange(batch_size * 4, dtype=np.int32).reshape(batch_size, 4) - 3
        actual = executable(inputs=[runtime])
        expected = execute_reference(module, inputs=[runtime])
        np.testing.assert_array_equal(actual, expected)

    assert executable.cached_batch_sizes == (2, 5)


def test_view_is_pure_for_dce_and_exact_cse_but_remains_a_fusion_boundary():
    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="int32")
    first = value.view((3, 2))
    duplicate = value.view((3, 2))
    value.view((6,))
    module = builder.finish(first + duplicate)

    assert common_subexpression_eliminate(module) == 1
    assert dead_code_eliminate(module) == 1
    opcodes = [op.opcode for op in module.function.ops]
    assert opcodes == ["input", "view", "add", "return"]
