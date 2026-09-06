import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    IndexMap,
    LoopAlloc,
    LoopBinaryInto,
    LoopInput,
    LoopKernel,
    LoopProgram,
    LoopReturn,
    LoopView,
    StorageLayout,
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
from tiny_tensor_compiler.ir import DType, TensorType
from tiny_tensor_compiler.serialization import deserialize_module, serialize_module


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _slice_binary_into_module(operator: str = "add"):
    builder = GraphBuilder()
    base = builder.input((3, 6), dtype="int32")
    source = builder.input((3, 3), dtype="int32")
    root = base.relu()
    target = root.slice(axis=1, start=1, stop=6, step=2)
    updated = root.binary_into(target, source, operator=operator)
    return builder.finish(updated)


@pytest.mark.parametrize("operator", ["add", "mul"])
def test_binary_into_updates_only_target_region_and_matches_reference(operator: str):
    module = _slice_binary_into_module(operator)
    base = np.arange(18, dtype=np.int32).reshape(3, 6) - 4
    source = (10 + np.arange(9, dtype=np.int32)).reshape(3, 3)

    actual = execute_reference(module, inputs=[base, source])
    expected = np.maximum(base, 0)
    target = expected[:, 1:6:2]
    if operator == "add":
        target[...] = np.add(target, source, dtype=np.int32)
    else:
        target[...] = np.multiply(target, source, dtype=np.int32)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(base, np.arange(18, dtype=np.int32).reshape(3, 6) - 4)


def test_binary_into_reuses_root_storage_and_returns_fresh_full_root_handle():
    module = _slice_binary_into_module()
    cpu = lower_to_cpu(module)
    plan = plan_memory(cpu)
    loops = lower_to_loops(cpu)

    assert len(cpu.binary_intos) == 1
    effect = cpu.binary_intos[0]
    alias = plan.alias_for(effect.output)
    assert alias is not None
    assert alias.physical == plan.physical_for(effect.root)

    assert len(loops.binary_intos) == 1
    loop_effect = loops.binary_intos[0]
    assert loops.storage_root(loop_effect.output) == loops.storage_root(loop_effect.root)
    assert loop_effect.output not in {alloc.buffer for alloc in loops.allocations}


def test_binary_into_same_root_shifted_source_uses_snapshot_before_write():
    builder = GraphBuilder()
    base = builder.input((6,), dtype="int32")
    root = base.relu()
    target = root.slice(axis=0, start=1, stop=6, step=1)
    source = root.slice(axis=0, start=0, stop=5, step=1)
    updated = root.binary_into(target, source, operator="add")
    module = builder.finish(updated)

    value = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    actual = execute_reference(module, inputs=[value])
    expected = value.copy()
    expected[1:6] = value[1:6] + value[0:5]

    np.testing.assert_array_equal(actual, expected)
    assert "reshape" in module.dump()
    assert "binary_into" in module.dump()


def test_binary_into_same_root_snapshot_matches_native_execution():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    base = builder.input((6,), dtype="int32")
    root = base.relu()
    target = root.slice(axis=0, start=1, stop=6, step=1)
    source = root.slice(axis=0, start=0, stop=5, step=1)
    module = builder.finish(root.add_into(target, source))

    value = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    expected = value.copy()
    expected[1:6] = value[1:6] + value[0:5]

    actual = compile_module(module, parallel=True)(inputs=[value])
    np.testing.assert_array_equal(actual, expected)


def test_binary_into_advances_generation_and_rejects_stale_aliases():
    builder = GraphBuilder()
    base = builder.input((6,), dtype="int32")
    source = builder.input((3,), dtype="int32")
    root = base.relu()
    target = root.slice(axis=0, start=0, stop=6, step=2)
    updated = root.binary_into(target, source, operator="add")
    verify(builder.finish(updated.relu()))

    builder = GraphBuilder()
    base = builder.input((6,), dtype="int32")
    source = builder.input((3,), dtype="int32")
    root = base.relu()
    target = root.slice(axis=0, start=0, stop=6, step=2)
    root.binary_into(target, source, operator="add")
    with pytest.raises(VerificationError, match="stale tensor view/alias"):
        verify(builder.finish(target.relu()))


def test_binary_into_rejects_unsafe_root_target_source_types_and_operator():
    builder = GraphBuilder()
    root = builder.input((4,), dtype="int32")
    source = builder.input((2,), dtype="int32")
    target = root.slice(axis=0, start=0, stop=4, step=2)
    with pytest.raises(ValueError, match="internal computed storage"):
        root.binary_into(target, source, operator="add")

    builder = GraphBuilder()
    base = builder.input((4,), dtype="int32")
    other = builder.input((4,), dtype="int32")
    root = base.relu()
    target = other.slice(axis=0, start=0, stop=4, step=2)
    source = builder.input((2,), dtype="int32")
    with pytest.raises(ValueError, match="target must alias"):
        root.binary_into(target, source, operator="add")

    builder = GraphBuilder()
    base = builder.input((4,), dtype="int32")
    root = base.relu()
    target = root.slice(axis=0, start=0, stop=4, step=2)
    wrong = builder.input((3,), dtype="int32")
    with pytest.raises(ValueError, match="exactly match"):
        root.binary_into(target, wrong, operator="add")

    builder = GraphBuilder()
    base = builder.input((4,), dtype="int32")
    root = base.relu()
    target = root.slice(axis=0, start=0, stop=4, step=2)
    source = builder.input((2,), dtype="int32")
    with pytest.raises(ValueError, match="operator"):
        root.binary_into(target, source, operator="sub")


def test_binary_into_loop_verifier_rejects_self_overlapping_target_layout():
    full = TensorType((4,), DType.INT32)
    partial = TensorType((2,), DType.INT32)
    overlapping = StorageLayout(offset=0, strides=(0,))

    with pytest.raises(ValueError, match="must not overlap itself"):
        LoopProgram(
            (
                LoopAlloc(0, full),
                LoopAlloc(1, full),
                LoopAlloc(2, partial),
                LoopInput(0, 0),
                LoopInput(2, 1),
                LoopKernel(
                    opcode="relu",
                    output=1,
                    inputs=(0,),
                    iteration_shape=(4,),
                    input_maps=(IndexMap((0,)),),
                ),
                LoopView(3, 1, partial, overlapping),
                LoopBinaryInto(
                    output=4,
                    root=1,
                    target=3,
                    source=2,
                    operator="add",
                    type=full,
                    layout=StorageLayout.contiguous((4,)),
                ),
                LoopReturn(4),
            )
        )


def test_binary_into_negative_stride_target_runs_cpu_native_and_parallel_with_borrowed_source():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    base = builder.input((2, 4), dtype="float32")
    source = builder.input((2, 4), dtype="float32")
    root = base.relu()
    target = root.reverse(axis=1)
    updated = root.binary_into(target, source, operator="mul")
    module = builder.finish(updated)
    loops = lower_to_loops(lower_to_cpu(module))
    borrowed = borrow_inputs(loops)

    base_value = np.array([[-3, 2, 4, 5], [6, -7, 1, 3]], dtype=np.float32)
    source_value = np.arange(8, dtype=np.float32).reshape(2, 4) + 2
    expected = np.maximum(base_value, 0)
    reversed_target = expected[:, ::-1]
    reversed_target[...] = reversed_target * source_value

    np.testing.assert_array_equal(execute_loop(borrowed, inputs=[base_value, source_value]), expected)
    actual = compile_module(module, borrow_inputs=True, parallel=True)(
        inputs=[base_value, source_value]
    )
    np.testing.assert_array_equal(actual, expected)


def test_binary_into_generated_c_is_serial_target_layout_update():
    loops = lower_to_loops(lower_to_cpu(_slice_binary_into_module("add")))
    effect = loops.binary_intos[0]
    source = generate_c(loops, parallel=True)

    assert f"p{effect.root}[" in source
    assert f"p{effect.source}[" in source
    assert f"int32_t *p{effect.output} = p{effect.root};" in source
    assert "* 2" in source
    effect_start = source.index(f"int32_t *p{effect.output} = p{effect.root};")
    effect_prefix = source[max(0, effect_start - 1200) : effect_start]
    assert "#pragma omp parallel for" not in effect_prefix


def test_binary_into_serialization_and_dynamic_specialization():
    restored = deserialize_module(serialize_module(_slice_binary_into_module("mul")))
    assert "binary_into" in restored.dump()
    verify(restored)

    _default_compiler_or_skip()
    from tiny_tensor_compiler import SymbolicDim

    batch = SymbolicDim("B")
    builder = GraphBuilder()
    base = builder.input((batch, 6), dtype="int32")
    source = builder.input((batch, 3), dtype="int32")
    root = base.relu()
    target = root.slice(axis=1, start=1, stop=6, step=2)
    updated = root.binary_into(target, source, operator="add")
    module = builder.finish(updated)
    executable = compile_dynamic_module(module, borrow_inputs=True, parallel=True)

    for size in (2, 0, 5, 2):
        base_value = np.arange(size * 6, dtype=np.int32).reshape(size, 6) - 5
        source_value = 50 + np.arange(size * 3, dtype=np.int32).reshape(size, 3)
        actual = executable(inputs=[base_value, source_value])
        expected = np.maximum(base_value, 0)
        target_value = expected[:, 1:6:2]
        target_value[...] = target_value + source_value
        np.testing.assert_array_equal(actual, expected)

    assert executable.cached_batch_sizes == (0, 2, 5)
