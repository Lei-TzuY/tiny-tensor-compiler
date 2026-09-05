import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    StorageLayout,
    SymbolicDim,
    TypeInferenceError,
    borrow_inputs,
    compile_dynamic_module,
    compile_module,
    dead_code_eliminate,
    execute_loop,
    execute_reference,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
)


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def test_reverse_is_typed_zero_copy_view_with_reference_semantics():
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="float32")
    reversed_ = value.reverse(axis=1)
    module = builder.finish(reversed_)

    assert reversed_.type == value.type
    assert "%1 = reverse %0" in module.dump()

    runtime = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    actual = execute_reference(module, inputs=[runtime])
    np.testing.assert_array_equal(actual, np.flip(runtime, axis=1))


def test_reverse_layout_uses_negative_stride_without_allocation():
    builder = GraphBuilder()
    value = builder.input((3, 4), dtype="int32")
    reversed_ = value.reverse(axis=1)
    loops = lower_to_loops(lower_to_cpu(builder.finish(reversed_.relu())))

    assert len(loops.views) == 1
    view = loops.views[0]
    assert view.layout == StorageLayout(offset=3, strides=(4, -1))
    assert loops.storage_root(view.output) == loops.storage_root(view.source)
    assert view.output not in {alloc.buffer for alloc in loops.allocations}
    assert not view.layout.is_contiguous(view.type.shape)


def test_signed_layout_bounds_cover_minimum_and_maximum_reachable_offsets():
    StorageLayout(offset=3, strides=(-1,)).validate_bounds((4,), storage_elements=4)
    StorageLayout(offset=0, strides=(-1,)).validate_bounds((0,), storage_elements=0)

    with pytest.raises(ValueError, match="bounds"):
        StorageLayout(offset=0, strides=(-1,)).validate_bounds((4,), storage_elements=4)
    with pytest.raises(ValueError, match="non-zero"):
        StorageLayout(offset=0, strides=(0,))


def test_reverse_composes_with_positive_slice_and_transpose_layouts():
    builder = GraphBuilder()
    value = builder.input((3, 6), dtype="int32")
    sliced = value.slice(axis=1, start=1, stop=6, step=2)
    reversed_ = sliced.reverse(axis=1)
    transposed = reversed_.transpose((1, 0))
    loops = lower_to_loops(lower_to_cpu(builder.finish(transposed.relu())))

    assert len(loops.views) == 3
    assert loops.views[0].layout == StorageLayout(offset=1, strides=(6, 2))
    assert loops.views[1].layout == StorageLayout(offset=5, strides=(6, -2))
    assert loops.views[2].layout == StorageLayout(offset=5, strides=(-2, 6))
    assert loops.storage_root(loops.views[2].output) == loops.storage_root(loops.views[0].source)


def test_reverse_rejects_invalid_axis_and_does_not_expand_slice_syntax():
    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="float32")

    with pytest.raises(TypeInferenceError, match="axis"):
        value.reverse(axis=2)
    with pytest.raises(TypeInferenceError, match="axis"):
        value.reverse(axis=True)
    with pytest.raises(TypeInferenceError, match="step"):
        value.slice(axis=1, start=2, stop=0, step=-1)


def test_reverse_composes_with_borrowed_input_cpu_and_native_multi_output():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    value = builder.input((3, 4), dtype="float32")
    reversed_ = value.reverse(axis=1)
    module = builder.finish((reversed_, reversed_.relu()))
    loops = lower_to_loops(lower_to_cpu(module))
    borrowed = borrow_inputs(loops)

    runtime = np.linspace(-6.0, 5.0, 12, dtype=np.float32).reshape(3, 4)
    expected = execute_reference(module, inputs=[runtime])
    cpu_actual = execute_loop(borrowed, inputs=[runtime])
    native_actual = compile_module(module, borrow_inputs=True)(inputs=[runtime])

    assert isinstance(expected, tuple)
    assert isinstance(cpu_actual, tuple)
    assert isinstance(native_actual, tuple)
    for cpu_value, native_value, wanted in zip(
        cpu_actual, native_actual, expected, strict=True
    ):
        np.testing.assert_array_equal(cpu_value, wanted)
        np.testing.assert_array_equal(native_value, wanted)


def test_generated_c_uses_offset_pointer_negative_stride_and_general_indexing():
    builder = GraphBuilder()
    value = builder.input((2, 4), dtype="int32")
    reversed_ = value.reverse(axis=1)
    loops = lower_to_loops(lower_to_cpu(builder.finish(reversed_.relu())))
    view = loops.views[0]

    source = generate_c(loops)
    root = loops.storage_root(view.output)
    assert f"const int32_t *p{view.output} = p{root} + 3;" in source
    assert "* -1" in source
    assert "_mm_loadu_si128" not in source


def test_dynamic_reverse_on_symbolic_axis_specializes_and_reuses_native_cache():
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    value = builder.input((batch, 3), dtype="int32")
    reversed_ = value.reverse(axis=0)
    module = builder.finish(reversed_.relu())
    executable = compile_dynamic_module(module, borrow_inputs=True)

    assert reversed_.type.shape == (batch, 3)
    for batch_size in (2, 5, 0, 2):
        runtime = np.arange(batch_size * 3, dtype=np.int32).reshape(batch_size, 3) - 4
        actual = executable(inputs=[runtime])
        expected = execute_reference(module, inputs=[runtime])
        np.testing.assert_array_equal(actual, expected)

    assert executable.cached_batch_sizes == (0, 2, 5)


def test_unused_reverse_is_pure_for_dce():
    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="int32")
    value.reverse(axis=1)
    module = builder.finish(value.relu())

    assert dead_code_eliminate(module) == 1
    assert [op.opcode for op in module.function.ops] == ["input", "relu", "return"]
