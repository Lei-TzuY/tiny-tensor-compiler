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


def test_positive_stride_slice_is_typed_zero_copy_view_with_reference_semantics():
    builder = GraphBuilder()
    value = builder.input((3, 6), dtype="float32")
    sliced = value.slice(axis=1, start=1, stop=6, step=2)
    module = builder.finish(sliced)

    assert sliced.type.shape == (3, 3)
    assert sliced.type.dtype == value.type.dtype
    assert "%1 = slice %0" in module.dump()

    runtime = np.arange(18, dtype=np.float32).reshape(3, 6)
    actual = execute_reference(module, inputs=[runtime])
    np.testing.assert_array_equal(actual, runtime[:, 1:6:2])


def test_slice_layout_tracks_absolute_offset_positive_strides_and_no_allocation():
    builder = GraphBuilder()
    value = builder.input((3, 6), dtype="int32")
    sliced = value.slice(axis=1, start=1, stop=6, step=2)
    loops = lower_to_loops(lower_to_cpu(builder.finish(sliced.relu())))

    assert len(loops.views) == 1
    view = loops.views[0]
    assert view.layout == StorageLayout(offset=1, strides=(6, 2))
    assert loops.storage_root(view.output) == loops.storage_root(view.source)
    assert view.output not in {alloc.buffer for alloc in loops.allocations}
    assert loops.value_layouts[view.output] == view.layout


def test_empty_contiguous_layout_keeps_strides_positive():
    layout = StorageLayout.contiguous((3, 0))

    assert layout.strides == (1, 1)
    layout.validate_bounds((3, 0), storage_elements=0)


def test_slice_rejects_invalid_bounds_step_and_symbolic_sliced_axis():
    builder = GraphBuilder()
    value = builder.input((3, 6), dtype="float32")

    with pytest.raises(TypeInferenceError, match="step"):
        value.slice(axis=1, start=0, stop=6, step=0)
    with pytest.raises(TypeInferenceError, match="bounds|stop|extent"):
        value.slice(axis=1, start=5, stop=7, step=1)

    batch = SymbolicDim("B")
    dynamic_builder = GraphBuilder()
    dynamic = dynamic_builder.input((batch, 6), dtype="float32")
    with pytest.raises(TypeInferenceError, match="slice axis.*concrete|concrete.*slice axis"):
        dynamic.slice(axis=0, start=0, stop=1, step=1)


def test_slice_composes_with_borrowed_input_cpu_and_native_multi_output():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    value = builder.input((4, 6), dtype="float32")
    sliced = value.slice(axis=1, start=1, stop=6, step=2)
    module = builder.finish((sliced, sliced.relu()))
    loops = lower_to_loops(lower_to_cpu(module))
    borrowed = borrow_inputs(loops)

    runtime = np.linspace(-12.0, 11.0, 24, dtype=np.float32).reshape(4, 6)
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


def test_generated_c_uses_offset_pointer_and_strided_logical_indexing():
    builder = GraphBuilder()
    value = builder.input((3, 6), dtype="int32")
    sliced = value.slice(axis=1, start=1, stop=6, step=2)
    loops = lower_to_loops(lower_to_cpu(builder.finish(sliced.relu())))
    view = loops.views[0]

    source = generate_c(loops)
    root = loops.storage_root(view.output)
    assert f"const int32_t *p{view.output} = p{root} + 1;" in source
    assert f"int32_t p{view.output}[" not in source
    assert "* 2" in source


def test_dynamic_slice_on_concrete_axis_specializes_and_reuses_native_cache():
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    value = builder.input((batch, 6), dtype="int32")
    sliced = value.slice(axis=1, start=1, stop=6, step=2)
    module = builder.finish(sliced.relu())
    executable = compile_dynamic_module(module, borrow_inputs=True)

    for batch_size in (2, 5, 2):
        runtime = np.arange(batch_size * 6, dtype=np.int32).reshape(batch_size, 6) - 7
        actual = executable(inputs=[runtime])
        expected = execute_reference(module, inputs=[runtime])
        np.testing.assert_array_equal(actual, expected)

    assert executable.cached_batch_sizes == (2, 5)


def test_unused_slice_is_pure_for_dce():
    builder = GraphBuilder()
    value = builder.input((3, 6), dtype="int32")
    value.slice(axis=1, start=1, stop=6, step=2)
    module = builder.finish(value.relu())

    assert dead_code_eliminate(module) == 1
    assert [op.opcode for op in module.function.ops] == ["input", "relu", "return"]