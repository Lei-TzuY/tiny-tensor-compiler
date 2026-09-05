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


def test_transpose_is_typed_zero_copy_view_with_reference_semantics():
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="float32")
    transposed = value.transpose((2, 0, 1))
    module = builder.finish(transposed)

    assert transposed.type.shape == (4, 2, 3)
    assert transposed.type.dtype == value.type.dtype
    assert "%1 = transpose %0" in module.dump()

    runtime = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    actual = execute_reference(module, inputs=[runtime])
    np.testing.assert_array_equal(actual, np.transpose(runtime, (2, 0, 1)))


def test_transpose_layout_permutes_positive_strides_without_allocation():
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="int32")
    transposed = value.transpose((2, 0, 1))
    loops = lower_to_loops(lower_to_cpu(builder.finish(transposed.relu())))

    assert len(loops.views) == 1
    view = loops.views[0]
    assert view.layout == StorageLayout(offset=0, strides=(1, 12, 4))
    assert loops.storage_root(view.output) == loops.storage_root(view.source)
    assert view.output not in {alloc.buffer for alloc in loops.allocations}
    assert loops.value_layouts[view.output] == view.layout


def test_transpose_composes_with_existing_positive_stride_slice_layout():
    builder = GraphBuilder()
    value = builder.input((3, 6), dtype="int32")
    sliced = value.slice(axis=1, start=1, stop=6, step=2)
    transposed = sliced.transpose((1, 0))
    loops = lower_to_loops(lower_to_cpu(builder.finish(transposed.relu())))

    assert len(loops.views) == 2
    assert loops.views[0].layout == StorageLayout(offset=1, strides=(6, 2))
    assert loops.views[1].layout == StorageLayout(offset=1, strides=(2, 6))
    assert loops.storage_root(loops.views[1].output) == loops.storage_root(loops.views[0].source)


def test_transpose_rejects_invalid_permutations():
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="float32")

    with pytest.raises(TypeInferenceError, match="permutation|axes"):
        value.transpose((0, 1))
    with pytest.raises(TypeInferenceError, match="permutation|axes"):
        value.transpose((0, 0, 2))
    with pytest.raises(TypeInferenceError, match="permutation|axes"):
        value.transpose((0, 1, 3))
    with pytest.raises(TypeInferenceError, match="integer|axes"):
        value.transpose((0, 1, True))


def test_transpose_composes_with_borrowed_input_cpu_and_native_multi_output():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="float32")
    transposed = value.transpose((2, 0, 1))
    module = builder.finish((transposed, transposed.relu()))
    loops = lower_to_loops(lower_to_cpu(module))
    borrowed = borrow_inputs(loops)

    runtime = np.linspace(-12.0, 11.0, 24, dtype=np.float32).reshape(2, 3, 4)
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


def test_generated_c_uses_permuted_layout_and_general_indexing():
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="int32")
    transposed = value.transpose((2, 0, 1))
    loops = lower_to_loops(lower_to_cpu(builder.finish(transposed.relu())))
    view = loops.views[0]

    source = generate_c(loops)
    root = loops.storage_root(view.output)
    assert f"const int32_t *p{view.output} = p{root};" in source
    assert f"int32_t p{view.output}[" not in source
    assert "* 12" in source
    assert "* 4" in source
    assert "_mm_loadu_si128" not in source


def test_dynamic_transpose_permutes_symbolic_axes_and_reuses_native_cache():
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    value = builder.input((batch, 3), dtype="int32")
    transposed = value.transpose((1, 0))
    module = builder.finish(transposed.relu())
    executable = compile_dynamic_module(module, borrow_inputs=True)

    assert transposed.type.shape == (3, batch)
    for batch_size in (2, 5, 2):
        runtime = np.arange(batch_size * 3, dtype=np.int32).reshape(batch_size, 3) - 4
        actual = executable(inputs=[runtime])
        expected = execute_reference(module, inputs=[runtime])
        np.testing.assert_array_equal(actual, expected)

    assert executable.cached_batch_sizes == (2, 5)


def test_unused_transpose_is_pure_for_dce():
    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="int32")
    value.transpose((1, 0))
    module = builder.finish(value.relu())

    assert dead_code_eliminate(module) == 1
    assert [op.opcode for op in module.function.ops] == ["input", "relu", "return"]
