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


def test_permute_is_typed_zero_copy_view_with_reference_semantics():
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="float32")
    permuted = value.permute((2, 0, 1))
    module = builder.finish(permuted)

    assert permuted.type.shape == (4, 2, 3)
    assert permuted.type.dtype == value.type.dtype
    assert "%1 = permute %0" in module.dump()

    runtime = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    actual = execute_reference(module, inputs=[runtime])
    np.testing.assert_array_equal(actual, np.transpose(runtime, (2, 0, 1)))


def test_permute_layout_reorders_strides_without_allocating_storage():
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="int32")
    permuted = value.permute((2, 0, 1))
    loops = lower_to_loops(lower_to_cpu(builder.finish(permuted.relu())))

    assert len(loops.views) == 1
    view = loops.views[0]
    assert view.layout == StorageLayout(offset=0, strides=(1, 12, 4))
    assert loops.storage_root(view.output) == loops.storage_root(view.source)
    assert view.output not in {alloc.buffer for alloc in loops.allocations}
    assert loops.value_layouts[view.output] == view.layout


def test_permute_rejects_non_permutations():
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="float32")

    with pytest.raises(TypeInferenceError, match="permutation|axes"):
        value.permute((0, 0, 2))
    with pytest.raises(TypeInferenceError, match="permutation|axes"):
        value.permute((0, 1))
    with pytest.raises(TypeInferenceError, match="permutation|axes"):
        value.permute((0, 1, 3))


def test_permute_composes_with_positive_stride_slice_and_borrowed_native_execution():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    value = builder.input((3, 6), dtype="float32")
    sliced = value.slice(axis=1, start=1, stop=6, step=2)
    permuted = sliced.permute((1, 0))
    module = builder.finish((permuted, permuted.relu()))
    loops = lower_to_loops(lower_to_cpu(module))
    borrowed = borrow_inputs(loops)

    assert [view.layout for view in loops.views] == [
        StorageLayout(offset=1, strides=(6, 2)),
        StorageLayout(offset=1, strides=(2, 6)),
    ]

    runtime = np.linspace(-9.0, 8.0, 18, dtype=np.float32).reshape(3, 6)
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


def test_generated_c_indexes_permuted_strides_and_does_not_use_sse2_fast_path():
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="int32")
    permuted = value.permute((2, 0, 1))
    loops = lower_to_loops(lower_to_cpu(builder.finish(permuted.relu())))
    view = loops.views[0]

    source = generate_c(loops)
    root = loops.storage_root(view.output)
    assert f"const int32_t *p{view.output} = p{root};" in source
    assert f"int32_t p{view.output}[" not in source
    assert "(i0 * 1)" not in source
    assert "(i1 * 12)" in source
    assert "(i2 * 4)" in source
    assert "_mm_loadu_si128((const __m128i *)(p" not in source


def test_dynamic_permute_specializes_symbolic_axes_and_reuses_native_cache():
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    value = builder.input((batch, 3, 4), dtype="int32")
    permuted = value.permute((1, 0, 2))
    module = builder.finish(permuted.relu())
    executable = compile_dynamic_module(module, borrow_inputs=True)

    for batch_size in (2, 5, 2):
        runtime = np.arange(batch_size * 12, dtype=np.int32).reshape(batch_size, 3, 4) - 8
        actual = executable(inputs=[runtime])
        expected = execute_reference(module, inputs=[runtime])
        np.testing.assert_array_equal(actual, expected)

    assert executable.cached_batch_sizes == (2, 5)


def test_unused_permute_is_pure_for_dce():
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="int32")
    value.permute((2, 0, 1))
    module = builder.finish(value.relu())

    assert dead_code_eliminate(module) == 1
    assert [op.opcode for op in module.function.ops] == ["input", "relu", "return"]


def test_storage_layout_permutation_preserves_offset_and_reorders_strides():
    layout = StorageLayout(offset=5, strides=(12, 4, 1))
    permuted, shape = layout.permuted((2, 3, 4), (2, 0, 1))

    assert shape == (4, 2, 3)
    assert permuted == StorageLayout(offset=5, strides=(1, 12, 4))
