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
    alias_contiguous_reshapes,
    compile_dynamic_module,
    compile_module,
    execute_loop,
    execute_reference,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
)
from tiny_tensor_compiler.ir import DType, TensorType


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _type(shape, dtype=DType.INT32):
    return TensorType(tuple(shape), dtype)


def test_loop_view_requires_same_dtype_and_element_count():
    source = _type((2, 3))

    with pytest.raises(ValueError, match="dtype"):
        LoopProgram(
            (
                LoopAlloc(0, source),
                LoopInput(0, 0),
                LoopView(1, 0, _type((3, 2), DType.INT64)),
                LoopReturn(1),
            )
        )

    with pytest.raises(ValueError, match="element count"):
        LoopProgram(
            (
                LoopAlloc(0, source),
                LoopInput(0, 0),
                LoopView(1, 0, _type((4, 2))),
                LoopReturn(1),
            )
        )


def test_loop_view_rejects_storage_overwrite_while_alias_is_live():
    source = _type((2, 3))
    viewed = _type((3, 2))

    with pytest.raises(ValueError, match="alias view.*live|storage.*view"):
        LoopProgram(
            (
                LoopAlloc(0, source),
                LoopAlloc(1, viewed),
                LoopAlloc(2, source),
                LoopAlloc(3, viewed),
                LoopInput(0, 0),
                LoopInput(2, 1),
                LoopView(4, 0, viewed),
                LoopKernel("relu", 0, (2,), source.shape, (IndexMap((0, 1)),)),
                LoopKernel("relu", 3, (4,), viewed.shape, (IndexMap((0, 1)),)),
                LoopReturn(3),
            )
        )


def test_loop_view_allows_storage_overwrite_after_last_alias_use():
    source = _type((2, 3))
    viewed = _type((3, 2))

    program = LoopProgram(
        (
            LoopAlloc(0, source),
            LoopAlloc(1, viewed),
            LoopInput(0, 0),
            LoopView(2, 0, viewed),
            LoopKernel("relu", 1, (2,), viewed.shape, (IndexMap((0, 1)),)),
            LoopKernel("reshape", 0, (1,), source.shape, ()),
            LoopReturn(0),
        )
    )

    assert program.return_slot == 0
    assert program.storage_root(2) == 0


def test_alias_transform_rewrites_only_the_reshape_value_epoch():
    source = _type((2, 3))
    viewed = _type((3, 2))
    program = LoopProgram(
        (
            LoopAlloc(0, source),
            LoopAlloc(1, viewed),
            LoopAlloc(2, viewed),
            LoopAlloc(3, viewed),
            LoopInput(0, 0),
            LoopInput(2, 1),
            LoopKernel("reshape", 1, (0,), viewed.shape, ()),
            LoopKernel("relu", 3, (1,), viewed.shape, (IndexMap((0, 1)),)),
            LoopKernel("relu", 1, (2,), viewed.shape, (IndexMap((0, 1)),)),
            LoopReturn(1),
        )
    )

    transformed = alias_contiguous_reshapes(program)

    assert len(transformed.views) == 1
    view = transformed.views[0]
    assert view.source == 0
    first_relu, second_relu = transformed.kernels
    assert first_relu.inputs == (view.output,)
    assert second_relu.output == 1
    assert transformed.return_slot == 1
    assert all(kernel.opcode != "reshape" for kernel in transformed.kernels)


def test_alias_transform_keeps_copy_when_source_storage_is_overwritten():
    source = _type((2, 3))
    viewed = _type((3, 2))
    program = LoopProgram(
        (
            LoopAlloc(0, source),
            LoopAlloc(1, viewed),
            LoopAlloc(2, source),
            LoopAlloc(3, viewed),
            LoopInput(0, 0),
            LoopInput(2, 1),
            LoopKernel("reshape", 1, (0,), viewed.shape, ()),
            LoopKernel("relu", 0, (2,), source.shape, (IndexMap((0, 1)),)),
            LoopKernel("relu", 3, (1,), viewed.shape, (IndexMap((0, 1)),)),
            LoopReturn(3),
        )
    )

    transformed = alias_contiguous_reshapes(program)

    assert transformed.views == ()
    assert transformed.kernels[0].opcode == "reshape"


def test_alias_view_cpu_and_generated_c_eliminate_reshape_copy(monkeypatch):
    builder = GraphBuilder()
    value = builder.tensor(np.arange(6, dtype=np.int32).reshape(2, 3))
    module = builder.finish(value.reshape((3, 2)))
    loops = alias_contiguous_reshapes(lower_to_loops(lower_to_cpu(module)))

    assert len(loops.views) == 1
    source = generate_c(loops)
    view = loops.views[0]
    assert f"const int32_t *p{view.output} = p{view.source};" in source
    assert "[n] = p0[n];" not in source

    def _unexpected_copyto(*args, **kwargs):
        raise AssertionError("alias-view execution must not materialize reshape with np.copyto")

    monkeypatch.setattr(np, "copyto", _unexpected_copyto)
    actual = execute_loop(loops)
    np.testing.assert_array_equal(actual, np.arange(6, dtype=np.int32).reshape(3, 2))


def test_high_level_native_alias_view_composes_with_borrowing_parallel_and_multi_output():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    value = builder.input((3, 4), dtype="float32")
    reshaped = value.reshape((2, 6))
    module = builder.finish((reshaped, reshaped.relu()))
    runtime = np.linspace(-5.0, 6.0, 12, dtype=np.float32).reshape(3, 4)

    executable = compile_module(module, borrow_inputs=True, parallel=True)
    assert "const float *p" in executable._source
    assert "reshape" not in executable._source
    actual = executable(inputs=[runtime])
    expected = execute_reference(module, inputs=[runtime])

    assert isinstance(actual, tuple)
    assert isinstance(expected, tuple)
    for result, wanted in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, wanted)


def test_dynamic_native_alias_view_specializes_and_reuses_cache():
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    value = builder.input((batch, 4), dtype="int32")
    module = builder.finish(value.reshape((2, 2 * batch)))
    executable = compile_dynamic_module(module, borrow_inputs=True)

    for size in (2, 5, 2):
        runtime = np.arange(size * 4, dtype=np.int32).reshape(size, 4)
        actual = executable(inputs=[runtime])
        np.testing.assert_array_equal(actual, runtime.reshape(2, 2 * size))

    assert executable.cached_batch_sizes == (2, 5)
