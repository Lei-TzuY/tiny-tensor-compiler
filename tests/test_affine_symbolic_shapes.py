import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    AffineDim,
    GraphBuilder,
    SymbolicDim,
    TypeInferenceError,
    bind_dynamic_shapes,
    compile_dynamic_module,
    execute_reference,
    specialize_module,
)


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def test_affine_dimension_is_canonical_and_specializes_to_concrete_shape():
    batch = SymbolicDim("B")
    dim = 2 * batch + 1
    assert dim == AffineDim(batch, scale=2, offset=1)
    assert str(dim) == "2*B+1"

    builder = GraphBuilder()
    value = builder.input((dim, 3), dtype="float32")
    module = builder.finish(value.relu())

    concrete = specialize_module(module, {batch: 3})
    input_op = next(op for op in concrete.function.ops if op.opcode == "input")
    assert input_op.results[0].type.shape == (7, 3)


def test_reference_infers_affine_binding_from_runtime_extent():
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    value = builder.input((2 * batch + 1, 2), dtype="float32")
    module = builder.finish(value.relu())
    runtime = np.arange(14, dtype=np.float32).reshape(7, 2) - 8

    assert bind_dynamic_shapes(module, [runtime]) == {batch: 3}
    np.testing.assert_array_equal(
        execute_reference(module, inputs=[runtime]),
        np.maximum(runtime, np.float32(0)),
    )


def test_affine_binding_rejects_non_integral_and_below_offset_extents():
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    value = builder.input((2 * batch + 3,), dtype="int32")
    module = builder.finish(value)

    with pytest.raises(ValueError, match="not divisible by scale 2"):
        bind_dynamic_shapes(module, [np.zeros((8,), dtype=np.int32)])
    with pytest.raises(ValueError, match="smaller than affine offset 3"):
        bind_dynamic_shapes(module, [np.zeros((2,), dtype=np.int32)])


def test_direct_and_affine_occurrences_must_bind_same_symbol():
    size = SymbolicDim("N")
    builder = GraphBuilder()
    direct = builder.input((size,), dtype="float32")
    affine = builder.input((2 * size + 1,), dtype="float32")
    module = builder.finish((direct.relu(), affine.relu()))

    with pytest.raises(ValueError, match="existing binding is 2"):
        bind_dynamic_shapes(
            module,
            [
                np.zeros((2,), dtype=np.float32),
                np.zeros((7,), dtype=np.float32),
            ],
        )


def test_affine_broadcast_requires_structural_dimension_equality():
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    lhs = builder.input((2 * batch + 1, 1), dtype="float32")
    rhs = builder.input((2 * batch + 1, 4), dtype="float32")
    result = lhs + rhs
    assert result.type.shape == (2 * batch + 1, 4)

    other_builder = GraphBuilder()
    direct = other_builder.input((batch, 1), dtype="float32")
    affine = other_builder.input((2 * batch, 4), dtype="float32")
    with pytest.raises(TypeInferenceError, match="cannot broadcast symbolic dimensions"):
        _ = direct + affine


def test_affine_dynamic_native_execution_and_cache_use_solved_bindings():
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    width = SymbolicDim("W")
    builder = GraphBuilder()
    lhs = builder.input((2 * batch + 1, 1), dtype="int32")
    rhs = builder.input((1, 3 * width + 2), dtype="int32")
    module = builder.finish((lhs + rhs, lhs.relu()))
    executable = compile_dynamic_module(module, borrow_inputs=True)

    lhs5 = np.arange(5, dtype=np.int32).reshape(5, 1) - 2
    rhs8 = np.arange(8, dtype=np.int32).reshape(1, 8) - 3
    add_result, relu_result = executable(inputs=[lhs5, rhs8])
    np.testing.assert_array_equal(add_result, lhs5 + rhs8)
    np.testing.assert_array_equal(relu_result, np.maximum(lhs5, 0))

    first = executable.specialize({batch: 2, width: 2})
    assert executable.specialize({"W": 2, "B": 2}) is first
    assert executable.cached_bindings == ((("B", 2), ("W", 2)),)


def test_affine_zero_extent_specializes_and_executes_natively():
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    value = builder.input((2 * batch,), dtype="int32")
    module = builder.finish((value + 1).relu())
    executable = compile_dynamic_module(module)

    runtime = np.empty((0,), dtype=np.int32)
    result = executable(inputs=[runtime])
    assert result.shape == (0,)
    assert executable.cached_bindings == ((("B", 0),),)


def test_affine_dimension_rejects_non_positive_scale_and_negative_offset():
    batch = SymbolicDim("B")
    with pytest.raises(ValueError, match="positive integer scale"):
        AffineDim(batch, scale=0)
    with pytest.raises(ValueError, match="non-negative integer offset"):
        AffineDim(batch, offset=-1)
