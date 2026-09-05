import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    SymbolicDim,
    SymbolicShapeError,
    bind_dynamic_shapes,
    compile_dynamic_module,
    execute_reference,
)


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _two_symbol_module():
    batch = SymbolicDim("B")
    width = SymbolicDim("W")
    builder = GraphBuilder()
    lhs = builder.input((batch, 1), dtype="int32")
    rhs = builder.input((1, width), dtype="int32")
    module = builder.finish((lhs + rhs, (lhs * 2).relu()))
    return batch, width, module


def test_reference_binds_multiple_symbols_on_nonleading_axes():
    batch = SymbolicDim("B")
    width = SymbolicDim("W")
    builder = GraphBuilder()
    lhs = builder.input((2, batch, 1), dtype="float32")
    rhs = builder.input((1, batch, width), dtype="float32")
    module = builder.finish((lhs + rhs).relu())

    lhs_value = np.arange(6, dtype=np.float32).reshape(2, 3, 1) - 4
    rhs_value = np.arange(12, dtype=np.float32).reshape(1, 3, 4) - 5

    bindings = bind_dynamic_shapes(module, [lhs_value, rhs_value])
    assert bindings == {batch: 3, width: 4}
    np.testing.assert_array_equal(
        execute_reference(module, inputs=[lhs_value, rhs_value]),
        np.maximum(lhs_value + rhs_value, np.float32(0)),
    )


def test_runtime_binding_requires_repeated_symbol_occurrences_to_agree():
    size = SymbolicDim("N")
    builder = GraphBuilder()
    value = builder.input((size, size), dtype="float32")
    module = builder.finish(value.relu())

    with pytest.raises(ValueError, match="existing binding is 2"):
        bind_dynamic_shapes(module, [np.zeros((2, 3), dtype=np.float32)])


def test_dynamic_executable_caches_complete_multi_symbol_bindings():
    _default_compiler_or_skip()
    batch, width, module = _two_symbol_module()
    executable = compile_dynamic_module(module, borrow_inputs=True)

    assert executable.symbolic_dims == (batch, width)
    with pytest.raises(SymbolicShapeError, match="single symbolic dimension"):
        _ = executable.symbolic_dim
    with pytest.raises(SymbolicShapeError, match="single symbolic dimension"):
        _ = executable.cached_batch_sizes

    lhs23 = np.arange(2, dtype=np.int32).reshape(2, 1) - 1
    rhs23 = np.arange(3, dtype=np.int32).reshape(1, 3) * 2
    add23, relu23 = executable(inputs=[lhs23, rhs23])
    np.testing.assert_array_equal(add23, lhs23 + rhs23)
    np.testing.assert_array_equal(relu23, np.maximum(lhs23 * 2, 0))

    first = executable.specialize({"B": 2, width: 3})
    assert executable.specialize({batch: 2, "W": 3}) is first

    lhs25 = np.arange(2, dtype=np.int32).reshape(2, 1) + 3
    rhs25 = np.arange(5, dtype=np.int32).reshape(1, 5) - 2
    add25, relu25 = executable(inputs=[lhs25, rhs25])
    np.testing.assert_array_equal(add25, lhs25 + rhs25)
    np.testing.assert_array_equal(relu25, np.maximum(lhs25 * 2, 0))

    assert executable.cached_bindings == (
        (("B", 2), ("W", 3)),
        (("B", 2), ("W", 5)),
    )


def test_multi_symbol_native_execution_supports_zero_extent_binding():
    _default_compiler_or_skip()
    _, _, module = _two_symbol_module()
    executable = compile_dynamic_module(module)

    lhs = np.empty((0, 1), dtype=np.int32)
    rhs = np.arange(4, dtype=np.int32).reshape(1, 4)
    add_result, relu_result = executable(inputs=[lhs, rhs])

    assert add_result.shape == (0, 4)
    assert relu_result.shape == (0, 1)
    np.testing.assert_array_equal(add_result, lhs + rhs)
    np.testing.assert_array_equal(relu_result, np.maximum(lhs * 2, 0))
    assert executable.cached_bindings == ((("B", 0), ("W", 4)),)


def test_single_symbol_specialize_rejects_bool_as_invalid_type():
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    value = builder.input((batch, 2), dtype="float32")
    executable = compile_dynamic_module(builder.finish(value.relu()))

    with pytest.raises(TypeError, match="bool"):
        executable.specialize(True)


def test_dynamic_module_requires_at_least_one_runtime_symbol():
    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="float32")
    module = builder.finish(value.relu())

    with pytest.raises(SymbolicShapeError, match="at least one symbolic dimension"):
        compile_dynamic_module(module)
