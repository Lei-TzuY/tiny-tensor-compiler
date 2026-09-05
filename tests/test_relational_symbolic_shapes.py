import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    LinearDim,
    SymbolicDim,
    SymbolicShapeError,
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


def test_linear_dimension_arithmetic_is_canonical_and_specializes():
    batch = SymbolicDim("B")
    width = SymbolicDim("W")

    dim = width + batch + batch + 3
    assert dim == LinearDim(((batch, 2), (width, 1)), offset=3)
    assert str(dim) == "2*B+W+3"
    assert batch + batch + 1 == 2 * batch + 1

    builder = GraphBuilder()
    value = builder.input((dim,), dtype="float32")
    module = builder.finish(value.relu())
    concrete = specialize_module(module, {batch: 2, width: 4})
    input_op = next(op for op in concrete.function.ops if op.opcode == "input")
    assert input_op.results[0].type.shape == (11,)


def test_runtime_solves_unique_two_symbol_system_from_input_axes():
    batch = SymbolicDim("B")
    width = SymbolicDim("W")
    builder = GraphBuilder()
    value = builder.input((batch + width, 2 * batch + width), dtype="float32")
    module = builder.finish(value.relu())
    runtime = np.arange(35, dtype=np.float32).reshape(5, 7) - 18

    assert bind_dynamic_shapes(module, [runtime]) == {batch: 2, width: 3}
    np.testing.assert_array_equal(
        execute_reference(module, inputs=[runtime]),
        np.maximum(runtime, np.float32(0)),
    )


def test_relational_solver_rejects_underdetermined_system():
    batch = SymbolicDim("B")
    width = SymbolicDim("W")
    builder = GraphBuilder()
    value = builder.input((batch + width,), dtype="int32")
    module = builder.finish(value)

    with pytest.raises(SymbolicShapeError, match="do not uniquely determine"):
        bind_dynamic_shapes(module, [np.zeros((5,), dtype=np.int32)])


def test_relational_solver_rejects_inconsistent_system():
    batch = SymbolicDim("B")
    width = SymbolicDim("W")
    builder = GraphBuilder()
    value = builder.input((batch + width, batch + width), dtype="int32")
    module = builder.finish(value)

    with pytest.raises(ValueError, match="inconsistent"):
        bind_dynamic_shapes(module, [np.zeros((5, 6), dtype=np.int32)])


def test_relational_solver_rejects_fractional_and_negative_unique_solutions():
    batch = SymbolicDim("B")
    width = SymbolicDim("W")

    fractional_builder = GraphBuilder()
    fractional = fractional_builder.input(
        (2 * batch + width, batch + 2 * width), dtype="int32"
    )
    fractional_module = fractional_builder.finish(fractional)
    with pytest.raises(ValueError, match="non-negative integer"):
        bind_dynamic_shapes(
            fractional_module,
            [np.zeros((1, 1), dtype=np.int32)],
        )

    negative_builder = GraphBuilder()
    negative = negative_builder.input(
        (batch + width, 2 * batch + width), dtype="int32"
    )
    negative_module = negative_builder.finish(negative)
    with pytest.raises(ValueError, match="non-negative integer"):
        bind_dynamic_shapes(
            negative_module,
            [np.zeros((1, 3), dtype=np.int32)],
        )


def test_direct_affine_and_relational_constraints_share_one_binding_system():
    batch = SymbolicDim("B")
    width = SymbolicDim("W")
    builder = GraphBuilder()
    value = builder.input(
        (batch, batch + width, 2 * width + 1),
        dtype="float32",
    )
    module = builder.finish(value)
    runtime = np.zeros((2, 5, 7), dtype=np.float32)

    assert bind_dynamic_shapes(module, [runtime]) == {batch: 2, width: 3}

    inconsistent = np.zeros((2, 5, 9), dtype=np.float32)
    with pytest.raises(ValueError, match="existing binding is 4|inconsistent"):
        bind_dynamic_shapes(module, [inconsistent])


def test_relational_broadcasting_requires_structural_expression_equality():
    batch = SymbolicDim("B")
    width = SymbolicDim("W")
    builder = GraphBuilder()
    lhs = builder.input((batch + width, 1), dtype="float32")
    rhs = builder.input((batch + width, 4), dtype="float32")
    assert (lhs + rhs).type.shape == (batch + width, 4)

    other_builder = GraphBuilder()
    lhs_other = other_builder.input((batch + width, 1), dtype="float32")
    rhs_other = other_builder.input((2 * batch + width, 4), dtype="float32")
    with pytest.raises(TypeInferenceError, match="cannot broadcast symbolic dimensions"):
        _ = lhs_other + rhs_other


def test_relational_native_multi_output_borrowing_and_cache_use_solved_bindings():
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    width = SymbolicDim("W")
    builder = GraphBuilder()
    lhs = builder.input((batch + width, 1), dtype="int32")
    rhs = builder.input((1, 2 * batch + width), dtype="int32")
    module = builder.finish((lhs + rhs, lhs.relu()))
    executable = compile_dynamic_module(module, borrow_inputs=True)

    lhs5 = np.arange(5, dtype=np.int32).reshape(5, 1) - 2
    rhs7 = np.arange(7, dtype=np.int32).reshape(1, 7) - 3
    add_result, relu_result = executable(inputs=[lhs5, rhs7])
    np.testing.assert_array_equal(add_result, lhs5 + rhs7)
    np.testing.assert_array_equal(relu_result, np.maximum(lhs5, 0))
    assert executable.cached_bindings == ((("B", 2), ("W", 3)),)

    first = executable.specialize({batch: 2, width: 3})
    assert executable.specialize({"W": 3, "B": 2}) is first

    lhs3 = np.arange(3, dtype=np.int32).reshape(3, 1)
    rhs4 = np.arange(4, dtype=np.int32).reshape(1, 4)
    second_add, _ = executable(inputs=[lhs3, rhs4])
    np.testing.assert_array_equal(second_add, lhs3 + rhs4)
    assert executable.cached_bindings == (
        (("B", 1), ("W", 2)),
        (("B", 2), ("W", 3)),
    )


def test_relational_solver_accepts_unique_zero_binding_system():
    batch = SymbolicDim("B")
    width = SymbolicDim("W")
    builder = GraphBuilder()
    value = builder.input((batch + width, batch + 2 * width), dtype="int32")
    module = builder.finish(value)
    runtime = np.empty((0, 0), dtype=np.int32)

    assert bind_dynamic_shapes(module, [runtime]) == {batch: 0, width: 0}


def test_linear_dimension_rejects_invalid_coefficients_and_offset():
    batch = SymbolicDim("B")
    width = SymbolicDim("W")

    with pytest.raises(ValueError, match="positive integer coefficient"):
        LinearDim(((batch, 0), (width, 1)))
    with pytest.raises(ValueError, match="non-negative integer offset"):
        LinearDim(((batch, 1), (width, 1)), offset=-1)
    with pytest.raises(ValueError, match="at least two distinct symbols"):
        LinearDim(((batch, 2),))
