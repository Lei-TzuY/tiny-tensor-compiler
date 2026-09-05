import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    SymbolicDim,
    SymbolicShapeError,
    TypeInferenceError,
    compile_dynamic_module,
    compile_module,
    execute_reference,
    has_symbolic_shapes,
    lower_to_cpu,
    specialize_module,
    verify,
)


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _symbolic_relu_module():
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    values = builder.input((batch, 3), dtype="float32")
    bias = builder.input((3,), dtype="float32")
    module = builder.finish((values + bias).relu())
    return batch, module


def test_symbolic_batch_is_typed_verified_and_dumped_in_tensor_ir():
    batch, module = _symbolic_relu_module()

    verify(module)

    input_ops = [op for op in module.function.ops if op.opcode == "input"]
    result = module.function.ops[-1].operands[0]
    assert input_ops[0].results[0].type.shape == (batch, 3)
    assert input_ops[1].results[0].type.shape == (3,)
    assert result.type.shape == (batch, 3)
    assert "tensor<Bx3xf32>" in module.dump()
    assert has_symbolic_shapes(module)


def test_symbolic_broadcast_rejects_concrete_or_different_nonunit_dimensions():
    batch = SymbolicDim("B")
    other = SymbolicDim("C")

    builder = GraphBuilder()
    lhs = builder.input((batch, 3), dtype="float32")
    rhs = builder.input((2, 3), dtype="float32")
    with pytest.raises(TypeInferenceError, match="symbolic dimensions"):
        _ = lhs + rhs

    builder = GraphBuilder()
    lhs = builder.input((batch, 3), dtype="float32")
    rhs = builder.input((other, 3), dtype="float32")
    with pytest.raises(TypeInferenceError, match="symbolic dimensions"):
        _ = lhs + rhs


def test_specialization_clones_and_reverifies_a_concrete_module():
    batch, module = _symbolic_relu_module()
    original_dump = module.dump()

    concrete = specialize_module(module, {batch: 5})

    verify(concrete)
    assert not has_symbolic_shapes(concrete)
    assert "tensor<5x3xf32>" in concrete.dump()
    assert module.dump() == original_dump
    assert has_symbolic_shapes(module)
    lower_to_cpu(concrete)

    with pytest.raises(SymbolicShapeError, match="missing bindings"):
        specialize_module(module, {})
    with pytest.raises(SymbolicShapeError, match="non-negative integer"):
        specialize_module(module, {batch: -1})


def test_reference_execution_specializes_the_same_module_for_multiple_batch_sizes():
    _, module = _symbolic_relu_module()
    bias = np.array([0.5, -1.0, 3.0], dtype=np.float32)

    for batch_size in (0, 2, 7):
        values = np.arange(batch_size * 3, dtype=np.float32).reshape(batch_size, 3) - 4
        actual = execute_reference(module, inputs=[values, bias])
        expected = np.maximum(values + bias, np.float32(0))
        np.testing.assert_array_equal(actual, expected)


def test_runtime_binding_rejects_inconsistent_shared_batch_and_static_tail():
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    lhs = builder.input((batch, 3), dtype="float32")
    rhs = builder.input((batch, 3), dtype="float32")
    module = builder.finish(lhs + rhs)

    with pytest.raises(ValueError, match="existing binding is 2"):
        execute_reference(
            module,
            inputs=[
                np.zeros((2, 3), dtype=np.float32),
                np.zeros((7, 3), dtype=np.float32),
            ],
        )

    with pytest.raises(ValueError, match="axis 1 requires 3"):
        execute_reference(
            module,
            inputs=[
                np.zeros((2, 4), dtype=np.float32),
                np.zeros((2, 3), dtype=np.float32),
            ],
        )


def test_dynamic_contract_accepts_nonleading_and_multiple_symbols():
    batch = SymbolicDim("B")
    width = SymbolicDim("W")

    builder = GraphBuilder()
    value = builder.input((3, batch), dtype="float32")
    module = builder.finish(value.relu())
    executable = compile_dynamic_module(module)
    assert executable.symbolic_dim == batch

    builder = GraphBuilder()
    lhs = builder.input((batch, 1), dtype="float32")
    rhs = builder.input((1, width), dtype="float32")
    module = builder.finish(lhs + rhs)
    executable = compile_dynamic_module(module)
    assert executable.symbolic_dims == (batch, width)


def test_static_compile_entrypoint_rejects_unspecialized_symbolic_ir():
    _, module = _symbolic_relu_module()

    with pytest.raises(ValueError, match="compile_dynamic_module"):
        compile_module(module)


def test_dynamic_executable_freezes_caller_owned_module_before_specialization():
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    values = builder.input((batch, 2), dtype="float32")
    module = builder.finish(values * 2)
    executable = compile_dynamic_module(module)

    const_op = next(op for op in module.function.ops if op.opcode == "const")
    const_op.attrs["value"] = np.array(9, dtype=np.float32)
    verify(module)

    runtime_values = np.array([[1.0, -3.0], [4.0, 2.0]], dtype=np.float32)
    actual = executable(inputs=[runtime_values])
    np.testing.assert_array_equal(actual, runtime_values * np.float32(2))


def test_dynamic_native_execution_reuses_each_batch_specialization():
    _default_compiler_or_skip()
    _, module = _symbolic_relu_module()
    executable = compile_dynamic_module(module)
    bias = np.array([1.0, -2.0, 0.5], dtype=np.float32)

    values2 = np.arange(6, dtype=np.float32).reshape(2, 3) - 3
    actual2 = executable(inputs=[values2, bias])
    np.testing.assert_array_equal(actual2, np.maximum(values2 + bias, np.float32(0)))
    first_specialization = executable.specialize(2)
    assert executable.specialize(2) is first_specialization

    values7 = np.arange(21, dtype=np.float32).reshape(7, 3) - 8
    actual7 = executable(inputs=[values7, bias])
    np.testing.assert_array_equal(actual7, np.maximum(values7 + bias, np.float32(0)))

    actual2_again = executable(inputs=[values2, bias])
    np.testing.assert_array_equal(actual2_again, actual2)
    assert executable.cached_batch_sizes == (2, 7)


def test_dynamic_native_integrates_multi_output_and_verified_borrowed_inputs():
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    lhs = builder.input((batch, 4), dtype="int32")
    rhs = builder.input((batch, 4), dtype="int32")
    module = builder.finish((lhs + rhs, (lhs * 2).relu()))
    executable = compile_dynamic_module(module, borrow_inputs=True)

    lhs_value = np.arange(12, dtype=np.int32).reshape(3, 4) - 5
    rhs_value = np.arange(12, dtype=np.int32).reshape(3, 4) * 2
    add_result, relu_result = executable(inputs=[lhs_value, rhs_value])

    np.testing.assert_array_equal(add_result, lhs_value + rhs_value)
    np.testing.assert_array_equal(relu_result, np.maximum(lhs_value * 2, 0))
    assert executable.cached_batch_sizes == (3,)
