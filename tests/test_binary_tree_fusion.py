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
    execute_loop,
    execute_native,
    execute_reference,
    fuse_elementwise,
    lower_to_cpu,
    lower_to_loops,
)
from tiny_tensor_compiler.ir import DType, TensorType


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _binary(lhs, rhs, opcode):
    return lhs + rhs if opcode == "add" else lhs * rhs


@pytest.mark.parametrize(
    ("left_opcode", "right_opcode", "root_opcode"),
    [
        (left, right, root)
        for left in ("add", "mul")
        for right in ("add", "mul")
        for root in ("add", "mul")
    ],
)
def test_integer_binary_tree_fusion_preserves_broadcast_maps_and_semantics(
    left_opcode,
    right_opcode,
    root_opcode,
):
    builder = GraphBuilder()
    left_lhs = builder.input((2, 1), dtype="int32")
    left_rhs = builder.input((1, 3), dtype="int32")
    right_lhs = builder.input((2, 1), dtype="int32")
    right_rhs = builder.input((1, 3), dtype="int32")
    left = _binary(left_lhs, left_rhs, left_opcode)
    right = _binary(right_lhs, right_rhs, right_opcode)
    module = builder.finish(_binary(left, right, root_opcode))

    original = lower_to_loops(lower_to_cpu(module))
    fused = fuse_elementwise(original)
    expected_opcode = f"tree_{left_opcode}_{right_opcode}_{root_opcode}"

    assert [kernel.opcode for kernel in original.kernels] == [
        left_opcode,
        right_opcode,
        root_opcode,
    ]
    assert [kernel.opcode for kernel in fused.kernels] == [expected_opcode]
    tree = fused.kernels[0]
    assert tree.iteration_shape == (2, 3)
    assert tuple(index_map.axes for index_map in tree.input_maps) == (
        (0, None),
        (None, 1),
        (0, None),
        (None, 1),
    )

    inputs = [
        np.array([[1], [2]], dtype=np.int32),
        np.array([[10, 20, 30]], dtype=np.int32),
        np.array([[3], [4]], dtype=np.int32),
        np.array([[5, 6, 7]], dtype=np.int32),
    ]
    expected = execute_reference(module, inputs=inputs)
    np.testing.assert_array_equal(execute_loop(fused, inputs=inputs), expected)
    assert fuse_elementwise(fused).dump() == fused.dump()


@pytest.mark.parametrize(
    ("left_opcode", "right_opcode", "root_opcode"),
    [
        (left, right, root)
        for left in ("add", "mul")
        for right in ("add", "mul")
        for root in ("add", "mul")
    ],
)
def test_integer_binary_tree_fuses_safe_trailing_relu(
    left_opcode,
    right_opcode,
    root_opcode,
):
    builder = GraphBuilder()
    left_lhs = builder.input((2, 1), dtype="int32")
    left_rhs = builder.input((1, 3), dtype="int32")
    right_lhs = builder.input((2, 1), dtype="int32")
    right_rhs = builder.input((1, 3), dtype="int32")
    left = _binary(left_lhs, left_rhs, left_opcode)
    right = _binary(right_lhs, right_rhs, right_opcode)
    module = builder.finish(_binary(left, right, root_opcode).relu())

    original = lower_to_loops(lower_to_cpu(module))
    fused = fuse_elementwise(original)
    expected_opcode = f"relu_tree_{left_opcode}_{right_opcode}_{root_opcode}"

    assert [kernel.opcode for kernel in original.kernels] == [
        left_opcode,
        right_opcode,
        root_opcode,
        "relu",
    ]
    assert [kernel.opcode for kernel in fused.kernels] == [expected_opcode]
    tree = fused.kernels[0]
    assert tree.iteration_shape == (2, 3)
    assert tuple(index_map.axes for index_map in tree.input_maps) == (
        (0, None),
        (None, 1),
        (0, None),
        (None, 1),
    )

    inputs = [
        np.array([[-9], [2]], dtype=np.int32),
        np.array([[1, 20, -30]], dtype=np.int32),
        np.array([[3], [-4]], dtype=np.int32),
        np.array([[5, -6, 7]], dtype=np.int32),
    ]
    expected = execute_reference(module, inputs=inputs)
    np.testing.assert_array_equal(execute_loop(fused, inputs=inputs), expected)
    assert fuse_elementwise(fused).dump() == fused.dump()


def test_binary_tree_loop_ir_rejects_float_dtype():
    float_type = TensorType((1,), DType.FLOAT32)
    identity = IndexMap((0,))

    with pytest.raises(ValueError, match="integer binary-tree"):
        LoopProgram(
            (
                LoopAlloc(0, float_type),
                LoopAlloc(1, float_type),
                LoopAlloc(2, float_type),
                LoopAlloc(3, float_type),
                LoopAlloc(4, float_type),
                LoopInput(0, 0),
                LoopInput(1, 1),
                LoopInput(2, 2),
                LoopInput(3, 3),
                LoopKernel(
                    opcode="tree_add_mul_add",
                    output=4,
                    inputs=(0, 1, 2, 3),
                    iteration_shape=(1,),
                    input_maps=(identity, identity, identity, identity),
                ),
                LoopReturn(4),
            )
        )


def test_relu_binary_tree_loop_ir_rejects_float_dtype():
    float_type = TensorType((1,), DType.FLOAT32)
    identity = IndexMap((0,))

    with pytest.raises(ValueError, match="integer ReLU binary-tree"):
        LoopProgram(
            (
                LoopAlloc(0, float_type),
                LoopAlloc(1, float_type),
                LoopAlloc(2, float_type),
                LoopAlloc(3, float_type),
                LoopAlloc(4, float_type),
                LoopInput(0, 0),
                LoopInput(1, 1),
                LoopInput(2, 2),
                LoopInput(3, 3),
                LoopKernel(
                    opcode="relu_tree_add_mul_add",
                    output=4,
                    inputs=(0, 1, 2, 3),
                    iteration_shape=(1,),
                    input_maps=(identity, identity, identity, identity),
                ),
                LoopReturn(4),
            )
        )


def test_integer_binary_tree_fusion_refuses_final_output_alias_with_leaf_input():
    builder = GraphBuilder()
    left_lhs = builder.input((3,), dtype="int32")
    left_rhs = builder.input((3,), dtype="int32")
    right_lhs = builder.input((3,), dtype="int32")
    right_rhs = builder.input((3,), dtype="int32")
    left = left_lhs + left_rhs
    right = right_lhs * right_rhs
    module = builder.finish(left + right)

    fused = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))

    assert all(not kernel.opcode.startswith("tree_") for kernel in fused.kernels)


def test_integer_tree_relu_fusion_refuses_final_output_alias_with_leaf_input():
    type_ = TensorType((2,), DType.INT32)
    identity = IndexMap((0,))
    program = LoopProgram(
        (
            *(LoopAlloc(buffer, type_) for buffer in range(7)),
            LoopInput(0, 0),
            LoopInput(1, 1),
            LoopInput(2, 2),
            LoopInput(3, 3),
            LoopKernel("add", 4, (0, 1), (2,), (identity, identity)),
            LoopKernel("mul", 5, (2, 3), (2,), (identity, identity)),
            LoopKernel("add", 6, (4, 5), (2,), (identity, identity)),
            LoopKernel("relu", 0, (6,), (2,), (identity,)),
            LoopReturn(0),
        )
    )

    fused = fuse_elementwise(program)

    assert [kernel.opcode for kernel in fused.kernels] == ["tree_add_mul_add", "relu"]


def test_integer_binary_tree_fusion_refuses_producer_with_later_use():
    builder = GraphBuilder()
    left_lhs = builder.input((2, 1), dtype="int64")
    left_rhs = builder.input((1, 3), dtype="int64")
    right_lhs = builder.input((2, 1), dtype="int64")
    right_rhs = builder.input((1, 3), dtype="int64")
    left = left_lhs + left_rhs
    right = right_lhs * right_rhs
    root = left + right
    module = builder.finish(root * left)

    fused = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))

    assert all(not kernel.opcode.startswith("tree_") for kernel in fused.kernels)


def test_integer_tree_relu_fusion_refuses_root_with_later_use():
    builder = GraphBuilder()
    left_lhs = builder.input((2, 1), dtype="int64")
    left_rhs = builder.input((1, 3), dtype="int64")
    right_lhs = builder.input((2, 1), dtype="int64")
    right_rhs = builder.input((1, 3), dtype="int64")
    left = left_lhs + left_rhs
    right = right_lhs * right_rhs
    root = left + right
    module = builder.finish(root.relu() + root)

    fused = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))

    assert any(kernel.opcode == "tree_add_mul_add" for kernel in fused.kernels)
    assert all(not kernel.opcode.startswith("relu_tree_") for kernel in fused.kernels)


def test_integer_binary_tree_native_execution_preserves_both_intermediate_overflows():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    left_lhs = builder.input((2, 1), dtype="int32")
    left_rhs = builder.input((1, 3), dtype="int32")
    right_lhs = builder.input((2, 1), dtype="int32")
    right_rhs = builder.input((1, 3), dtype="int32")
    left = left_lhs + left_rhs
    right = right_lhs * right_rhs
    module = builder.finish(left * right)
    fused = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))

    assert [kernel.opcode for kernel in fused.kernels] == ["tree_add_mul_mul"]

    inputs = [
        np.array([[2_000_000_000], [-2_000_000_000]], dtype=np.int32),
        np.array([[1_000_000_000, -1_000_000_000, 123_456_789]], dtype=np.int32),
        np.array([[70_000], [-70_000]], dtype=np.int32),
        np.array([[70_000, -70_000, 65_537]], dtype=np.int32),
    ]
    expected = execute_reference(module, inputs=inputs)
    np.testing.assert_array_equal(execute_loop(fused, inputs=inputs), expected)
    np.testing.assert_array_equal(execute_native(fused, inputs=inputs), expected)


def test_integer_tree_relu_native_execution_preserves_overflow_then_relu():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    left_lhs = builder.input((2, 1), dtype="int32")
    left_rhs = builder.input((1, 3), dtype="int32")
    right_lhs = builder.input((2, 1), dtype="int32")
    right_rhs = builder.input((1, 3), dtype="int32")
    left = left_lhs + left_rhs
    right = right_lhs * right_rhs
    module = builder.finish((left * right).relu())
    fused = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))

    assert [kernel.opcode for kernel in fused.kernels] == ["relu_tree_add_mul_mul"]

    inputs = [
        np.array([[2_000_000_000], [-2_000_000_000]], dtype=np.int32),
        np.array([[1_000_000_000, -1_000_000_000, 123_456_789]], dtype=np.int32),
        np.array([[70_000], [-70_000]], dtype=np.int32),
        np.array([[70_000, -70_000, 65_537]], dtype=np.int32),
    ]
    expected = execute_reference(module, inputs=inputs)
    np.testing.assert_array_equal(execute_loop(fused, inputs=inputs), expected)
    np.testing.assert_array_equal(execute_native(fused, inputs=inputs), expected)
