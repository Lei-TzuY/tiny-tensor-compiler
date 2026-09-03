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
    ("inner_opcode", "left_opcode", "right_opcode", "root_opcode"),
    [
        (inner, left, right, root)
        for inner in ("add", "mul")
        for left in ("add", "mul")
        for right in ("add", "mul")
        for root in ("add", "mul")
    ],
)
def test_integer_chain_tree_fusion_preserves_broadcast_maps_and_semantics(
    inner_opcode,
    left_opcode,
    right_opcode,
    root_opcode,
):
    builder = GraphBuilder()
    a = builder.input((2, 1), dtype="int32")
    b = builder.input((1, 3), dtype="int32")
    c = builder.input((1, 3), dtype="int32")
    d = builder.input((2, 1), dtype="int32")
    e = builder.input((1, 3), dtype="int32")
    left_inner = _binary(a, b, inner_opcode)
    left = _binary(left_inner, c, left_opcode)
    right = _binary(d, e, right_opcode)
    module = builder.finish(_binary(left, right, root_opcode))

    original = lower_to_loops(lower_to_cpu(module))
    fused = fuse_elementwise(original)
    expected_opcode = (
        f"chain_tree_{inner_opcode}_{left_opcode}_{right_opcode}_{root_opcode}"
    )

    assert [kernel.opcode for kernel in original.kernels] == [
        inner_opcode,
        left_opcode,
        right_opcode,
        root_opcode,
    ]
    assert [kernel.opcode for kernel in fused.kernels] == [expected_opcode]
    kernel = fused.kernels[0]
    assert kernel.iteration_shape == (2, 3)
    assert tuple(index_map.axes for index_map in kernel.input_maps) == (
        (0, None),
        (None, 1),
        (None, 1),
        (0, None),
        (None, 1),
    )

    inputs = [
        np.array([[2], [-3]], dtype=np.int32),
        np.array([[5, -7, 11]], dtype=np.int32),
        np.array([[13, 17, -19]], dtype=np.int32),
        np.array([[-23], [29]], dtype=np.int32),
        np.array([[31, -37, 41]], dtype=np.int32),
    ]
    expected = execute_reference(module, inputs=inputs)
    np.testing.assert_array_equal(execute_loop(fused, inputs=inputs), expected)
    assert fuse_elementwise(fused).dump() == fused.dump()


def test_chain_tree_loop_ir_rejects_float_dtype():
    float_type = TensorType((1,), DType.FLOAT32)
    identity = IndexMap((0,))

    with pytest.raises(ValueError, match="integer chain-tree"):
        LoopProgram(
            (
                *(LoopAlloc(buffer, float_type) for buffer in range(6)),
                *(LoopInput(buffer, buffer) for buffer in range(5)),
                LoopKernel(
                    opcode="chain_tree_add_mul_add_mul",
                    output=5,
                    inputs=(0, 1, 2, 3, 4),
                    iteration_shape=(1,),
                    input_maps=(identity,) * 5,
                ),
                LoopReturn(5),
            )
        )


def test_integer_chain_tree_fusion_refuses_final_output_alias_with_leaf_input():
    type_ = TensorType((2,), DType.INT32)
    identity = IndexMap((0,))
    program = LoopProgram(
        (
            *(LoopAlloc(buffer, type_) for buffer in range(8)),
            *(LoopInput(buffer, buffer) for buffer in range(5)),
            LoopKernel("add", 5, (0, 1), (2,), (identity, identity)),
            LoopKernel("mul", 6, (5, 2), (2,), (identity, identity)),
            LoopKernel("add", 7, (3, 4), (2,), (identity, identity)),
            LoopKernel("mul", 0, (6, 7), (2,), (identity, identity)),
            LoopReturn(0),
        )
    )

    fused = fuse_elementwise(program)

    assert all(not kernel.opcode.startswith("chain_tree_") for kernel in fused.kernels)


@pytest.mark.parametrize("reused", ["inner", "left", "right"])
def test_integer_chain_tree_fusion_refuses_intermediate_with_later_use(reused):
    builder = GraphBuilder()
    a = builder.input((2, 1), dtype="int64")
    b = builder.input((1, 3), dtype="int64")
    c = builder.input((1, 3), dtype="int64")
    d = builder.input((2, 1), dtype="int64")
    e = builder.input((1, 3), dtype="int64")
    left_inner = a + b
    left = left_inner * c
    right = d + e
    root = left * right
    extra = {"inner": left_inner, "left": left, "right": right}[reused]
    module = builder.finish(root + extra)

    fused = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))

    assert all(not kernel.opcode.startswith("chain_tree_") for kernel in fused.kernels)


def test_integer_chain_tree_fusion_keeps_mirrored_shape_out_of_scope():
    builder = GraphBuilder()
    a = builder.input((2, 1), dtype="int32")
    b = builder.input((1, 3), dtype="int32")
    c = builder.input((2, 1), dtype="int32")
    d = builder.input((1, 3), dtype="int32")
    e = builder.input((1, 3), dtype="int32")
    left = a + b
    right_inner = c * d
    right = right_inner + e
    module = builder.finish(left * right)

    fused = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))

    assert all(not kernel.opcode.startswith("chain_tree_") for kernel in fused.kernels)


def test_integer_chain_tree_fusion_keeps_reversed_root_out_of_scope():
    builder = GraphBuilder()
    a = builder.input((2, 1), dtype="int32")
    b = builder.input((1, 3), dtype="int32")
    c = builder.input((1, 3), dtype="int32")
    d = builder.input((2, 1), dtype="int32")
    e = builder.input((1, 3), dtype="int32")
    left_inner = a + b
    left = left_inner * c
    right = d + e
    module = builder.finish(right * left)

    fused = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))

    assert all(not kernel.opcode.startswith("chain_tree_") for kernel in fused.kernels)


def test_integer_chain_tree_fusion_does_not_absorb_trailing_relu():
    builder = GraphBuilder()
    a = builder.input((2, 1), dtype="int32")
    b = builder.input((1, 3), dtype="int32")
    c = builder.input((1, 3), dtype="int32")
    d = builder.input((2, 1), dtype="int32")
    e = builder.input((1, 3), dtype="int32")
    left_inner = a + b
    left = left_inner * c
    right = d + e
    module = builder.finish((left * right).relu())

    fused = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))

    assert [kernel.opcode for kernel in fused.kernels] == [
        "chain_tree_add_mul_add_mul",
        "relu",
    ]


def test_integer_chain_tree_native_execution_preserves_all_intermediate_overflows():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    a = builder.input((2, 1), dtype="int32")
    b = builder.input((1, 3), dtype="int32")
    c = builder.input((1, 3), dtype="int32")
    d = builder.input((2, 1), dtype="int32")
    e = builder.input((1, 3), dtype="int32")
    left_inner = a + b
    left = left_inner * c
    right = d * e
    module = builder.finish(left + right)
    fused = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))

    assert [kernel.opcode for kernel in fused.kernels] == ["chain_tree_add_mul_mul_add"]

    inputs = [
        np.array([[2_000_000_000], [-2_000_000_000]], dtype=np.int32),
        np.array([[1_000_000_000, -1_000_000_000, 123_456_789]], dtype=np.int32),
        np.array([[70_000, -70_000, 65_537]], dtype=np.int32),
        np.array([[90_000], [-90_000]], dtype=np.int32),
        np.array([[90_000, -90_000, 131_071]], dtype=np.int32),
    ]
    expected = execute_reference(module, inputs=inputs)
    np.testing.assert_array_equal(execute_loop(fused, inputs=inputs), expected)
    np.testing.assert_array_equal(execute_native(fused, inputs=inputs), expected)
