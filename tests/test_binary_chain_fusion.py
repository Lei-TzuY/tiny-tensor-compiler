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
    ("inner_opcode", "outer_opcode"),
    [("add", "add"), ("add", "mul"), ("mul", "add"), ("mul", "mul")],
)
def test_integer_binary_chain_fusion_preserves_broadcast_maps_and_semantics(
    inner_opcode,
    outer_opcode,
):
    builder = GraphBuilder()
    lhs = builder.input((2, 1), dtype="int32")
    rhs = builder.input((1, 3), dtype="int32")
    tail = builder.input((), dtype="int32")
    inner = _binary(lhs, rhs, inner_opcode)
    module = builder.finish(_binary(inner, tail, outer_opcode))

    original = lower_to_loops(lower_to_cpu(module))
    fused = fuse_elementwise(original)
    chain = next(kernel for kernel in fused.kernels if kernel.opcode == f"chain_{inner_opcode}_{outer_opcode}")

    assert len(fused.kernels) == len(original.kernels) - 1
    assert chain.iteration_shape == (2, 3)
    assert tuple(index_map.axes for index_map in chain.input_maps) == (
        (0, None),
        (None, 1),
        (),
    )

    inputs = [
        np.array([[1], [2]], dtype=np.int32),
        np.array([[10, 20, 30]], dtype=np.int32),
        np.array(3, dtype=np.int32),
    ]
    np.testing.assert_array_equal(
        execute_loop(fused, inputs=inputs),
        execute_reference(module, inputs=inputs),
    )


def test_integer_binary_chain_fusion_accepts_producer_as_rhs_consumer_operand():
    builder = GraphBuilder()
    lhs = builder.input((2, 1), dtype="int64")
    rhs = builder.input((1, 3), dtype="int64")
    head = builder.input((), dtype="int64")
    inner = lhs + rhs
    module = builder.finish(head * inner)

    fused = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))

    assert [kernel.opcode for kernel in fused.kernels] == ["chain_add_mul"]
    assert tuple(index_map.axes for index_map in fused.kernels[0].input_maps) == (
        (0, None),
        (None, 1),
        (),
    )


def test_integer_binary_chain_fusion_refuses_shared_producer_value():
    builder = GraphBuilder()
    lhs = builder.input((2, 1), dtype="int32")
    rhs = builder.input((1, 3), dtype="int32")
    tail = builder.input((), dtype="int32")
    inner = lhs + rhs
    outer = inner * tail
    module = builder.finish(outer + inner)

    fused = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))

    assert fused.kernels[0].opcode == "add"
    assert all(kernel.opcode != "chain_add_mul" for kernel in fused.kernels)


def test_integer_binary_chain_fusion_refuses_final_output_alias_with_inner_input():
    builder = GraphBuilder()
    lhs = builder.input((3,), dtype="int32")
    rhs = builder.input((3,), dtype="int32")
    tail = builder.input((3,), dtype="int32")
    module = builder.finish((lhs + rhs) * tail)

    original = lower_to_loops(lower_to_cpu(module))
    fused = fuse_elementwise(original)

    assert [kernel.opcode for kernel in fused.kernels] == [
        kernel.opcode for kernel in original.kernels
    ]


def test_integer_binary_chain_fusion_does_not_preempt_existing_binary_relu_fusion():
    builder = GraphBuilder()
    lhs = builder.input((2, 1), dtype="int32")
    rhs = builder.input((1, 3), dtype="int32")
    tail = builder.input((), dtype="int32")
    module = builder.finish(((lhs + rhs) * tail).relu())

    fused = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))

    assert [kernel.opcode for kernel in fused.kernels] == ["add", "relu_mul"]


def test_binary_chain_loop_ir_rejects_float_dtype():
    float_type = TensorType((1,), DType.FLOAT32)
    identity = IndexMap((0,))

    with pytest.raises(ValueError, match="integer binary-chain"):
        LoopProgram(
            (
                LoopAlloc(0, float_type),
                LoopAlloc(1, float_type),
                LoopAlloc(2, float_type),
                LoopAlloc(3, float_type),
                LoopInput(0, 0),
                LoopInput(1, 1),
                LoopInput(2, 2),
                LoopKernel(
                    opcode="chain_add_mul",
                    output=3,
                    inputs=(0, 1, 2),
                    iteration_shape=(1,),
                    input_maps=(identity, identity, identity),
                ),
                LoopReturn(3),
            )
        )


def test_integer_binary_chain_native_execution_preserves_intermediate_overflow():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    lhs = builder.input((2, 1), dtype="int32")
    rhs = builder.input((1, 3), dtype="int32")
    tail = builder.input((), dtype="int32")
    module = builder.finish((lhs + rhs) * tail)
    fused = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))

    assert [kernel.opcode for kernel in fused.kernels] == ["chain_add_mul"]

    inputs = [
        np.array([[2_000_000_000], [-2_000_000_000]], dtype=np.int32),
        np.array([[1_000_000_000, -1_000_000_000, 123_456_789]], dtype=np.int32),
        np.array(3, dtype=np.int32),
    ]
    expected = execute_reference(module, inputs=inputs)
    np.testing.assert_array_equal(execute_loop(fused, inputs=inputs), expected)
    np.testing.assert_array_equal(execute_native(fused, inputs=inputs), expected)
