import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    LoopKernel,
    execute_cpu,
    execute_loop,
    execute_reference,
    fuse_elementwise,
    lower_to_cpu,
    lower_to_loops,
)


def test_loop_lowering_makes_broadcast_indexing_explicit_and_deterministic():
    builder = GraphBuilder()
    lhs = builder.tensor([[1], [2]], dtype="int32")
    rhs = builder.tensor([[10, 20, 30]], dtype="int32")
    module = builder.finish(lhs + rhs)

    loop_program = lower_to_loops(lower_to_cpu(module))
    add = next(
        op for op in loop_program.operations if isinstance(op, LoopKernel) and op.opcode == "add"
    )

    assert add.iteration_shape == (2, 3)
    assert add.input_maps[0].axes == (0, None)
    assert add.input_maps[1].axes == (None, 1)
    assert loop_program.dump() == """alloc p0 : tensor<2x1xi32>
alloc p1 : tensor<1x3xi32>
alloc p2 : tensor<2x3xi32>
for [i0<2, i1<1]: p0[i0, i1] = const [[1], [2]][i0, i1]
for [i0<1, i1<3]: p1[i0, i1] = const [[10, 20, 30]][i0, i1]
for [i0<2, i1<3]: p2[i0, i1] = add p0[i0, 0], p1[0, i1]
return p2"""


def test_loop_lowering_represents_scalar_broadcast_with_empty_index_map():
    builder = GraphBuilder()
    vector = builder.tensor([1, 2, 3], dtype="int32")
    scalar = builder.tensor(4, dtype="int32")
    module = builder.finish(vector * scalar)

    loop_program = lower_to_loops(lower_to_cpu(module))
    mul = next(
        op for op in loop_program.operations if isinstance(op, LoopKernel) and op.opcode == "mul"
    )

    assert mul.iteration_shape == (3,)
    assert mul.input_maps[0].axes == (0,)
    assert mul.input_maps[1].axes == ()


def test_relu_loop_uses_identity_indexing_and_cpu_executes_loop_ir():
    builder = GraphBuilder()
    value = builder.tensor([[-2.0, 1.5], [3.0, -4.0]], dtype="float32")
    module = builder.finish(value.relu())

    buffer_program = lower_to_cpu(module)
    loop_program = lower_to_loops(buffer_program)
    relu = next(
        op for op in loop_program.operations if isinstance(op, LoopKernel) and op.opcode == "relu"
    )

    assert relu.iteration_shape == (2, 2)
    assert relu.input_maps[0].axes == (0, 1)
    np.testing.assert_array_equal(execute_cpu(buffer_program), execute_reference(module))


def test_loop_execution_matches_reference_for_broadcasting():
    builder = GraphBuilder()
    lhs = builder.tensor([[-2.0], [3.5]], dtype="float32")
    rhs = builder.tensor([[1.0, -4.0, 2.0]], dtype="float32")
    module = builder.finish((lhs + rhs).relu())

    np.testing.assert_array_equal(execute_cpu(lower_to_cpu(module)), execute_reference(module))


@pytest.mark.parametrize("opcode", ["add", "mul"])
def test_elementwise_fusion_combines_safe_broadcast_binary_relu(opcode):
    builder = GraphBuilder()
    lhs = builder.tensor([[-3.0], [2.0]], dtype="float32")
    rhs = builder.tensor([[1.0, 2.0, 4.0]], dtype="float32")
    binary = lhs + rhs if opcode == "add" else lhs * rhs
    module = builder.finish(binary.relu())

    original = lower_to_loops(lower_to_cpu(module))
    fused = fuse_elementwise(original)
    fused_kernel = next(kernel for kernel in fused.kernels if kernel.opcode == f"relu_{opcode}")

    assert len(fused.kernels) == len(original.kernels) - 1
    assert fused_kernel.iteration_shape == (2, 3)
    assert fused_kernel.input_maps[0].axes == (0, None)
    assert fused_kernel.input_maps[1].axes == (None, 1)
    np.testing.assert_array_equal(execute_loop(fused), execute_reference(module))


def test_elementwise_fusion_refuses_output_input_alias():
    builder = GraphBuilder()
    lhs = builder.tensor([-3.0, 2.0], dtype="float32")
    rhs = builder.tensor([1.0, 4.0], dtype="float32")
    module = builder.finish((lhs + rhs).relu())

    original = lower_to_loops(lower_to_cpu(module))
    fused = fuse_elementwise(original)

    assert [kernel.opcode for kernel in fused.kernels] == [kernel.opcode for kernel in original.kernels]
    np.testing.assert_array_equal(execute_loop(fused), execute_reference(module))


def test_elementwise_fusion_refuses_shared_producer_value():
    builder = GraphBuilder()
    lhs = builder.tensor([[-3.0], [2.0]], dtype="float32")
    rhs = builder.tensor([[1.0, 2.0, 4.0]], dtype="float32")
    shared = lhs + rhs
    activated = shared.relu()
    module = builder.finish(activated + shared)

    original = lower_to_loops(lower_to_cpu(module))
    fused = fuse_elementwise(original)

    assert all(kernel.opcode != "relu_add" for kernel in fused.kernels)
    np.testing.assert_array_equal(execute_loop(fused), execute_reference(module))
