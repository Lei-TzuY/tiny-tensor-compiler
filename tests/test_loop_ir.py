import numpy as np

from tiny_tensor_compiler import (
    GraphBuilder,
    LoopKernel,
    execute_cpu,
    execute_reference,
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
