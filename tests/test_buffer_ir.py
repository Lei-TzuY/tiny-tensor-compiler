import numpy as np
import pytest

from tiny_tensor_compiler import GraphBuilder, execute_cpu, execute_reference, lower_to_cpu
from tiny_tensor_compiler.lowering import BufferAlloc, BufferKernel, BufferReturn, CPUProgram


def build_example():
    builder = GraphBuilder()
    x = builder.tensor([1, 2, 3])
    return builder.finish((x * 2 + 1).relu())


def test_lowering_emits_explicit_deterministic_buffer_ir():
    module = build_example()
    program = lower_to_cpu(module)

    expected = """alloc b0 : tensor<3xi64>
b0 = const [1, 2, 3]
alloc b1 : tensor<i64>
b1 = const 2
alloc b2 : tensor<3xi64>
b2 = mul b0, b1
alloc b3 : tensor<i64>
b3 = const 1
alloc b4 : tensor<3xi64>
b4 = add b2, b3
alloc b5 : tensor<3xi64>
b5 = relu b4
return b5"""
    assert program.dump() == expected
    assert program.dump() == expected

    allocs = [op for op in program.operations if isinstance(op, BufferAlloc)]
    kernels = [op for op in program.operations if isinstance(op, BufferKernel)]
    returns = [op for op in program.operations if isinstance(op, BufferReturn)]
    assert len(allocs) == 6
    assert len(kernels) == 6
    assert len(returns) == 1
    assert [alloc.buffer for alloc in allocs] == list(range(6))


def test_buffer_ir_execution_matches_tensor_reference():
    builder = GraphBuilder()
    lhs = builder.tensor([[1.0], [2.0]], dtype="float32")
    rhs = builder.tensor([[10.0, 20.0, 30.0]], dtype="float32")
    module = builder.finish((lhs + rhs).relu())

    expected = execute_reference(module)
    actual = execute_cpu(lower_to_cpu(module))
    np.testing.assert_array_equal(actual, expected)


def test_buffer_program_rejects_kernel_before_allocation():
    with pytest.raises(ValueError, match="buffer b0 is not allocated"):
        CPUProgram(
            (
                BufferKernel("relu", output=0, inputs=(1,)),
                BufferReturn(0),
            )
        )
