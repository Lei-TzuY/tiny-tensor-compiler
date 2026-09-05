import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    IndexMap,
    LoopAlloc,
    LoopInput,
    LoopKernel,
    LoopProgram,
    LoopReturn,
    fuse_elementwise,
    lower_to_cpu,
    lower_to_loops,
)
from tiny_tensor_compiler.fused_expr import (
    binary_chain_expression,
    binary_tree_expression,
    chain_tree_expression,
    encode_fused_opcode,
)
from tiny_tensor_compiler.input_binding import borrow_inputs
from tiny_tensor_compiler.ir import DType, TensorType
from tiny_tensor_compiler.loop_ir import fused_expression_for_kernel


def _fused(module):
    return fuse_elementwise(lower_to_loops(lower_to_cpu(module)))


def test_fusion_materializes_structured_chain_tree_and_relu_expressions() -> None:
    builder = GraphBuilder()
    a = builder.input((4,), dtype="int32")
    b = builder.input((4,), dtype="int32")
    c = builder.input((4,), dtype="int32")
    chain = _fused(builder.finish(((a + b) * c).relu()))
    assert [kernel.opcode for kernel in chain.kernels] == ["relu_chain_add_mul"]
    assert chain.kernels[0].fused_expression == binary_chain_expression(
        "add", "mul", terminal_relu=True
    )

    builder = GraphBuilder()
    a = builder.input((4,), dtype="int32")
    b = builder.input((4,), dtype="int32")
    c = builder.input((4,), dtype="int32")
    d = builder.input((4,), dtype="int32")
    tree = _fused(builder.finish((a + b) * (c + d)))
    assert [kernel.opcode for kernel in tree.kernels] == ["tree_add_add_mul"]
    assert tree.kernels[0].fused_expression == binary_tree_expression(
        "add", "add", "mul"
    )

    builder = GraphBuilder()
    a = builder.input((4,), dtype="int32")
    b = builder.input((4,), dtype="int32")
    c = builder.input((4,), dtype="int32")
    d = builder.input((4,), dtype="int32")
    e = builder.input((4,), dtype="int32")
    chain_tree = _fused(builder.finish(((a + b) * c) * (d + e)))
    assert [kernel.opcode for kernel in chain_tree.kernels] == [
        "chain_tree_add_mul_add_mul"
    ]
    assert chain_tree.kernels[0].fused_expression == chain_tree_expression(
        "add", "mul", "add", "mul"
    )

    for program in (chain, tree, chain_tree):
        expression = program.kernels[0].fused_expression
        assert expression is not None
        assert encode_fused_opcode(expression) == program.kernels[0].opcode
        assert program.kernels[0].opcode in program.dump()


def test_legacy_fused_kernel_without_metadata_still_decodes() -> None:
    kernel = LoopKernel(
        opcode="chain_add_mul",
        output=3,
        inputs=(0, 1, 2),
        iteration_shape=(2,),
        input_maps=(IndexMap((0,)),) * 3,
    )
    assert kernel.fused_expression is None
    assert fused_expression_for_kernel(kernel) == binary_chain_expression("add", "mul")


def test_loop_ir_rejects_structured_expression_opcode_mismatch() -> None:
    type_ = TensorType((2,), DType.INT32)
    identity = IndexMap((0,))
    expression = binary_chain_expression("add", "mul")

    with pytest.raises(ValueError, match="metadata does not match opcode"):
        LoopProgram(
            (
                LoopAlloc(0, type_),
                LoopAlloc(1, type_),
                LoopAlloc(2, type_),
                LoopAlloc(3, type_),
                LoopInput(0, 0),
                LoopInput(1, 1),
                LoopInput(2, 2),
                LoopKernel(
                    opcode="chain_mul_add",
                    output=3,
                    inputs=(0, 1, 2),
                    iteration_shape=(2,),
                    input_maps=(identity, identity, identity),
                    fused_expression=expression,
                ),
                LoopReturn(3),
            )
        )


def test_input_borrowing_preserves_structured_fused_expression() -> None:
    builder = GraphBuilder()
    lhs = builder.input((4,), dtype="int32")
    rhs = builder.input((4,), dtype="int32")
    tail = builder.input((4,), dtype="int32")
    fused = _fused(builder.finish((lhs + rhs) * tail))

    expression = fused.kernels[0].fused_expression
    assert expression == binary_chain_expression("add", "mul")

    borrowed = borrow_inputs(fused)
    assert borrowed.kernels[0].fused_expression == expression
    assert borrowed.kernels[0].opcode == fused.kernels[0].opcode
