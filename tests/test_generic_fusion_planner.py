import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    IndexMap,
    LoopAlloc,
    LoopInput,
    LoopKernel,
    LoopProgram,
    LoopReturn,
    execute_loop,
    execute_native,
    fuse_elementwise,
)
from tiny_tensor_compiler.fused_expr import binary_tree_expression, chain_tree_expression
from tiny_tensor_compiler.ir import DType, TensorType


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _vector_type() -> TensorType:
    return TensorType((4,), DType.INT32)


def _mirror_tree_program() -> LoopProgram:
    type_ = _vector_type()
    identity = IndexMap((0,))
    return LoopProgram(
        (
            *(LoopAlloc(buffer, type_) for buffer in range(7)),
            LoopInput(0, 0),
            LoopInput(1, 1),
            LoopInput(2, 2),
            LoopInput(3, 3),
            # Materialize the root's RHS producer before its LHS producer.
            LoopKernel("mul", 5, (2, 3), (4,), (identity, identity)),
            LoopKernel("add", 4, (0, 1), (4,), (identity, identity)),
            LoopKernel("add", 6, (4, 5), (4,), (identity, identity)),
            LoopReturn(6),
        )
    )


def _mirror_chain_tree_program() -> LoopProgram:
    type_ = _vector_type()
    identity = IndexMap((0,))
    return LoopProgram(
        (
            *(LoopAlloc(buffer, type_) for buffer in range(9)),
            LoopInput(0, 0),
            LoopInput(1, 1),
            LoopInput(2, 2),
            LoopInput(3, 3),
            LoopInput(4, 4),
            # The simple root branch is emitted first. The chain is the root RHS,
            # and its internal producer is itself consumed as the chain RHS.
            LoopKernel("mul", 5, (0, 1), (4,), (identity, identity)),
            LoopKernel("add", 6, (2, 3), (4,), (identity, identity)),
            LoopKernel("mul", 7, (4, 6), (4,), (identity, identity)),
            LoopKernel("add", 8, (5, 7), (4,), (identity, identity)),
            LoopReturn(8),
        )
    )


def test_generic_planner_fuses_tree_independent_of_producer_order() -> None:
    program = _mirror_tree_program()
    fused = fuse_elementwise(program)

    assert [kernel.opcode for kernel in fused.kernels] == ["tree_add_mul_add"]
    assert fused.kernels[0].fused_expression == binary_tree_expression("add", "mul", "add")
    assert fused.kernels[0].inputs == (0, 1, 2, 3)

    inputs = [
        np.array([1, -2, 3, -4], dtype=np.int32),
        np.array([5, 6, -7, -8], dtype=np.int32),
        np.array([2, 3, 4, 5], dtype=np.int32),
        np.array([-1, 2, -3, 4], dtype=np.int32),
    ]
    expected = execute_loop(program, inputs=inputs)
    np.testing.assert_array_equal(execute_loop(fused, inputs=inputs), expected)

    _default_compiler_or_skip()
    np.testing.assert_array_equal(execute_native(fused, inputs=inputs), expected)


def test_generic_planner_fuses_mirrored_chain_tree_without_reassociation() -> None:
    program = _mirror_chain_tree_program()
    fused = fuse_elementwise(program)

    assert [kernel.opcode for kernel in fused.kernels] == ["chain_tree_add_mul_mul_add"]
    assert fused.kernels[0].fused_expression == chain_tree_expression(
        "add",
        "mul",
        "mul",
        "add",
    )
    assert fused.kernels[0].inputs == (2, 3, 4, 0, 1)

    inputs = [
        np.array([2, -3, 5, -7], dtype=np.int32),
        np.array([11, 13, -17, -19], dtype=np.int32),
        np.array([23, -29, 31, -37], dtype=np.int32),
        np.array([-41, 43, 47, -53], dtype=np.int32),
        np.array([59, -61, 67, -71], dtype=np.int32),
    ]
    expected = execute_loop(program, inputs=inputs)
    np.testing.assert_array_equal(execute_loop(fused, inputs=inputs), expected)

    _default_compiler_or_skip()
    np.testing.assert_array_equal(execute_native(fused, inputs=inputs), expected)


def test_generic_planner_does_not_reassociate_three_deep_chain() -> None:
    type_ = _vector_type()
    identity = IndexMap((0,))
    program = LoopProgram(
        (
            *(LoopAlloc(buffer, type_) for buffer in range(7)),
            LoopInput(0, 0),
            LoopInput(1, 1),
            LoopInput(2, 2),
            LoopInput(3, 3),
            LoopKernel("add", 4, (0, 1), (4,), (identity, identity)),
            LoopKernel("mul", 5, (4, 2), (4,), (identity, identity)),
            LoopKernel("add", 6, (5, 3), (4,), (identity, identity)),
            LoopReturn(6),
        )
    )

    fused = fuse_elementwise(program)

    assert [kernel.opcode for kernel in fused.kernels] == ["chain_add_mul", "add"]


def test_generic_planner_is_idempotent_for_mirrored_topologies() -> None:
    for program in (_mirror_tree_program(), _mirror_chain_tree_program()):
        fused = fuse_elementwise(program)
        assert fuse_elementwise(fused).dump() == fused.dump()
