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
    generate_c,
)
from tiny_tensor_compiler.ir import DType, TensorType


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _type() -> TensorType:
    return TensorType((5,), DType.INT32)


def _five_node_tree(*, terminal_relu: bool = False, include_mul: bool = False) -> LoopProgram:
    type_ = _type()
    identity = IndexMap((0,))
    operation = "mul" if include_mul else "add"
    allocations = 12 if terminal_relu else 11
    operations = [*(LoopAlloc(buffer, type_) for buffer in range(allocations))]
    operations.extend(LoopInput(buffer, buffer) for buffer in range(6))
    operations.extend(
        (
            LoopKernel("add", 6, (0, 1), (5,), (identity, identity)),
            LoopKernel(operation, 7, (2, 3), (5,), (identity, identity)),
            LoopKernel("add", 8, (4, 5), (5,), (identity, identity)),
            LoopKernel("add", 9, (6, 7), (5,), (identity, identity)),
            LoopKernel("add", 10, (9, 8), (5,), (identity, identity)),
        )
    )
    result = 10
    if terminal_relu:
        operations.append(LoopKernel("relu", 11, (10,), (5,), (identity,)))
        result = 11
    operations.append(LoopReturn(result))
    return LoopProgram(tuple(operations))


def _inputs() -> list[np.ndarray]:
    return [
        np.array([1, -2, 3, -4, 5], dtype=np.int32),
        np.array([6, 7, -8, -9, 10], dtype=np.int32),
        np.array([-11, 12, 13, -14, 15], dtype=np.int32),
        np.array([16, -17, 18, 19, -20], dtype=np.int32),
        np.array([21, 22, -23, 24, -25], dtype=np.int32),
        np.array([-26, 27, 28, -29, 30], dtype=np.int32),
    ]


def test_five_node_tree_uses_structured_generic_dag_and_executes_end_to_end() -> None:
    program = _five_node_tree()
    fused = fuse_elementwise(program)

    assert len(fused.kernels) == 1
    kernel = fused.kernels[0]
    assert kernel.opcode == "fused_dag"
    assert kernel.fused_expression is not None
    assert kernel.fused_expression.family == "generic-dag"
    assert kernel.fused_expression.input_count == 6
    assert tuple(step.opcode for step in kernel.fused_expression.steps) == ("add",) * 5

    inputs = _inputs()
    expected = execute_loop(program, inputs=inputs)
    np.testing.assert_array_equal(execute_loop(fused, inputs=inputs), expected)

    source = generate_c(fused)
    assert source.count("_mm_add_epi32") == 5

    _default_compiler_or_skip()
    np.testing.assert_array_equal(execute_native(fused, inputs=inputs), expected)


def test_generic_dag_with_multiply_keeps_scalar_c_fallback_and_native_semantics() -> None:
    program = _five_node_tree(include_mul=True)
    fused = fuse_elementwise(program)

    assert len(fused.kernels) == 1
    kernel = fused.kernels[0]
    assert kernel.opcode == "fused_dag"
    assert kernel.fused_expression is not None
    assert "mul" in tuple(step.opcode for step in kernel.fused_expression.steps)

    source = generate_c(fused)
    assert "#if TINY_TENSOR_HAS_SSE2" not in source.split("TINY_TENSOR_EXPORT void", 1)[1]

    inputs = _inputs()
    expected = execute_loop(program, inputs=inputs)
    np.testing.assert_array_equal(execute_loop(fused, inputs=inputs), expected)

    _default_compiler_or_skip()
    np.testing.assert_array_equal(execute_native(fused, inputs=inputs), expected)


def test_generic_dag_absorbs_terminal_relu_without_new_opcode_family() -> None:
    program = _five_node_tree(terminal_relu=True)
    fused = fuse_elementwise(program)

    assert len(fused.kernels) == 1
    kernel = fused.kernels[0]
    assert kernel.opcode == "fused_dag"
    assert kernel.fused_expression is not None
    assert kernel.fused_expression.family == "generic-dag"
    assert kernel.fused_expression.terminal_relu
    assert tuple(step.opcode for step in kernel.fused_expression.steps) == ("add",) * 5 + ("relu",)

    inputs = _inputs()
    expected = execute_loop(program, inputs=inputs)
    np.testing.assert_array_equal(execute_loop(fused, inputs=inputs), expected)

    _default_compiler_or_skip()
    np.testing.assert_array_equal(execute_native(fused, inputs=inputs), expected)


def test_four_node_topologies_keep_legacy_compatibility_encoding() -> None:
    type_ = _type()
    identity = IndexMap((0,))
    program = LoopProgram(
        (
            *(LoopAlloc(buffer, type_) for buffer in range(9)),
            *(LoopInput(buffer, buffer) for buffer in range(5)),
            LoopKernel("add", 5, (0, 1), (5,), (identity, identity)),
            LoopKernel("add", 6, (5, 2), (5,), (identity, identity)),
            LoopKernel("add", 7, (3, 4), (5,), (identity, identity)),
            LoopKernel("add", 8, (6, 7), (5,), (identity, identity)),
            LoopReturn(8),
        )
    )

    fused = fuse_elementwise(program)
    assert [kernel.opcode for kernel in fused.kernels] == ["chain_tree_add_add_add_add"]
    assert fused.kernels[0].fused_expression is not None
    assert fused.kernels[0].fused_expression.family == "chain-tree"


def test_returned_internal_value_blocks_generic_dag_fusion() -> None:
    program = _five_node_tree()
    operations = list(program.operations[:-1])
    operations.extend((LoopReturn(6), LoopReturn(10)))
    returned = LoopProgram(tuple(operations))

    fused = fuse_elementwise(returned)
    assert all(kernel.opcode != "fused_dag" for kernel in fused.kernels)
