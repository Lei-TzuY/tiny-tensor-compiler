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
    execute_native,
    execute_reference,
    generate_c,
)
from tiny_tensor_compiler.ir import DType, TensorType


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _manual_fused_program(
    opcode: str,
    input_types: tuple[TensorType, ...],
    output_type: TensorType,
    input_maps: tuple[IndexMap, ...] | None = None,
) -> LoopProgram:
    if input_maps is None:
        identity = IndexMap(tuple(range(len(output_type.shape))))
        input_maps = (identity,) * len(input_types)
    output = len(input_types)
    return LoopProgram(
        (
            *(LoopAlloc(index, type_) for index, type_ in enumerate(input_types)),
            LoopAlloc(output, output_type),
            *(LoopInput(index, index) for index in range(len(input_types))),
            LoopKernel(
                opcode=opcode,
                output=output,
                inputs=tuple(range(len(input_types))),
                iteration_shape=output_type.shape,
                input_maps=input_maps,
            ),
            LoopReturn(output),
        )
    )


def test_relu_add_tree_uses_expression_driven_sse2_and_matches_reference():
    type_ = TensorType((9,), DType.INT32)
    program = _manual_fused_program(
        "relu_tree_add_add_add",
        (type_,) * 4,
        type_,
    )
    source = generate_c(program)

    assert "__m128i left = _mm_add_epi32(a, b);" in source
    assert "__m128i right = _mm_add_epi32(c, d);" in source
    assert "__m128i result = _mm_add_epi32(left, right);" in source
    assert "__m128i positive = _mm_cmpgt_epi32(result, zero);" in source
    assert "_mm_storeu_si128" in source

    builder = GraphBuilder()
    a = builder.input((9,), dtype="int32")
    b = builder.input((9,), dtype="int32")
    c = builder.input((9,), dtype="int32")
    d = builder.input((9,), dtype="int32")
    module = builder.finish(((a + b) + (c + d)).relu())
    inputs = [
        np.array(
            [2_147_483_647, -2_147_483_648, 7, -9, 11, -13, 17, -19, 23],
            dtype=np.int32,
        ),
        np.array([1, -1, -8, 10, -12, 14, -18, 20, -24], dtype=np.int32),
        np.array(
            [2_000_000_000, -2_000_000_000, 5, -6, 29, -31, 37, -41, 43],
            dtype=np.int32,
        ),
        np.array(
            [1_000_000_000, -1_000_000_000, -4, 7, -30, 32, -38, 42, -44],
            dtype=np.int32,
        ),
    ]

    _default_compiler_or_skip()
    np.testing.assert_array_equal(
        execute_native(program, inputs=inputs),
        execute_reference(module, inputs=inputs),
    )


def test_add_chain_tree_uses_expression_driven_sse2_and_matches_reference():
    type_ = TensorType((9,), DType.INT32)
    program = _manual_fused_program(
        "chain_tree_add_add_add_add",
        (type_,) * 5,
        type_,
    )
    source = generate_c(program)

    assert "__m128i inner = _mm_add_epi32(first_lhs, first_rhs);" in source
    assert "__m128i left = _mm_add_epi32(inner, left_tail);" in source
    assert "__m128i right = _mm_add_epi32(right_lhs, right_rhs);" in source
    assert "__m128i result = _mm_add_epi32(left, right);" in source
    assert "_mm_storeu_si128" in source

    builder = GraphBuilder()
    a = builder.input((9,), dtype="int32")
    b = builder.input((9,), dtype="int32")
    c = builder.input((9,), dtype="int32")
    d = builder.input((9,), dtype="int32")
    e = builder.input((9,), dtype="int32")
    module = builder.finish(((a + b) + c) + (d + e))
    inputs = [
        np.array(
            [2_147_483_647, -2_147_483_648, 1, -2, 3, -4, 5, -6, 7],
            dtype=np.int32,
        ),
        np.array([1, -1, 8, -9, 10, -11, 12, -13, 14], dtype=np.int32),
        np.array(
            [1_000_000_000, -1_000_000_000, 15, -16, 17, -18, 19, -20, 21],
            dtype=np.int32,
        ),
        np.array(
            [2_000_000_000, -2_000_000_000, 22, -23, 24, -25, 26, -27, 28],
            dtype=np.int32,
        ),
        np.array(
            [1_000_000_000, -1_000_000_000, -29, 30, -31, 32, -33, 34, -35],
            dtype=np.int32,
        ),
    ]

    _default_compiler_or_skip()
    np.testing.assert_array_equal(
        execute_native(program, inputs=inputs),
        execute_reference(module, inputs=inputs),
    )


def test_expression_driven_sse2_keeps_dtype_broadcast_and_mul_boundaries():
    int64_type = TensorType((9,), DType.INT64)
    int64_source = generate_c(
        _manual_fused_program(
            "relu_tree_add_add_add",
            (int64_type,) * 4,
            int64_type,
        )
    )

    output_type = TensorType((2, 3), DType.INT32)
    lhs_type = TensorType((2, 1), DType.INT32)
    rhs_type = TensorType((1, 3), DType.INT32)
    broadcast_source = generate_c(
        _manual_fused_program(
            "relu_tree_add_add_add",
            (lhs_type, rhs_type, lhs_type, rhs_type),
            output_type,
            (
                IndexMap((0, None)),
                IndexMap((None, 1)),
                IndexMap((0, None)),
                IndexMap((None, 1)),
            ),
        )
    )

    int32_type = TensorType((9,), DType.INT32)
    mul_source = generate_c(
        _manual_fused_program(
            "tree_add_mul_add",
            (int32_type,) * 4,
            int32_type,
        )
    )

    signature = "_mm_loadu_si128"
    assert signature not in int64_source
    assert signature not in broadcast_source
    assert signature not in mul_source
    assert "int32_t right = ((int32_t)p2[n] * (int32_t)p3[n]);" in mul_source
