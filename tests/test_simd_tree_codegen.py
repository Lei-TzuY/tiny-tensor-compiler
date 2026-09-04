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


def _manual_tree_program(
    *,
    opcode: str = "tree_add_add_add",
    dtype: DType = DType.INT32,
    output_shape: tuple[int, ...] = (9,),
    input_types: tuple[TensorType, TensorType, TensorType, TensorType] | None = None,
    input_maps: tuple[IndexMap, IndexMap, IndexMap, IndexMap] | None = None,
) -> LoopProgram:
    output_type = TensorType(output_shape, dtype)
    if input_types is None:
        input_types = (output_type, output_type, output_type, output_type)
    if input_maps is None:
        identity = IndexMap(tuple(range(len(output_shape))))
        input_maps = (identity, identity, identity, identity)

    return LoopProgram(
        (
            LoopAlloc(0, input_types[0]),
            LoopAlloc(1, input_types[1]),
            LoopAlloc(2, input_types[2]),
            LoopAlloc(3, input_types[3]),
            LoopAlloc(4, output_type),
            LoopInput(0, 0),
            LoopInput(1, 1),
            LoopInput(2, 2),
            LoopInput(3, 3),
            LoopKernel(
                opcode=opcode,
                output=4,
                inputs=(0, 1, 2, 3),
                iteration_shape=output_shape,
                input_maps=input_maps,
            ),
            LoopReturn(4),
        )
    )


def test_generate_c_uses_sse2_for_contiguous_i32_add_tree_with_scalar_tail():
    source = generate_c(_manual_tree_program())

    assert "__m128i a = _mm_loadu_si128((const __m128i *)&p0[n]);" in source
    assert "__m128i b = _mm_loadu_si128((const __m128i *)&p1[n]);" in source
    assert "__m128i c = _mm_loadu_si128((const __m128i *)&p2[n]);" in source
    assert "__m128i d = _mm_loadu_si128((const __m128i *)&p3[n]);" in source
    assert "__m128i left = _mm_add_epi32(a, b);" in source
    assert "__m128i right = _mm_add_epi32(c, d);" in source
    assert "__m128i result = _mm_add_epi32(left, right);" in source
    assert "_mm_storeu_si128((__m128i *)&p4[n], result);" in source
    assert "for (; n < 9; ++n)" in source
    assert "int32_t left = ((int32_t)p0[n] + (int32_t)p1[n]);" in source
    assert "int32_t right = ((int32_t)p2[n] + (int32_t)p3[n]);" in source
    assert "p4[n] = ((int32_t)left + (int32_t)right);" in source


def test_sse2_i32_add_tree_matches_reference_across_branch_and_root_wraps():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    a = builder.input((9,), dtype="int32")
    b = builder.input((9,), dtype="int32")
    c = builder.input((9,), dtype="int32")
    d = builder.input((9,), dtype="int32")
    module = builder.finish((a + b) + (c + d))
    inputs = [
        np.array(
            [
                2_147_483_647,
                -2_147_483_648,
                2_000_000_000,
                -2_000_000_000,
                1_500_000_000,
                -1_500_000_000,
                17,
                -91,
                123,
            ],
            dtype=np.int32,
        ),
        np.array(
            [1, -1, 1_000_000_000, -1_000_000_000, 1_500_000_000, -1_500_000_000, -17, 91, -456],
            dtype=np.int32,
        ),
        np.array(
            [
                2_147_483_647,
                -2_147_483_648,
                1_900_000_000,
                -1_900_000_000,
                1_600_000_000,
                -1_600_000_000,
                5,
                -5,
                789,
            ],
            dtype=np.int32,
        ),
        np.array(
            [1, -1, 900_000_000, -900_000_000, 1_600_000_000, -1_600_000_000, -5, 5, -321],
            dtype=np.int32,
        ),
    ]

    expected = execute_reference(module, inputs=inputs)
    actual = execute_native(_manual_tree_program(), inputs=inputs)

    np.testing.assert_array_equal(actual, expected)


def test_sse2_add_tree_path_is_not_used_outside_exact_i32_contiguous_contract():
    int64_source = generate_c(_manual_tree_program(dtype=DType.INT64))
    scalar_source = generate_c(_manual_tree_program(output_shape=()))
    other_opcode_source = generate_c(_manual_tree_program(opcode="tree_add_mul_add"))

    lhs_type = TensorType((2, 1), DType.INT32)
    rhs_type = TensorType((1, 3), DType.INT32)
    broadcast_source = generate_c(
        _manual_tree_program(
            output_shape=(2, 3),
            input_types=(lhs_type, rhs_type, lhs_type, rhs_type),
            input_maps=(
                IndexMap((0, None)),
                IndexMap((None, 1)),
                IndexMap((0, None)),
                IndexMap((None, 1)),
            ),
        )
    )

    signature = "__m128i result = _mm_add_epi32(left, right);"
    assert signature not in int64_source
    assert signature not in scalar_source
    assert signature not in other_opcode_source
    assert signature not in broadcast_source
