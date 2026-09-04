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


def _manual_relu_chain_program(
    *,
    opcode: str = "relu_chain_add_add",
    dtype: DType = DType.INT32,
    output_shape: tuple[int, ...] = (9,),
    input_types: tuple[TensorType, TensorType, TensorType] | None = None,
    input_maps: tuple[IndexMap, IndexMap, IndexMap] | None = None,
) -> LoopProgram:
    output_type = TensorType(output_shape, dtype)
    if input_types is None:
        input_types = (output_type, output_type, output_type)
    if input_maps is None:
        identity = IndexMap(tuple(range(len(output_shape))))
        input_maps = (identity, identity, identity)

    return LoopProgram(
        (
            LoopAlloc(0, input_types[0]),
            LoopAlloc(1, input_types[1]),
            LoopAlloc(2, input_types[2]),
            LoopAlloc(3, output_type),
            LoopInput(0, 0),
            LoopInput(1, 1),
            LoopInput(2, 2),
            LoopKernel(
                opcode=opcode,
                output=3,
                inputs=(0, 1, 2),
                iteration_shape=output_shape,
                input_maps=input_maps,
            ),
            LoopReturn(3),
        )
    )


def test_generate_c_uses_sse2_for_contiguous_i32_relu_chain_add_add_with_scalar_tail():
    source = generate_c(_manual_relu_chain_program())

    assert "__m128i lhs = _mm_loadu_si128((const __m128i *)&p0[n]);" in source
    assert "__m128i rhs = _mm_loadu_si128((const __m128i *)&p1[n]);" in source
    assert "__m128i tail = _mm_loadu_si128((const __m128i *)&p2[n]);" in source
    assert "__m128i inner = _mm_add_epi32(lhs, rhs);" in source
    assert "__m128i result = _mm_add_epi32(inner, tail);" in source
    assert "__m128i zero = _mm_setzero_si128();" in source
    assert "__m128i positive = _mm_cmpgt_epi32(result, zero);" in source
    assert "__m128i relu = _mm_and_si128(result, positive);" in source
    assert "_mm_storeu_si128((__m128i *)&p3[n], relu);" in source
    assert "for (; n < 9; ++n)" in source
    assert "int32_t inner = ((int32_t)p0[n] + (int32_t)p1[n]);" in source
    assert "int32_t value = ((int32_t)inner + (int32_t)p2[n]);" in source
    assert "p3[n] = value < 0 ? 0 : value;" in source


def test_sse2_i32_relu_chain_add_add_matches_reference_across_wraps_relu_and_tail():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    lhs = builder.input((9,), dtype="int32")
    rhs = builder.input((9,), dtype="int32")
    tail = builder.input((9,), dtype="int32")
    module = builder.finish(((lhs + rhs) + tail).relu())
    program = _manual_relu_chain_program()
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
            [
                1,
                -1,
                1_000_000_000,
                -1_000_000_000,
                1_500_000_000,
                -1_500_000_000,
                -17,
                91,
                -456,
            ],
            dtype=np.int32,
        ),
        np.array(
            [
                2_147_483_647,
                -2_147_483_648,
                1_000_000_000,
                -1_000_000_000,
                1_500_000_000,
                -1_500_000_000,
                5,
                -5,
                789,
            ],
            dtype=np.int32,
        ),
    ]

    expected = execute_reference(module, inputs=inputs)
    actual = execute_native(program, inputs=inputs)

    np.testing.assert_array_equal(actual, expected)


def test_sse2_relu_chain_add_add_path_requires_exact_i32_contiguous_contract():
    int64_source = generate_c(_manual_relu_chain_program(dtype=DType.INT64))
    scalar_source = generate_c(_manual_relu_chain_program(output_shape=()))
    other_opcode_source = generate_c(_manual_relu_chain_program(opcode="relu_chain_add_mul"))

    broadcast_type_lhs = TensorType((2, 1), DType.INT32)
    broadcast_type_rhs = TensorType((1, 3), DType.INT32)
    broadcast_type_tail = TensorType((), DType.INT32)
    broadcast_source = generate_c(
        _manual_relu_chain_program(
            output_shape=(2, 3),
            input_types=(broadcast_type_lhs, broadcast_type_rhs, broadcast_type_tail),
            input_maps=(IndexMap((0, None)), IndexMap((None, 1)), IndexMap(())),
        )
    )

    signature = "__m128i positive = _mm_cmpgt_epi32(result, zero);"
    assert signature not in int64_source
    assert signature not in scalar_source
    assert signature not in other_opcode_source
    assert signature not in broadcast_source
