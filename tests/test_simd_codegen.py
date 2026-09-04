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
    fuse_elementwise,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
)
from tiny_tensor_compiler.ir import DType, TensorType


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _contiguous_i32_add_program(length: int = 9):
    builder = GraphBuilder()
    lhs = builder.input((length,), dtype="int32")
    rhs = builder.input((length,), dtype="int32")
    module = builder.finish(lhs + rhs)
    return module, lower_to_loops(lower_to_cpu(module))


def _contiguous_i32_relu_program(length: int = 9):
    builder = GraphBuilder()
    x = builder.input((length,), dtype="int32")
    module = builder.finish(x.relu())
    return module, lower_to_loops(lower_to_cpu(module))


def _contiguous_i32_relu_add_program(length: int = 9):
    builder = GraphBuilder()
    lhs = builder.input((length,), dtype="int32")
    rhs = builder.input((length,), dtype="int32")
    activated = (lhs + rhs).relu()
    module = builder.finish((activated + lhs) + rhs)
    loops = lower_to_loops(lower_to_cpu(module))
    return module, fuse_elementwise(loops)


def _manual_chain_program(
    *,
    opcode: str = "chain_add_add",
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


def test_generate_c_uses_sse2_for_contiguous_i32_add_with_scalar_tail():
    _, program = _contiguous_i32_add_program()

    source = generate_c(program)

    assert "#define TINY_TENSOR_HAS_SSE2 1" in source
    assert "#include <emmintrin.h>" in source
    assert "for (; n + 4 <= 9; n += 4)" in source
    assert "_mm_loadu_si128" in source
    assert "_mm_add_epi32" in source
    assert "_mm_storeu_si128" in source
    assert "for (; n < 9; ++n)" in source


def test_sse2_i32_add_matches_reference_with_overflow_and_tail():
    _default_compiler_or_skip()
    module, program = _contiguous_i32_add_program()
    inputs = [
        np.array(
            [
                2_147_483_647,
                -2_147_483_648,
                2_000_000_000,
                -2_000_000_000,
                1,
                -1,
                17,
                -91,
                123,
            ],
            dtype=np.int32,
        ),
        np.array(
            [1, -1, 2_000_000_000, -2_000_000_000, -2, 2, -17, 91, -456],
            dtype=np.int32,
        ),
    ]

    expected = execute_reference(module, inputs=inputs)
    actual = execute_native(program, inputs=inputs)

    np.testing.assert_array_equal(actual, expected)


def test_generate_c_uses_sse2_for_contiguous_i32_relu_with_scalar_tail():
    _, program = _contiguous_i32_relu_program()

    source = generate_c(program)

    assert "for (; n + 4 <= 9; n += 4)" in source
    assert "__m128i value = _mm_loadu_si128((const __m128i *)&p0[n]);" in source
    assert "__m128i zero = _mm_setzero_si128();" in source
    assert "__m128i positive = _mm_cmpgt_epi32(value, zero);" in source
    assert "__m128i relu = _mm_and_si128(value, positive);" in source
    assert "_mm_storeu_si128((__m128i *)&p1[n], relu);" in source
    assert "for (; n < 9; ++n)" in source
    assert "p1[n] = value < 0 ? 0 : value;" in source


def test_sse2_i32_relu_matches_reference_for_extrema_zero_and_tail():
    _default_compiler_or_skip()
    module, program = _contiguous_i32_relu_program()
    inputs = [
        np.array(
            [
                -2_147_483_648,
                -2_000_000_000,
                -1,
                0,
                1,
                17,
                91,
                2_000_000_000,
                2_147_483_647,
            ],
            dtype=np.int32,
        )
    ]

    expected = execute_reference(module, inputs=inputs)
    actual = execute_native(program, inputs=inputs)

    np.testing.assert_array_equal(actual, expected)


def test_generate_c_uses_sse2_for_contiguous_i32_relu_add_with_scalar_tail():
    _, program = _contiguous_i32_relu_add_program()

    source = generate_c(program)

    assert "for (; n + 4 <= 9; n += 4)" in source
    assert "__m128i sum = _mm_add_epi32(lhs, rhs);" in source
    assert "__m128i zero = _mm_setzero_si128();" in source
    assert "__m128i positive = _mm_cmpgt_epi32(sum, zero);" in source
    assert "__m128i relu = _mm_and_si128(sum, positive);" in source
    assert "_mm_storeu_si128" in source
    assert "for (; n < 9; ++n)" in source
    assert "int32_t value = ((int32_t)p0[n] + (int32_t)p1[n]);" in source


def test_sse2_i32_relu_add_matches_reference_after_wrapping_add_and_tail():
    _default_compiler_or_skip()
    module, program = _contiguous_i32_relu_add_program()
    inputs = [
        np.array(
            [
                2_147_483_647,
                -2_147_483_648,
                2_000_000_000,
                -2_000_000_000,
                5,
                -5,
                17,
                -91,
                123,
            ],
            dtype=np.int32,
        ),
        np.array(
            [1, -1, 2_000_000_000, -2_000_000_000, -10, 10, -17, 91, -456],
            dtype=np.int32,
        ),
    ]

    expected = execute_reference(module, inputs=inputs)
    actual = execute_native(program, inputs=inputs)

    np.testing.assert_array_equal(actual, expected)


def test_generate_c_uses_sse2_for_contiguous_i32_chain_add_add_with_scalar_tail():
    source = generate_c(_manual_chain_program())

    assert "__m128i lhs = _mm_loadu_si128((const __m128i *)&p0[n]);" in source
    assert "__m128i rhs = _mm_loadu_si128((const __m128i *)&p1[n]);" in source
    assert "__m128i tail = _mm_loadu_si128((const __m128i *)&p2[n]);" in source
    assert "__m128i inner = _mm_add_epi32(lhs, rhs);" in source
    assert "__m128i result = _mm_add_epi32(inner, tail);" in source
    assert "_mm_storeu_si128((__m128i *)&p3[n], result);" in source
    assert "for (; n < 9; ++n)" in source
    assert "int32_t inner = ((int32_t)p0[n] + (int32_t)p1[n]);" in source
    assert "p3[n] = ((int32_t)inner + (int32_t)p2[n]);" in source


def test_sse2_i32_chain_add_add_matches_reference_across_two_wrap_boundaries_and_tail():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    lhs = builder.input((9,), dtype="int32")
    rhs = builder.input((9,), dtype="int32")
    tail = builder.input((9,), dtype="int32")
    module = builder.finish((lhs + rhs) + tail)
    program = _manual_chain_program()
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


def test_sse2_chain_add_add_path_is_not_used_outside_exact_i32_contiguous_contract():
    int64_source = generate_c(_manual_chain_program(dtype=DType.INT64))
    scalar_source = generate_c(_manual_chain_program(output_shape=()))
    other_opcode_source = generate_c(_manual_chain_program(opcode="chain_add_mul"))

    broadcast_type_lhs = TensorType((2, 1), DType.INT32)
    broadcast_type_rhs = TensorType((1, 3), DType.INT32)
    broadcast_type_tail = TensorType((), DType.INT32)
    broadcast_source = generate_c(
        _manual_chain_program(
            output_shape=(2, 3),
            input_types=(broadcast_type_lhs, broadcast_type_rhs, broadcast_type_tail),
            input_maps=(IndexMap((0, None)), IndexMap((None, 1)), IndexMap(())),
        )
    )

    signature = "__m128i inner = _mm_add_epi32(lhs, rhs);"
    assert signature not in int64_source
    assert signature not in scalar_source
    assert signature not in other_opcode_source
    assert signature not in broadcast_source


def test_sse2_relu_path_is_not_used_for_float_broadcast_or_scalar_kernels():
    float_builder = GraphBuilder()
    float_x = float_builder.input((9,), dtype="float32")
    float_program = lower_to_loops(lower_to_cpu(float_builder.finish(float_x.relu())))

    broadcast_builder = GraphBuilder()
    broadcast_lhs = broadcast_builder.input((2, 1), dtype="int32")
    broadcast_rhs = broadcast_builder.input((1, 3), dtype="int32")
    broadcast_module = broadcast_builder.finish((broadcast_lhs + broadcast_rhs).relu())
    broadcast_program = fuse_elementwise(lower_to_loops(lower_to_cpu(broadcast_module)))

    scalar_builder = GraphBuilder()
    scalar = scalar_builder.input((), dtype="int32")
    scalar_program = lower_to_loops(lower_to_cpu(scalar_builder.finish(scalar.relu())))

    assert "_mm_cmpgt_epi32" not in generate_c(float_program)
    assert "_mm_cmpgt_epi32" not in generate_c(broadcast_program)
    assert "_mm_cmpgt_epi32" not in generate_c(scalar_program)


def test_sse2_relu_add_path_is_not_used_for_float_or_broadcast_kernels():
    float_builder = GraphBuilder()
    float_lhs = float_builder.input((9,), dtype="float32")
    float_rhs = float_builder.input((9,), dtype="float32")
    float_module = float_builder.finish((float_lhs + float_rhs).relu())
    float_program = fuse_elementwise(lower_to_loops(lower_to_cpu(float_module)))

    broadcast_builder = GraphBuilder()
    broadcast_lhs = broadcast_builder.input((2, 1), dtype="int32")
    broadcast_rhs = broadcast_builder.input((1, 3), dtype="int32")
    broadcast_module = broadcast_builder.finish((broadcast_lhs + broadcast_rhs).relu())
    broadcast_program = fuse_elementwise(lower_to_loops(lower_to_cpu(broadcast_module)))

    assert "_mm_cmpgt_epi32" not in generate_c(float_program)
    assert "_mm_cmpgt_epi32" not in generate_c(broadcast_program)


def test_sse2_path_is_not_used_for_float_or_broadcast_add():
    float_builder = GraphBuilder()
    float_lhs = float_builder.input((9,), dtype="float32")
    float_rhs = float_builder.input((9,), dtype="float32")
    float_program = lower_to_loops(lower_to_cpu(float_builder.finish(float_lhs + float_rhs)))

    broadcast_builder = GraphBuilder()
    broadcast_lhs = broadcast_builder.input((2, 1), dtype="int32")
    broadcast_rhs = broadcast_builder.input((1, 3), dtype="int32")
    broadcast_program = lower_to_loops(
        lower_to_cpu(broadcast_builder.finish(broadcast_lhs + broadcast_rhs))
    )

    assert "_mm_add_epi32" not in generate_c(float_program)
    assert "_mm_add_epi32" not in generate_c(broadcast_program)
