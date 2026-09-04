import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    execute_native,
    execute_reference,
    fuse_elementwise,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
)


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
