import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    execute_native,
    execute_reference,
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
