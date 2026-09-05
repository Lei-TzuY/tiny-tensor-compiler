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


def _fused(module):
    return fuse_elementwise(lower_to_loops(lower_to_cpu(module)))


def test_relu_add_tree_uses_expression_driven_sse2_and_matches_reference():
    builder = GraphBuilder()
    a = builder.input((9,), dtype="int32")
    b = builder.input((9,), dtype="int32")
    c = builder.input((9,), dtype="int32")
    d = builder.input((9,), dtype="int32")
    module = builder.finish(((a + b) + (c + d)).relu())
    fused = _fused(module)

    assert [kernel.opcode for kernel in fused.kernels] == ["relu_tree_add_add_add"]
    source = generate_c(fused)
    assert "__m128i left = _mm_add_epi32(a, b);" in source
    assert "__m128i right = _mm_add_epi32(c, d);" in source
    assert "__m128i result = _mm_add_epi32(left, right);" in source
    assert "__m128i positive = _mm_cmpgt_epi32(result, zero);" in source
    assert "_mm_storeu_si128" in source

    _default_compiler_or_skip()
    inputs = [
        np.array([2_147_483_647, -2_147_483_648, 7, -9, 11, -13, 17, -19, 23], dtype=np.int32),
        np.array([1, -1, -8, 10, -12, 14, -18, 20, -24], dtype=np.int32),
        np.array([2_000_000_000, -2_000_000_000, 5, -6, 29, -31, 37, -41, 43], dtype=np.int32),
        np.array([1_000_000_000, -1_000_000_000, -4, 7, -30, 32, -38, 42, -44], dtype=np.int32),
    ]
    np.testing.assert_array_equal(
        execute_native(fused, inputs=inputs),
        execute_reference(module, inputs=inputs),
    )


def test_add_chain_tree_uses_expression_driven_sse2_and_matches_reference():
    builder = GraphBuilder()
    a = builder.input((9,), dtype="int32")
    b = builder.input((9,), dtype="int32")
    c = builder.input((9,), dtype="int32")
    d = builder.input((9,), dtype="int32")
    e = builder.input((9,), dtype="int32")
    module = builder.finish(((a + b) + c) + (d + e))
    fused = _fused(module)

    assert [kernel.opcode for kernel in fused.kernels] == ["chain_tree_add_add_add_add"]
    source = generate_c(fused)
    assert "__m128i inner = _mm_add_epi32(first_lhs, first_rhs);" in source
    assert "__m128i left = _mm_add_epi32(inner, left_tail);" in source
    assert "__m128i right = _mm_add_epi32(right_lhs, right_rhs);" in source
    assert "__m128i result = _mm_add_epi32(left, right);" in source
    assert "_mm_storeu_si128" in source

    _default_compiler_or_skip()
    inputs = [
        np.array([2_147_483_647, -2_147_483_648, 1, -2, 3, -4, 5, -6, 7], dtype=np.int32),
        np.array([1, -1, 8, -9, 10, -11, 12, -13, 14], dtype=np.int32),
        np.array([1_000_000_000, -1_000_000_000, 15, -16, 17, -18, 19, -20, 21], dtype=np.int32),
        np.array([2_000_000_000, -2_000_000_000, 22, -23, 24, -25, 26, -27, 28], dtype=np.int32),
        np.array([1_000_000_000, -1_000_000_000, -29, 30, -31, 32, -33, 34, -35], dtype=np.int32),
    ]
    np.testing.assert_array_equal(
        execute_native(fused, inputs=inputs),
        execute_reference(module, inputs=inputs),
    )


def test_expression_driven_sse2_keeps_dtype_broadcast_and_mul_boundaries():
    int64_builder = GraphBuilder()
    a64 = int64_builder.input((9,), dtype="int64")
    b64 = int64_builder.input((9,), dtype="int64")
    c64 = int64_builder.input((9,), dtype="int64")
    d64 = int64_builder.input((9,), dtype="int64")
    int64_source = generate_c(_fused(int64_builder.finish(((a64 + b64) + (c64 + d64)).relu())))

    broadcast_builder = GraphBuilder()
    a = broadcast_builder.input((2, 1), dtype="int32")
    b = broadcast_builder.input((1, 3), dtype="int32")
    c = broadcast_builder.input((2, 1), dtype="int32")
    d = broadcast_builder.input((1, 3), dtype="int32")
    broadcast_source = generate_c(
        _fused(broadcast_builder.finish(((a + b) + (c + d)).relu()))
    )

    mul_builder = GraphBuilder()
    ma = mul_builder.input((9,), dtype="int32")
    mb = mul_builder.input((9,), dtype="int32")
    mc = mul_builder.input((9,), dtype="int32")
    md = mul_builder.input((9,), dtype="int32")
    mul_source = generate_c(_fused(mul_builder.finish((ma + mb) + (mc * md))))

    signature = "_mm_loadu_si128"
    assert signature not in int64_source
    assert signature not in broadcast_source
    assert signature not in mul_source
