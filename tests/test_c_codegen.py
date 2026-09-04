import os
import shutil
import subprocess

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


def _broadcast_program():
    builder = GraphBuilder()
    lhs = builder.tensor([[1], [2]], dtype="int32")
    rhs = builder.tensor([[10, 20, 30]], dtype="int32")
    return lower_to_loops(lower_to_cpu(builder.finish(lhs + rhs)))


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def test_generate_c_is_deterministic_and_encodes_broadcast_offsets():
    program = _broadcast_program()

    first = generate_c(program)
    second = generate_c(program)

    assert first == second
    assert "#define TINY_TENSOR_EXPORT __declspec(dllexport)" in first
    assert "TINY_TENSOR_EXPORT void tiny_tensor_run(int32_t *out)" in first
    assert "for (int64_t i0 = 0; i0 < 2; ++i0)" in first
    assert "for (int64_t i1 = 0; i1 < 3; ++i1)" in first
    assert "p0[i0]" in first
    assert "p1[i1]" in first
    assert "p2[(i0 * 3) + i1]" in first
    assert (
        "\n        TINY_TENSOR_VECTORIZE_LOOP\n"
        "        for (int64_t i0 = 0; i0 < 2; ++i0)"
    ) not in first


def test_generate_c_linearizes_contiguous_multidimensional_kernel():
    builder = GraphBuilder()
    lhs = builder.input((2, 3), dtype="int32")
    rhs = builder.input((2, 3), dtype="int32")
    program = lower_to_loops(lower_to_cpu(builder.finish(lhs + rhs)))

    source = generate_c(program)

    assert "#define TINY_TENSOR_VECTORIZE_LOOP __pragma(loop(ivdep))" in source
    assert '#define TINY_TENSOR_VECTORIZE_LOOP _Pragma("clang loop vectorize(enable)")' in source
    assert '#define TINY_TENSOR_VECTORIZE_LOOP _Pragma("GCC ivdep")' in source
    assert (
        "\n        TINY_TENSOR_VECTORIZE_LOOP\n"
        "        for (int64_t n = 0; n < 6; ++n)"
    ) in source
    assert "p2[n] = ((int32_t)p0[n] + (int32_t)p1[n]);" in source
    assert "for (int64_t i0 = 0; i0 < 2; ++i0)" not in source
    assert "for (int64_t i1 = 0; i1 < 3; ++i1)" not in source


def test_linearized_contiguous_kernel_matches_native_reference():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    lhs = builder.input((2, 3), dtype="int32")
    rhs = builder.input((2, 3), dtype="int32")
    module = builder.finish(lhs * rhs + lhs)
    program = lower_to_loops(lower_to_cpu(module))
    inputs = [
        np.array(
            [[2_000_000_000, -2_000_000_000, 65_537], [123_456_789, -91, 17]],
            dtype=np.int32,
        ),
        np.array([[3, -3, 65_537], [-17, 91, -2_000_000_000]], dtype=np.int32),
    ]

    source = generate_c(program)
    expected = execute_reference(module, inputs=inputs)

    assert "p2[n] = ((int32_t)p0[n] * (int32_t)p1[n]);" in source
    np.testing.assert_array_equal(execute_native(program, inputs=inputs), expected)


def test_generate_c_encodes_scalar_broadcast_and_relu_without_vector_intrinsics():
    builder = GraphBuilder()
    x = builder.tensor([-2.0, 1.5, 3.0], dtype="float32")
    scalar = builder.tensor(2.0, dtype="float32")
    program = lower_to_loops(lower_to_cpu(builder.finish((x + scalar).relu())))

    source = generate_c(program)

    assert "p1[0]" in source
    assert "value == 0.0f" in source
    assert "fabsf(value)" in source
    assert "np." not in source
    assert "simd" not in source.lower()


def test_generate_c_zero_extent_uses_valid_storage_and_zero_trip_loop():
    builder = GraphBuilder()
    x = builder.tensor([], dtype="int32")
    program = lower_to_loops(lower_to_cpu(builder.finish(x.relu())))

    source = generate_c(program)

    assert "int32_t p0[1];" in source
    assert "i0 < 0" in source


def test_generated_c_passes_c11_syntax_check_when_compiler_is_available(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("no C compiler available")

    path = tmp_path / "generated.c"
    path.write_text(generate_c(_broadcast_program()), encoding="utf-8")
    subprocess.run(
        [compiler, "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", str(path)],
        check=True,
    )
