import shutil
import subprocess

import pytest

from tiny_tensor_compiler import GraphBuilder, generate_c, lower_to_cpu, lower_to_loops


def _broadcast_program():
    builder = GraphBuilder()
    lhs = builder.tensor([[1], [2]], dtype="int32")
    rhs = builder.tensor([[10, 20, 30]], dtype="int32")
    return lower_to_loops(lower_to_cpu(builder.finish(lhs + rhs)))


def test_generate_c_is_deterministic_and_encodes_broadcast_offsets():
    program = _broadcast_program()

    first = generate_c(program)
    second = generate_c(program)

    assert first == second
    assert "void tiny_tensor_run(int32_t *out)" in first
    assert "for (int64_t i0 = 0; i0 < 2; ++i0)" in first
    assert "for (int64_t i1 = 0; i1 < 3; ++i1)" in first
    assert "p0[i0]" in first
    assert "p1[i1]" in first
    assert "p2[(i0 * 3) + i1]" in first


def test_generate_c_encodes_scalar_broadcast_and_relu_without_vector_intrinsics():
    builder = GraphBuilder()
    x = builder.tensor([-2.0, 1.5, 3.0], dtype="float32")
    scalar = builder.tensor(2.0, dtype="float32")
    program = lower_to_loops(lower_to_cpu(builder.finish((x + scalar).relu())))

    source = generate_c(program)

    assert "p1[0]" in source
    assert "? 0.0f :" in source
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
