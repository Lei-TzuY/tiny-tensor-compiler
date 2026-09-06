from __future__ import annotations

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    SymbolicDim,
    TypeInferenceError,
    common_subexpression_eliminate,
    compile_dynamic_module,
    compile_module,
    dead_code_eliminate,
    execute_cpu,
    execute_reference,
    fuse_elementwise,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
)
from tiny_tensor_compiler.serialization import deserialize_module, serialize_module


def _loops(module):
    return fuse_elementwise(lower_to_loops(lower_to_cpu(module)))


def _manual_i32_matmul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    out = np.empty((lhs.shape[0], rhs.shape[1]), dtype=np.int32)
    for i in range(lhs.shape[0]):
        for j in range(rhs.shape[1]):
            acc = np.int32(0)
            for k in range(lhs.shape[1]):
                product = np.int32(np.multiply(lhs[i, k], rhs[k, j]))
                acc = np.int32(np.add(acc, product))
            out[i, j] = acc
    return out


def test_rank2_matmul_builds_verified_primitives_and_executes_end_to_end() -> None:
    builder = GraphBuilder()
    lhs = builder.input((2, 3), dtype="float32")
    rhs = builder.input((3, 4), dtype="float32")
    result = lhs @ rhs
    module = builder.finish(result)

    assert result.type.shape == (2, 4)
    assert [op.opcode for op in module.function.ops] == [
        "input",
        "input",
        "reshape",
        "reshape",
        "mul",
        "sum",
        "return",
    ]
    assert module.function.ops[-2].attrs == {"axis": 1}

    lhs_value = np.arange(6, dtype=np.float32).reshape(2, 3) - 2
    rhs_value = np.arange(12, dtype=np.float32).reshape(3, 4) / 3
    expected = lhs_value @ rhs_value

    np.testing.assert_allclose(execute_reference(module, [lhs_value, rhs_value]), expected)
    np.testing.assert_allclose(execute_cpu(lower_to_cpu(module), [lhs_value, rhs_value]), expected)
    np.testing.assert_allclose(compile_module(module)(inputs=[lhs_value, rhs_value]), expected)


def test_matmul_uses_result_type_promotion_and_fixed_width_integer_steps() -> None:
    builder = GraphBuilder()
    lhs = builder.input((1, 3), dtype="int32")
    rhs = builder.input((3, 1), dtype="int32")
    result = lhs @ rhs
    module = builder.finish(result)

    lhs_value = np.array([[2_000_000_000, 2_000_000_000, -2_000_000_000]], dtype=np.int32)
    rhs_value = np.array([[2], [3], [4]], dtype=np.int32)
    expected = _manual_i32_matmul(lhs_value, rhs_value)
    np.testing.assert_array_equal(execute_reference(module, [lhs_value, rhs_value]), expected)
    np.testing.assert_array_equal(compile_module(module)(inputs=[lhs_value, rhs_value]), expected)

    promoted_builder = GraphBuilder()
    a = promoted_builder.input((2, 2), dtype="float32")
    b = promoted_builder.input((2, 2), dtype="int32")
    promoted = a @ b
    assert promoted.type.dtype.to_numpy() == np.dtype("float64")


def test_matmul_rejects_non_rank2_and_mismatched_inner_dimensions() -> None:
    builder = GraphBuilder()
    vector = builder.input((3,), dtype="float32")
    matrix = builder.input((3, 2), dtype="float32")
    with pytest.raises(TypeInferenceError, match="rank-2"):
        _ = vector @ matrix

    builder2 = GraphBuilder()
    lhs = builder2.input((2, 3), dtype="float32")
    rhs = builder2.input((4, 2), dtype="float32")
    with pytest.raises(TypeInferenceError, match="inner dimensions"):
        _ = lhs @ rhs


def test_zero_contract_extent_returns_additive_identity() -> None:
    builder = GraphBuilder()
    lhs = builder.input((2, 0), dtype="float64")
    rhs = builder.input((0, 3), dtype="float64")
    module = builder.finish(lhs @ rhs)

    lhs_value = np.empty((2, 0), dtype=np.float64)
    rhs_value = np.empty((0, 3), dtype=np.float64)
    expected = np.zeros((2, 3), dtype=np.float64)
    np.testing.assert_array_equal(execute_reference(module, [lhs_value, rhs_value]), expected)
    np.testing.assert_array_equal(compile_module(module)(inputs=[lhs_value, rhs_value]), expected)


def test_matmul_consumes_transposed_and_reversed_logical_views() -> None:
    builder = GraphBuilder()
    lhs_root = builder.input((3, 2), dtype="float32")
    rhs_root = builder.input((2, 3), dtype="float32")
    lhs = lhs_root.transpose((1, 0))
    rhs = rhs_root.transpose((1, 0)).reverse(0)
    module = builder.finish(lhs @ rhs)

    lhs_value = np.arange(6, dtype=np.float32).reshape(3, 2)
    rhs_value = (np.arange(6, dtype=np.float32) + 1).reshape(2, 3)
    expected = lhs_value.T @ np.flip(rhs_value.T, axis=0)
    np.testing.assert_allclose(execute_cpu(lower_to_cpu(module), [lhs_value, rhs_value]), expected)
    np.testing.assert_allclose(compile_module(module)(inputs=[lhs_value, rhs_value]), expected)


def test_symbolic_matmul_specializes_and_reuses_complete_bindings() -> None:
    b = SymbolicDim("B")
    k = SymbolicDim("K")
    n = SymbolicDim("N")
    builder = GraphBuilder()
    lhs = builder.input((b, k), dtype="float32")
    rhs = builder.input((k, n), dtype="float32")
    module = builder.finish(lhs @ rhs)
    executable = compile_dynamic_module(module, borrow_inputs=True)

    lhs1 = np.arange(6, dtype=np.float32).reshape(2, 3)
    rhs1 = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_allclose(executable(inputs=[lhs1, rhs1]), lhs1 @ rhs1)
    np.testing.assert_allclose(executable(inputs=[lhs1 + 1, rhs1]), (lhs1 + 1) @ rhs1)
    assert len(executable.cached_bindings) == 1

    lhs2 = np.arange(8, dtype=np.float32).reshape(2, 4)
    rhs2 = np.arange(8, dtype=np.float32).reshape(4, 2)
    np.testing.assert_allclose(executable(inputs=[lhs2, rhs2]), lhs2 @ rhs2)
    assert len(executable.cached_bindings) == 2


def test_matmul_direct_lowering_remains_a_fusion_boundary_and_runs_openmp_native() -> None:
    builder = GraphBuilder()
    lhs = builder.input((3, 4), dtype="float32")
    rhs = builder.input((4, 2), dtype="float32")
    module = builder.finish((lhs @ rhs).relu())
    loops = _loops(module)

    assert [kernel.opcode for kernel in loops.kernels] == ["matmul", "relu"]
    source = generate_c(loops, parallel=True)
    assert "#pragma omp parallel for schedule(static)" in source
    assert "for (i0 = 0; i0 < 3; ++i0)" in source
    assert "for (int64_t k = 0; k < 4; ++k)" in source

    lhs_value = np.arange(12, dtype=np.float32).reshape(3, 4) - 6
    rhs_value = np.arange(8, dtype=np.float32).reshape(4, 2) - 2
    expected = np.maximum(lhs_value @ rhs_value, 0)
    np.testing.assert_allclose(
        compile_module(module, parallel=True)(inputs=[lhs_value, rhs_value]),
        expected,
    )


def test_matmul_composition_participates_in_existing_dce_and_cse() -> None:
    builder = GraphBuilder()
    lhs = builder.input((2, 3), dtype="float32")
    rhs = builder.input((3, 2), dtype="float32")
    _ = lhs @ rhs
    live = lhs @ rhs
    module = builder.finish(live.relu())
    assert dead_code_eliminate(module) == 4

    builder2 = GraphBuilder()
    lhs2 = builder2.input((2, 3), dtype="float32")
    rhs2 = builder2.input((3, 2), dtype="float32")
    first = lhs2 @ rhs2
    second = lhs2 @ rhs2
    module2 = builder2.finish(first + second)
    assert common_subexpression_eliminate(module2) == 4
    assert [op.opcode for op in module2.function.ops].count("sum") == 1


def test_matmul_expansion_round_trips_through_canonical_serialization() -> None:
    builder = GraphBuilder("matmul_roundtrip")
    lhs = builder.input((2, 3), dtype="float32")
    rhs = builder.input((3, 4), dtype="float32")
    module = builder.finish(lhs @ rhs)
    encoded = serialize_module(module)
    decoded = deserialize_module(encoded)
    assert serialize_module(decoded) == encoded

    lhs_value = np.arange(6, dtype=np.float32).reshape(2, 3)
    rhs_value = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_allclose(execute_reference(decoded, [lhs_value, rhs_value]), lhs_value @ rhs_value)
