import numpy as np

from tiny_tensor_compiler import (
    GraphBuilder,
    algebraic_simplify,
    constant_fold,
    dead_code_eliminate,
    execute_cpu,
    execute_reference,
    lower_to_cpu,
    verify,
)


def build_example():
    builder = GraphBuilder()
    x = builder.tensor([1, 2, 3])
    z = (x * 2 + 1).relu()
    return builder.finish(z)


def test_end_to_end_pipeline_matches_numpy():
    module = build_example()
    verify(module)

    expected = np.maximum(np.array([1, 2, 3]) * 2 + 1, 0)
    np.testing.assert_array_equal(execute_reference(module), expected)
    np.testing.assert_array_equal(execute_cpu(lower_to_cpu(module)), expected)


def test_ir_dump_is_explicit_and_deterministic():
    module = build_example()
    expected = """func @main() {
  %0 = const [1, 2, 3] : tensor<3xi64>
  %1 = const 2 : tensor<i64>
  %2 = mul %0, %1 : tensor<3xi64>
  %3 = const 1 : tensor<i64>
  %4 = add %2, %3 : tensor<3xi64>
  %5 = relu %4 : tensor<3xi64>
  return %5
}"""
    assert module.dump() == expected
    assert module.dump() == expected


def test_broadcasting_is_inferred_and_executed():
    builder = GraphBuilder()
    lhs = builder.tensor([[1.0], [2.0]], dtype="float32")
    rhs = builder.tensor([[10.0, 20.0, 30.0]], dtype="float32")
    result = lhs + rhs
    assert result.type.shape == (2, 3)
    assert result.type.dtype.value == "f32"
    module = builder.finish(result)

    expected = np.array([[11.0, 21.0, 31.0], [12.0, 22.0, 32.0]], dtype=np.float32)
    np.testing.assert_array_equal(execute_cpu(lower_to_cpu(module)), expected)


def test_constant_folding_preserves_semantics():
    module = build_example()
    before = execute_reference(module)

    assert constant_fold(module) == 3
    verify(module)
    after = execute_cpu(lower_to_cpu(module))

    np.testing.assert_array_equal(after, before)
    assert all(op.opcode not in {"add", "mul", "relu"} for op in module.function.ops)


def test_algebraic_simplification_removes_integer_neutral_elements():
    builder = GraphBuilder()
    x = builder.tensor([1, -2, 3], dtype="int32")
    result = 1 * (((0 + x) * 1) + 0)
    module = builder.finish(result)
    before = execute_reference(module)

    assert algebraic_simplify(module) == 4
    verify(module)
    np.testing.assert_array_equal(execute_cpu(lower_to_cpu(module)), before)
    assert all(op.opcode not in {"add", "mul"} for op in module.function.ops)


def test_algebraic_simplification_preserves_promotion_and_broadcasting():
    builder = GraphBuilder()
    x = builder.tensor([[1], [2]], dtype="int32")
    promoted_zero = builder.tensor([[0, 0, 0]], dtype="int64")
    module = builder.finish(x + promoted_zero)
    before_dump = module.dump()

    assert algebraic_simplify(module) == 0
    verify(module)
    assert module.dump() == before_dump
    np.testing.assert_array_equal(
        execute_reference(module), np.array([[1, 1, 1], [2, 2, 2]], dtype=np.int64)
    )


def test_algebraic_simplification_is_conservative_for_floats():
    builder = GraphBuilder()
    x = builder.tensor([-0.0, 2.0], dtype="float32")
    module = builder.finish((x + 0.0) * 1.0)
    before_dump = module.dump()

    assert algebraic_simplify(module) == 0
    verify(module)
    assert module.dump() == before_dump


def test_dead_code_elimination_removes_cascading_unused_chain():
    builder = GraphBuilder()
    live = builder.tensor([1, 2, 3], dtype="int32")
    dead = builder.tensor([10, 20, 30], dtype="int32")
    _unused = (dead * 2 + 1).relu()
    module = builder.finish(live)
    before = execute_reference(module)

    assert dead_code_eliminate(module) == 6
    verify(module)
    np.testing.assert_array_equal(execute_cpu(lower_to_cpu(module)), before)
    assert [op.opcode for op in module.function.ops] == ["const", "return"]


def test_dead_code_elimination_preserves_live_chain():
    module = build_example()
    before_dump = module.dump()
    before = execute_reference(module)

    assert dead_code_eliminate(module) == 0
    verify(module)
    assert module.dump() == before_dump
    np.testing.assert_array_equal(execute_cpu(lower_to_cpu(module)), before)


def test_dead_code_elimination_cleans_simplification_residue():
    builder = GraphBuilder()
    x = builder.tensor([1, 2, 3], dtype="int32")
    module = builder.finish(x + 0)

    assert algebraic_simplify(module) == 1
    assert dead_code_eliminate(module) == 1
    verify(module)
    assert [op.opcode for op in module.function.ops] == ["const", "return"]


def test_randomized_reference_and_lowered_cpu_agree():
    rng = np.random.default_rng(42)
    for _ in range(25):
        x_data = rng.normal(size=(3, 4)).astype(np.float32)
        scale_data = rng.normal(size=(4,)).astype(np.float32)
        bias_data = rng.normal(size=(1, 4)).astype(np.float32)

        builder = GraphBuilder()
        x = builder.tensor(x_data)
        scale = builder.tensor(scale_data)
        bias = builder.tensor(bias_data)
        module = builder.finish((x * scale + bias).relu())

        expected = np.maximum(x_data * scale_data + bias_data, 0)
        np.testing.assert_allclose(execute_reference(module), expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(
            execute_cpu(lower_to_cpu(module)), expected, rtol=1e-6, atol=1e-6
        )
