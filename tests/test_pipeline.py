import numpy as np

from tiny_tensor_compiler import (
    GraphBuilder,
    constant_fold,
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
