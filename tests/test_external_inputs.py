import os
import shutil

import numpy as np
import pytest

import tiny_tensor_compiler.native as native_module
from tiny_tensor_compiler import (
    GraphBuilder,
    execute_cpu,
    execute_loop,
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


def test_external_inputs_are_explicit_and_execute_across_reference_and_loop_backends():
    builder = GraphBuilder()
    lhs = builder.input((2, 1), dtype="float32")
    rhs = builder.input((1, 3), dtype="float32")
    module = builder.finish((lhs + rhs).relu())

    assert module.dump() == """func @main() {
  %0 = input 0 : tensor<2x1xf32>
  %1 = input 1 : tensor<1x3xf32>
  %2 = add %0, %1 : tensor<2x3xf32>
  %3 = relu %2 : tensor<2x3xf32>
  return %3
}"""

    lhs_value = np.array([[-3.0], [2.0]], dtype=np.float32)
    rhs_value = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    inputs = [lhs_value, rhs_value]
    expected = np.maximum(lhs_value + rhs_value, np.float32(0.0))

    cpu = lower_to_cpu(module)
    assert "b0 = input 0" in cpu.dump()
    assert "b1 = input 1" in cpu.dump()
    loops = lower_to_loops(cpu)
    assert "input 0" in loops.dump()
    assert "input 1" in loops.dump()

    np.testing.assert_array_equal(execute_reference(module, inputs=inputs), expected)
    np.testing.assert_array_equal(execute_cpu(cpu, inputs=inputs), expected)
    np.testing.assert_array_equal(execute_loop(loops, inputs=inputs), expected)


def test_external_input_runtime_requires_exact_count_shape_and_dtype():
    builder = GraphBuilder()
    value = builder.input((2,), dtype="int32")
    module = builder.finish(value.relu())

    with pytest.raises(ValueError, match="expected 1 runtime inputs, got 0"):
        execute_reference(module)
    with pytest.raises(ValueError, match="input 0 shape"):
        execute_reference(module, inputs=[np.array([[1, 2]], dtype=np.int32)])
    with pytest.raises(ValueError, match="input 0 dtype"):
        execute_reference(module, inputs=[np.array([1, 2], dtype=np.int64)])


def test_generated_c_extends_output_pointer_abi_with_typed_input_pointers():
    builder = GraphBuilder()
    lhs = builder.input((2,), dtype="float32")
    rhs = builder.input((2,), dtype="int32")
    module = builder.finish(lhs + rhs)
    loops = lower_to_loops(lower_to_cpu(module))

    source = generate_c(loops)

    assert "TINY_TENSOR_EXPORT void tiny_tensor_run(float *out, const float *input0, const int32_t *input1)" in source
    assert "p0[i0] = input0[i0];" in source
    assert "p1[i0] = input1[i0];" in source


def test_native_compiled_graph_reuses_artifact_for_different_runtime_input_values(monkeypatch):
    _default_compiler_or_skip()
    native_module.clear_native_cache()
    builder = GraphBuilder()
    value = builder.input((3,), dtype="int32")
    module = builder.finish((value * 2 + 1).relu())
    loops = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))

    compile_calls = 0
    original_run = native_module.subprocess.run

    def counting_run(*args, **kwargs):
        nonlocal compile_calls
        compile_calls += 1
        return original_run(*args, **kwargs)

    monkeypatch.setattr(native_module.subprocess, "run", counting_run)

    first_input = np.array([-2, 0, 3], dtype=np.int32)
    second_input = np.array([4, -1, 7], dtype=np.int32)
    first = execute_native(loops, inputs=[first_input])
    second = execute_native(loops, inputs=[second_input])

    assert compile_calls == 1
    np.testing.assert_array_equal(first, np.maximum(first_input * 2 + 1, 0))
    np.testing.assert_array_equal(second, np.maximum(second_input * 2 + 1, 0))
