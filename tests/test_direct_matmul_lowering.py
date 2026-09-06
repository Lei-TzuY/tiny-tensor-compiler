from __future__ import annotations

import numpy as np

from tiny_tensor_compiler import (
    GraphBuilder,
    compile_module,
    execute_cpu,
    execute_reference,
    fuse_elementwise,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
)
from tiny_tensor_compiler.analysis import analyze_module


def _loops(module):
    return fuse_elementwise(lower_to_loops(lower_to_cpu(module)))


def test_direct_matmul_lowering_removes_compositional_intermediate_storage() -> None:
    builder = GraphBuilder("direct_matmul")
    lhs = builder.input((2, 3), dtype="float32")
    rhs = builder.input((3, 4), dtype="float32")
    module = builder.finish(lhs @ rhs)

    # Tensor IR remains the canonical compositional semantic oracle.
    assert [op.opcode for op in module.function.ops] == [
        "input", "input", "reshape", "reshape", "mul", "sum", "return"
    ]

    cpu = lower_to_cpu(module)
    assert [kernel.opcode for kernel in cpu.instructions] == ["matmul"]
    assert len(cpu.allocations) == 3
    loops = _loops(module)
    assert [kernel.opcode for kernel in loops.kernels] == ["matmul"]
    assert loops.kernels[0].input_maps == ()

    report = analyze_module(module)
    assert report.pre_fusion_kernel_counts == (("matmul", 1),)
    assert all(slot.shape != (2, 3, 4) for slot in report.storage_slots)

    lhs_value = np.arange(6, dtype=np.float32).reshape(2, 3) - 2
    rhs_value = np.arange(12, dtype=np.float32).reshape(3, 4) / 5
    expected = execute_reference(module, [lhs_value, rhs_value])
    np.testing.assert_allclose(execute_cpu(cpu, [lhs_value, rhs_value]), expected)
    np.testing.assert_allclose(compile_module(module)(inputs=[lhs_value, rhs_value]), expected)


def test_direct_matmul_lowering_refuses_shared_intermediates() -> None:
    builder = GraphBuilder("shared_matmul_shape")
    lhs = builder.input((2, 3), dtype="float32")
    rhs = builder.input((3, 4), dtype="float32")
    lhs_expanded = lhs.reshape((2, 3, 1))
    rhs_expanded = rhs.reshape((1, 3, 4))
    products = lhs_expanded * rhs_expanded
    reduced = products.sum(axis=1)
    module = builder.finish((reduced, products))

    cpu = lower_to_cpu(module)
    assert "matmul" not in [kernel.opcode for kernel in cpu.instructions]
    assert [kernel.opcode for kernel in cpu.instructions] == [
        "reshape", "reshape", "mul", "sum"
    ]


def test_direct_matmul_codegen_preserves_k_order_and_parallelizes_outputs_only() -> None:
    builder = GraphBuilder("direct_matmul_codegen")
    lhs = builder.input((3, 4), dtype="int32")
    rhs = builder.input((4, 2), dtype="int32")
    module = builder.finish(lhs @ rhs)
    loops = _loops(module)

    source = generate_c(loops)
    assert "volatile int32_t matmul_product" in source
    assert "for (int64_t k = 0; k < 4; ++k)" in source
    assert "matmul_value" in source

    parallel_source = generate_c(loops, parallel=True)
    assert "#pragma omp parallel for schedule(static)" in parallel_source
    assert "for (i0 = 0; i0 < 3; ++i0)" in parallel_source
    assert "for (int64_t k = 0; k < 4; ++k)" in parallel_source
    assert "#pragma omp parallel for" not in parallel_source.split(
        "for (int64_t k = 0; k < 4; ++k)", 1
    )[1]
