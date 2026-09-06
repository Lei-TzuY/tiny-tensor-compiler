from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one patch anchor, found {count}: {old[:80]!r}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, content: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if marker in text:
        return
    target.write_text(text.rstrip() + "\n\n" + content.strip() + "\n", encoding="utf-8")


matmul_module = '''from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .inference import TypeInferenceError, infer_binary
from .ir import DType, Module, Operation, TensorType, Value


@dataclass(frozen=True)
class DirectMatmulPattern:
    """One exact reshape/mul/sum composition safe to contract during lowering."""

    reduction: Operation
    lhs: Value
    rhs: Value
    intermediates: tuple[Operation, ...]


def infer_direct_matmul_type(lhs: TensorType, rhs: TensorType) -> TensorType:
    """Infer the exact rank-2 matmul result used by direct physical lowering."""
    if len(lhs.shape) != 2 or len(rhs.shape) != 2:
        raise TypeInferenceError("direct matmul requires rank-2 tensor inputs")
    m, k = lhs.shape
    rhs_k, n = rhs.shape
    if k != rhs_k:
        raise TypeInferenceError(
            f"direct matmul inner dimensions must match exactly, got {k} and {rhs_k}"
        )
    try:
        dtype = DType.from_numpy(np.result_type(lhs.dtype.to_numpy(), rhs.dtype.to_numpy()))
    except TypeError as exc:
        raise TypeInferenceError(str(exc)) from exc
    return TensorType((m, n), dtype)


def find_direct_matmul_patterns(
    module: Module,
) -> tuple[dict[Operation, DirectMatmulPattern], frozenset[Operation]]:
    """Find non-overlapping compiler-owned matmul compositions with private intermediates."""
    matches: dict[Operation, DirectMatmulPattern] = {}
    skipped: set[Operation] = set()
    for op in module.function.ops:
        pattern = _match_reduction(op)
        if pattern is None:
            continue
        if any(intermediate in skipped for intermediate in pattern.intermediates):
            continue
        matches[op] = pattern
        skipped.update(pattern.intermediates)
    return matches, frozenset(skipped)


def _match_reduction(reduction: Operation) -> DirectMatmulPattern | None:
    if (
        reduction.opcode != "sum"
        or reduction.attrs != {"axis": 1}
        or len(reduction.operands) != 1
        or len(reduction.results) != 1
    ):
        return None
    product = reduction.operands[0].producer
    if (
        product is None
        or product.opcode != "mul"
        or product.attrs
        or len(product.operands) != 2
        or len(product.results) != 1
        or not _used_only_by(product.results[0], reduction, 0)
    ):
        return None

    first = product.operands[0].producer
    second = product.operands[1].producer
    if first is None or second is None or first is second:
        return None
    if first.opcode != "reshape" or second.opcode != "reshape":
        return None
    if first.attrs or second.attrs or len(first.operands) != 1 or len(second.operands) != 1:
        return None

    for lhs_reshape, rhs_reshape, lhs_position, rhs_position in (
        (first, second, 0, 1),
        (second, first, 1, 0),
    ):
        pattern = _match_orientation(
            reduction,
            product,
            lhs_reshape,
            rhs_reshape,
            lhs_position,
            rhs_position,
        )
        if pattern is not None:
            return pattern
    return None


def _match_orientation(
    reduction: Operation,
    product: Operation,
    lhs_reshape: Operation,
    rhs_reshape: Operation,
    lhs_position: int,
    rhs_position: int,
) -> DirectMatmulPattern | None:
    lhs = lhs_reshape.operands[0]
    rhs = rhs_reshape.operands[0]
    try:
        result_type = infer_direct_matmul_type(lhs.type, rhs.type)
    except TypeInferenceError:
        return None

    m, k = lhs.type.shape
    _, n = rhs.type.shape
    if lhs_reshape.results[0].type != TensorType((m, k, 1), lhs.type.dtype):
        return None
    if rhs_reshape.results[0].type != TensorType((1, k, n), rhs.type.dtype):
        return None
    if not _used_only_by(lhs_reshape.results[0], product, lhs_position):
        return None
    if not _used_only_by(rhs_reshape.results[0], product, rhs_position):
        return None

    expected_product = infer_binary(lhs_reshape.results[0].type, rhs_reshape.results[0].type)
    if product.results[0].type != expected_product or expected_product.shape != (m, k, n):
        return None
    if reduction.results[0].type != result_type:
        return None
    return DirectMatmulPattern(
        reduction=reduction,
        lhs=lhs,
        rhs=rhs,
        intermediates=(lhs_reshape, rhs_reshape, product),
    )


def _used_only_by(value: Value, user: Operation, operand_index: int) -> bool:
    return (
        len(value.uses) == 1
        and value.uses[0].user is user
        and value.uses[0].operand_index == operand_index
    )
'''
(ROOT / "src/tiny_tensor_compiler/matmul_lowering.py").write_text(matmul_module, encoding="utf-8")

# Buffer lowering: recognize the exact compositional pattern and skip private intermediates.
replace_once(
    "src/tiny_tensor_compiler/lowering.py",
    "from .layout import StorageLayout, element_count\nfrom .reduction import REDUCTION_OPCODES, ReductionPlan\n",
    "from .layout import StorageLayout, element_count\nfrom .matmul_lowering import find_direct_matmul_patterns, infer_direct_matmul_type\nfrom .reduction import REDUCTION_OPCODES, ReductionPlan\n",
)
replace_once(
    "src/tiny_tensor_compiler/lowering.py",
    "def lower_to_cpu(module: Module) -> CPUProgram:\n    verify(module)\n    buffers: dict[Value, int] = {}\n",
    "def lower_to_cpu(module: Module) -> CPUProgram:\n    verify(module)\n    direct_matmuls, skipped_matmul_ops = find_direct_matmul_patterns(module)\n    buffers: dict[Value, int] = {}\n",
)
replace_once(
    "src/tiny_tensor_compiler/lowering.py",
    "    for op in module.function.ops:\n        if op.opcode == \"return\":\n",
    "    for op in module.function.ops:\n        if op in skipped_matmul_ops:\n            continue\n        if op.opcode == \"return\":\n",
)
replace_once(
    "src/tiny_tensor_compiler/lowering.py",
    "        literal = None\n        if op.opcode == \"const\":\n",
    "        direct_matmul = direct_matmuls.get(op)\n        if direct_matmul is not None:\n            operations.append(\n                BufferKernel(\n                    opcode=\"matmul\",\n                    output=buffer,\n                    inputs=(buffers[direct_matmul.lhs], buffers[direct_matmul.rhs]),\n                )\n            )\n            continue\n\n        literal = None\n        if op.opcode == \"const\":\n",
)
replace_once(
    "src/tiny_tensor_compiler/lowering.py",
    "            elif op.opcode == \"relu\":\n                if len(op.inputs) != 1 or op.literal is not None:\n",
    "            elif op.opcode == \"matmul\":\n                if (\n                    len(op.inputs) != 2\n                    or op.literal is not None\n                    or op.reduction_axis is not None\n                ):\n                    raise ValueError(\n                        \"matmul kernel requires two inputs, no literal, and no reduction axis\"\n                    )\n                expected = infer_direct_matmul_type(\n                    allocated[op.inputs[0]], allocated[op.inputs[1]]\n                )\n                if expected != output_type:\n                    raise ValueError(\n                        \"matmul kernel output buffer type does not match inference\"\n                    )\n            elif op.opcode == \"relu\":\n                if len(op.inputs) != 1 or op.literal is not None:\n",
)

# Loop IR: matmul owns contraction indexing, so it has no broadcast IndexMap.
replace_once(
    "src/tiny_tensor_compiler/loop_ir.py",
    "from .layout import StorageLayout, element_count\nfrom .lowering import (\n",
    "from .layout import StorageLayout, element_count\nfrom .matmul_lowering import infer_direct_matmul_type\nfrom .lowering import (\n",
)
replace_once(
    "src/tiny_tensor_compiler/loop_ir.py",
    "            if op.opcode == \"reshape\" or op.reduction is not None\n",
    "            if op.opcode in {\"reshape\", \"matmul\"} or op.reduction is not None\n",
)
replace_once(
    "src/tiny_tensor_compiler/loop_ir.py",
    "            elif op.opcode == \"reshape\":\n                rhs = f\"reshape p{op.inputs[0]}[linear]\"\n            elif op.reduction is not None:\n",
    "            elif op.opcode == \"reshape\":\n                rhs = f\"reshape p{op.inputs[0]}[linear]\"\n            elif op.opcode == \"matmul\":\n                rhs = f\"matmul p{op.inputs[0]}, p{op.inputs[1]} [k-order]\"\n            elif op.reduction is not None:\n",
)
replace_once(
    "src/tiny_tensor_compiler/loop_ir.py",
    "            elif op.opcode in {\"add\", \"mul\"}:\n                if len(op.inputs) != 2 or len(op.input_maps) != 2 or op.literal is not None:\n",
    "            elif op.opcode == \"matmul\":\n                if (\n                    len(op.inputs) != 2\n                    or op.input_maps\n                    or op.literal is not None\n                    or op.reduction_axis is not None\n                ):\n                    raise ValueError(\n                        \"matmul loop requires two inputs, no index maps, no literal, and no reduction axis\"\n                    )\n                expected = infer_direct_matmul_type(types[op.inputs[0]], types[op.inputs[1]])\n                if expected != output_type:\n                    raise ValueError(\n                        \"matmul loop output buffer type does not match inference\"\n                    )\n            elif op.opcode in {\"add\", \"mul\"}:\n                if len(op.inputs) != 2 or len(op.input_maps) != 2 or op.literal is not None:\n",
)

# Explicit CPU execution: preserve product and accumulator dtype boundaries in K order.
replace_once(
    "src/tiny_tensor_compiler/backends/cpu.py",
    "        output = buffers[op.output]\n        reduction = op.reduction\n",
    "        output = buffers[op.output]\n        if op.opcode == \"matmul\":\n            _execute_direct_matmul(buffers[op.inputs[0]], buffers[op.inputs[1]], output)\n            continue\n        reduction = op.reduction\n",
)
replace_once(
    "src/tiny_tensor_compiler/backends/cpu.py",
    "\ndef _execute_argmax_reduction(\n",
    '''\ndef _execute_direct_matmul(lhs: np.ndarray, rhs: np.ndarray, output: np.ndarray) -> None:\n    dtype = output.dtype\n    m, k_extent = lhs.shape\n    n = rhs.shape[1]\n    for i in range(m):\n        for j in range(n):\n            accumulator = dtype.type(0)\n            for k in range(k_extent):\n                product = dtype.type(np.multiply(lhs[i, k], rhs[k, j]))\n                accumulator = dtype.type(np.add(accumulator, product))\n            output[i, j] = accumulator\n\n\ndef _execute_argmax_reduction(\n''',
)

# Generated C: emit one direct contraction, leaving K serial and therefore deterministic.
replace_once(
    "src/tiny_tensor_compiler/c_codegen.py",
    "    reduction = op.reduction\n    if reduction is not None:\n",
    "    if op.opcode == \"matmul\":\n        return _emit_direct_matmul(op, types, layouts, lines)\n\n    reduction = op.reduction\n    if reduction is not None:\n",
)
replace_once(
    "src/tiny_tensor_compiler/c_codegen.py",
    "\ndef _emit_argmax_reduction(\n",
    '''\ndef _emit_direct_matmul(\n    op: LoopKernel,\n    types: dict[int, TensorType],\n    layouts: dict[int, StorageLayout],\n    lines: list[str],\n) -> list[str]:\n    if len(op.inputs) != 2:\n        raise RuntimeError("verified matmul loop unexpectedly has invalid arity")\n    lhs, rhs = op.inputs\n    lhs_type = types[lhs]\n    rhs_type = types[rhs]\n    output_type = types[op.output]\n    if len(lhs_type.shape) != 2 or len(rhs_type.shape) != 2 or len(output_type.shape) != 2:\n        raise RuntimeError("verified matmul loop unexpectedly has non-rank-2 tensors")\n\n    m, k_extent = lhs_type.shape\n    n = rhs_type.shape[1]\n    c_type = _c_type(output_type.dtype)\n    zero = _zero_literal(output_type.dtype)\n    lhs_ref = _matmul_input_ref(lhs, layouts[lhs], "i0", "k")\n    rhs_ref = _matmul_input_ref(rhs, layouts[rhs], "k", "i1")\n    lines.extend(\n        [\n            f"        for (int64_t i0 = 0; i0 < {m}; ++i0) {{",\n            f"            for (int64_t i1 = 0; i1 < {n}; ++i1) {{",\n            f"                {c_type} matmul_value = {zero};",\n            f"                for (int64_t k = 0; k < {k_extent}; ++k) {{",\n            f"                    volatile {c_type} matmul_product = "\n            f"(({c_type}){lhs_ref} * ({c_type}){rhs_ref});",\n            f"                    matmul_value = "\n            f"(({c_type})matmul_value + ({c_type})matmul_product);",\n            "                }",\n            f"                p{op.output}[(i0 * {n}) + i1] = matmul_value;",\n            "            }",\n            "        }",\n            "    }",\n            "",\n        ]\n    )\n    return lines\n\n\ndef _matmul_input_ref(\n    buffer: int,\n    layout: StorageLayout,\n    first_index: str,\n    second_index: str,\n) -> str:\n    terms: list[str] = []\n    for index, stride in zip((first_index, second_index), layout.strides, strict=True):\n        if stride == 1:\n            terms.append(index)\n        elif stride == -1:\n            terms.append(f"(-{index})")\n        else:\n            terms.append(f"({index} * {stride})")\n    offset = " + ".join(terms) if terms else "0"\n    return f"p{buffer}[{offset}]"\n\n\ndef _emit_argmax_reduction(\n''',
)

# Existing matmul integration test now observes direct physical lowering while Tensor IR stays compositional.
replace_once(
    "tests/test_matmul.py",
    'def test_matmul_expansion_remains_a_fusion_boundary_and_runs_openmp_native() -> None:\n',
    'def test_matmul_direct_lowering_remains_a_fusion_boundary_and_runs_openmp_native() -> None:\n',
)
replace_once(
    "tests/test_matmul.py",
    '    assert [kernel.opcode for kernel in loops.kernels] == ["reshape", "reshape", "mul", "sum", "relu"]\n',
    '    assert [kernel.opcode for kernel in loops.kernels] == ["matmul", "relu"]\n',
)
replace_once(
    "tests/test_matmul.py",
    '    assert "for (int64_t r = 0; r < 4; ++r)" in source\n',
    '    assert "for (int64_t k = 0; k < 4; ++k)" in source\n',
)

new_tests = '''from __future__ import annotations

import numpy as np

from tiny_tensor_compiler import (
    GraphBuilder,
    analyze_module,
    compile_module,
    execute_cpu,
    execute_reference,
    fuse_elementwise,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
)


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
'''
(ROOT / "tests/test_direct_matmul_lowering.py").write_text(new_tests, encoding="utf-8")

append_once(
    "docs/matmul.md",
    "## Direct physical lowering milestone",
    '''## Direct physical lowering milestone

After the compositional rank-2 semantic surface was established, the physical lowering
pipeline learned one conservative contraction optimization. The exact private
`reshape -> reshape -> mul -> sum(axis=1)` shape emitted by `Tensor.matmul()` is recognized
only when both reshape results and the product are single-use intermediates. Buffer/Loop IR
then contains one `matmul` kernel over the original `(M,K)` and `(K,N)` logical values, so
compiler-owned `(M,K,N)` product storage is not materialized.

The tensor IR deliberately remains compositional and serializable; reference execution is
therefore an independent oracle for the direct physical kernel. The direct kernel preserves
left-to-right `k=0..K-1` accumulation, casts each product and accumulator update through the
promoted output dtype, and returns additive identity for `K=0`. Generated C uses the same
ordered contraction. OpenMP may schedule independent output rows, but the `K` reduction is
never parallelized or reassociated. Logical transpose/reverse/slice layouts are indexed
through their verified strides without forcing a copy.

This is a storage-elimination and executable lowering claim, not a GEMM performance claim.
BLAS dispatch, tiling, vector-dot SIMD, batched matmul, transpose flags, and parallel K
reductions remain separate future work.''',
)

replace_once(
    "CHANGELOG.md",
    "## [Unreleased]\n\n### Added\n\n",
    "## [Unreleased]\n\n### Added\n\n- Direct verified physical lowering for the canonical rank-2 matmul composition, replacing private reshape/multiply/sum intermediates with one deterministic contraction kernel while retaining compositional tensor IR as the reference/serialization surface.\n",
)

print("direct matmul lowering patch applied")
