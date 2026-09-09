import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    compile_module,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
)
from tiny_tensor_compiler.ir import DType, TensorType
from tiny_tensor_compiler.layout import StorageLayout
from tiny_tensor_compiler.loop_ir import IndexMap, LoopCopyInto
from tiny_tensor_compiler.parallel_codegen import emit_parallel_copy_into

_OPENMP_PARALLEL_FOR = "#pragma omp parallel for schedule(static)"


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _broadcast_copy_module():
    builder = GraphBuilder()
    base = builder.input((2, 6), dtype="int32")
    source = builder.input((3,), dtype="int32")
    root = base.relu()
    target = root.slice(axis=1, start=0, stop=6, step=2)
    return builder.finish(root.copy_into(target, source))


def test_parallel_codegen_schedules_injective_broadcast_copy_effect():
    loops = lower_to_loops(lower_to_cpu(_broadcast_copy_module()))
    effect = loops.copies[0]

    serial = generate_c(loops, parallel=False)
    parallel = generate_c(loops, parallel=True)

    assert _OPENMP_PARALLEL_FOR not in serial
    assert parallel.count(_OPENMP_PARALLEL_FOR) == 1
    assert "int64_t i0;" in parallel
    assert "for (i0 = 0;" in parallel
    assert f"p{effect.source}[i1]" in parallel


def test_parallel_copy_effect_falls_back_for_scalar_and_zero_extent_targets():
    scalar = TensorType((), DType.INT32)
    scalar_op = LoopCopyInto(
        output=3,
        root=0,
        target=1,
        source=2,
        type=scalar,
        layout=StorageLayout.contiguous(()),
    )
    scalar_lines = emit_parallel_copy_into(
        scalar_op,
        {0: scalar, 1: scalar, 2: scalar, 3: scalar},
        {
            0: StorageLayout.contiguous(()),
            1: StorageLayout.contiguous(()),
            2: StorageLayout.contiguous(()),
            3: StorageLayout.contiguous(()),
        },
    )
    assert _OPENMP_PARALLEL_FOR not in "\n".join(scalar_lines)

    zero_target = TensorType((0, 3), DType.INT32)
    zero_source = TensorType((3,), DType.INT32)
    zero_op = LoopCopyInto(
        output=3,
        root=0,
        target=1,
        source=2,
        type=zero_target,
        layout=StorageLayout.contiguous((0, 3)),
        source_map=IndexMap((1,)),
    )
    zero_lines = emit_parallel_copy_into(
        zero_op,
        {0: zero_target, 1: zero_target, 2: zero_source, 3: zero_target},
        {
            0: StorageLayout.contiguous((0, 3)),
            1: StorageLayout.contiguous((0, 3)),
            2: StorageLayout.contiguous((3,)),
            3: StorageLayout.contiguous((0, 3)),
        },
    )
    assert _OPENMP_PARALLEL_FOR not in "\n".join(zero_lines)


def test_parallel_copy_effect_falls_back_for_overlapping_target_layout():
    root_type = TensorType((4,), DType.INT32)
    target_type = TensorType((2, 2), DType.INT32)
    source_type = TensorType((2, 2), DType.INT32)
    overlapping = StorageLayout(offset=0, strides=(1, 1))
    op = LoopCopyInto(
        output=3,
        root=0,
        target=1,
        source=2,
        type=root_type,
        layout=StorageLayout.contiguous((4,)),
    )

    lines = emit_parallel_copy_into(
        op,
        {0: root_type, 1: target_type, 2: source_type, 3: root_type},
        {
            0: StorageLayout.contiguous((4,)),
            1: overlapping,
            2: StorageLayout.contiguous((2, 2)),
            3: StorageLayout.contiguous((4,)),
        },
    )

    assert _OPENMP_PARALLEL_FOR not in "\n".join(lines)


def test_parallel_copy_native_preserves_broadcast_and_signed_target_layout():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    base = builder.input((2, 3), dtype="float32")
    source = builder.input((3,), dtype="float32")
    root = base.relu()
    target = root.reverse(axis=1)
    module = builder.finish(root.copy_into(target, source))

    base_value = np.array([[-2.0, 1.0, 4.0], [3.0, -1.0, 5.0]], dtype=np.float32)
    source_value = np.array([7.0, 11.0, 13.0], dtype=np.float32)
    actual = compile_module(module, borrow_inputs=True, parallel=True)(
        inputs=[base_value, source_value]
    )

    expected = np.maximum(base_value, 0)
    expected[:, ::-1] = source_value
    np.testing.assert_array_equal(actual, expected)


def test_parallel_copy_keeps_borrowed_source_unmodified():
    _default_compiler_or_skip()
    module = _broadcast_copy_module()
    base = np.arange(12, dtype=np.int32).reshape(2, 6) - 4
    source = np.array([17, 19, 23], dtype=np.int32)
    source_before = source.copy()

    actual = compile_module(module, borrow_inputs=True, parallel=True)(inputs=[base, source])

    expected = np.maximum(base, 0)
    expected[:, 0:6:2] = source
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, source_before)
