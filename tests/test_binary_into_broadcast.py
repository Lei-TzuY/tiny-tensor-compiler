import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    SymbolicDim,
    compile_dynamic_module,
    compile_module,
    execute_reference,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
)


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _broadcast_binary_into_module(operator: str = "add"):
    builder = GraphBuilder()
    base = builder.input((2, 6), dtype="int32")
    source = builder.input((3,), dtype="int32")
    root = base.relu()
    target = root.slice(axis=1, start=0, stop=6, step=2)
    updated = root.binary_into(target, source, operator=operator)
    return builder.finish(updated)


@pytest.mark.parametrize("operator", ["add", "mul"])
def test_binary_into_broadcasts_lower_rank_rhs_in_reference_execution(operator: str):
    module = _broadcast_binary_into_module(operator)
    base = np.arange(12, dtype=np.int32).reshape(2, 6) - 3
    source = np.array([10, 20, 30], dtype=np.int32)

    actual = execute_reference(module, inputs=[base, source])
    expected = np.maximum(base, 0)
    target = expected[:, 0:6:2]
    if operator == "add":
        target[...] = target + source
    else:
        target[...] = target * source

    np.testing.assert_array_equal(actual, expected)


def test_binary_into_broadcasts_scalar_rhs_into_strided_target():
    builder = GraphBuilder()
    base = builder.input((2, 6), dtype="float32")
    source = builder.input((), dtype="float32")
    root = base.relu()
    target = root.slice(axis=1, start=1, stop=6, step=2)
    module = builder.finish(root.add_into(target, source))

    base_value = np.arange(12, dtype=np.float32).reshape(2, 6) - 4
    source_value = np.array(2.5, dtype=np.float32)
    actual = execute_reference(module, inputs=[base_value, source_value])
    expected = np.maximum(base_value, 0)
    expected[:, 1:6:2] += source_value
    np.testing.assert_array_equal(actual, expected)


def test_binary_into_broadcast_native_parallel_and_generated_c_use_rhs_map():
    _default_compiler_or_skip()
    module = _broadcast_binary_into_module("add")
    loops = lower_to_loops(lower_to_cpu(module))
    effect = loops.binary_intos[0]

    assert effect.source_map is not None
    assert effect.source_map.axes == (1,)

    generated = generate_c(loops, parallel=True)
    assert "#pragma omp parallel for schedule(static)" in generated
    assert f"p{effect.source}[i1]" in generated

    base = np.arange(12, dtype=np.int32).reshape(2, 6) - 3
    source = np.array([7, 11, 13], dtype=np.int32)
    actual = compile_module(module, borrow_inputs=True, parallel=True)(inputs=[base, source])
    expected = np.maximum(base, 0)
    expected[:, 0:6:2] += source
    np.testing.assert_array_equal(actual, expected)


def test_binary_into_broadcast_dynamic_specialization_reuses_binding_cache():
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    builder = GraphBuilder()
    base = builder.input((batch, 6), dtype="int32")
    source = builder.input((3,), dtype="int32")
    root = base.relu()
    target = root.slice(axis=1, start=0, stop=6, step=2)
    module = builder.finish(root.add_into(target, source))
    executable = compile_dynamic_module(module, borrow_inputs=True, parallel=True)

    source_value = np.array([3, 5, 7], dtype=np.int32)
    for size in (2, 0, 4, 2):
        base_value = np.arange(size * 6, dtype=np.int32).reshape(size, 6) - 2
        actual = executable(inputs=[base_value, source_value])
        expected = np.maximum(base_value, 0)
        expected[:, 0:6:2] += source_value
        np.testing.assert_array_equal(actual, expected)

    assert executable.cached_batch_sizes == (0, 2, 4)


def test_binary_into_broadcast_rejects_dtype_change_and_nonbroadcastable_rhs():
    builder = GraphBuilder()
    base = builder.input((2, 6), dtype="int32")
    root = base.relu()
    target = root.slice(axis=1, start=0, stop=6, step=2)
    wrong_dtype = builder.input((3,), dtype="int64")
    with pytest.raises(ValueError, match="dtype"):
        root.add_into(target, wrong_dtype)

    builder = GraphBuilder()
    base = builder.input((2, 6), dtype="int32")
    root = base.relu()
    target = root.slice(axis=1, start=0, stop=6, step=2)
    wrong_shape = builder.input((2, 2), dtype="int32")
    with pytest.raises(ValueError, match="broadcast"):
        root.add_into(target, wrong_shape)
