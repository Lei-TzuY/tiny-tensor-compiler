import os
import shutil

import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    SymbolicDim,
    VerificationError,
    borrow_inputs,
    compile_dynamic_module,
    compile_module,
    execute_loop,
    execute_reference,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
    verify,
)


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _two_write_module(batch=2):
    builder = GraphBuilder()
    base = builder.input((batch, 6), dtype="int32")
    even_patch = builder.input((batch, 3), dtype="int32")
    odd_patch = builder.input((batch, 2), dtype="int32")

    owned = base.relu()
    even_target = owned.slice(axis=1, start=0, stop=6, step=2)
    generation1 = owned.copy_into(even_target, even_patch)

    snapshot = generation1 + 1
    odd_target = generation1.slice(axis=1, start=1, stop=5, step=2)
    generation2 = generation1.copy_into(odd_target, odd_patch)
    return builder.finish((generation2, snapshot))


def _expected(base, even_patch, odd_patch):
    generation1 = np.maximum(base, 0)
    generation1[:, 0:6:2] = even_patch
    snapshot = generation1 + 1
    generation2 = np.array(generation1, copy=True)
    generation2[:, 1:5:2] = odd_patch
    return generation2, snapshot


def test_two_ordered_writes_advance_one_root_across_reference_cpu_and_native():
    _default_compiler_or_skip()
    module = _two_write_module()
    base = np.arange(12, dtype=np.int32).reshape(2, 6) - 4
    even_patch = 100 + np.arange(6, dtype=np.int32).reshape(2, 3)
    odd_patch = 200 + np.arange(4, dtype=np.int32).reshape(2, 2)
    expected = _expected(base, even_patch, odd_patch)

    reference = execute_reference(module, inputs=[base, even_patch, odd_patch])
    loops = lower_to_loops(lower_to_cpu(module))
    cpu = execute_loop(borrow_inputs(loops), inputs=[base, even_patch, odd_patch])
    native = compile_module(module, borrow_inputs=True)(inputs=[base, even_patch, odd_patch])

    for actual in (reference, cpu, native):
        assert isinstance(actual, tuple)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    assert len(loops.copies) == 2
    assert loops.storage_root(loops.copies[0].output) == loops.storage_root(loops.copies[1].output)
    np.testing.assert_array_equal(base, np.arange(12, dtype=np.int32).reshape(2, 6) - 4)


def test_second_write_uses_fresh_full_root_handle_and_generated_c_keeps_it_mutable():
    module = _two_write_module()
    loops = lower_to_loops(lower_to_cpu(module))
    first, second = loops.copies

    assert second.root == first.output
    source = generate_c(loops)
    assert f"int32_t *p{first.output} = p{first.root};" in source
    assert f"p{second.root}[" in source


def test_tensor_verifier_rejects_stale_root_after_first_write():
    builder = GraphBuilder()
    base = builder.input((2, 6), dtype="int32")
    patch1 = builder.input((2, 3), dtype="int32")
    patch2 = builder.input((2, 2), dtype="int32")
    owned = base.relu()
    first_target = owned.slice(axis=1, start=0, stop=6, step=2)
    generation1 = owned.copy_into(first_target, patch1)
    second_target = generation1.slice(axis=1, start=1, stop=5, step=2)
    generation2 = owned.copy_into(second_target, patch2)
    module = builder.finish(generation2)

    with pytest.raises(VerificationError, match="stale tensor view/alias"):
        verify(module)


def test_tensor_verifier_rejects_returning_previous_generation_after_second_write():
    builder = GraphBuilder()
    base = builder.input((2, 6), dtype="int32")
    patch1 = builder.input((2, 3), dtype="int32")
    patch2 = builder.input((2, 2), dtype="int32")
    owned = base.relu()
    generation1 = owned.copy_into(
        owned.slice(axis=1, start=0, stop=6, step=2),
        patch1,
    )
    generation2 = generation1.copy_into(
        generation1.slice(axis=1, start=1, stop=5, step=2),
        patch2,
    )
    module = builder.finish((generation2, generation1))

    with pytest.raises(VerificationError, match="stale tensor view/alias"):
        verify(module)


def test_dynamic_ordered_writes_specialize_and_reuse_native_cache():
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    module = _two_write_module(batch)
    executable = compile_dynamic_module(module, borrow_inputs=True)

    for size in (2, 5, 0, 2):
        base = np.arange(size * 6, dtype=np.int32).reshape(size, 6) - 3
        even_patch = 50 + np.arange(size * 3, dtype=np.int32).reshape(size, 3)
        odd_patch = 90 + np.arange(size * 2, dtype=np.int32).reshape(size, 2)
        actual = executable(inputs=[base, even_patch, odd_patch])
        expected = _expected(base, even_patch, odd_patch)
        assert isinstance(actual, tuple)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    assert executable.cached_batch_sizes == (0, 2, 5)
