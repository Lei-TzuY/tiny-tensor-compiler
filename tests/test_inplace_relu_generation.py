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


def _module(batch=2):
    builder = GraphBuilder()
    base = builder.input((batch, 6), dtype="float32")
    owned = base + 1.0
    target = owned.slice(axis=1, start=0, stop=6, step=2).reverse(1)
    fresh = owned.relu_into(target)
    snapshot = fresh + 2.0
    return builder.finish((fresh, snapshot))


def _expected(base):
    owned = np.asarray(base, dtype=np.float32) + np.float32(1.0)
    expected = np.array(owned, copy=True)
    view = expected[:, 0:6:2][:, ::-1]
    np.maximum(view, np.float32(0.0), out=view)
    return expected, expected + np.float32(2.0)


def test_inplace_relu_updates_only_verified_alias_region_across_reference_loop_and_native():
    _default_compiler_or_skip()
    module = _module()
    base = np.array(
        [[-9.0, -8.0, -7.0, -6.0, -5.0, -4.0], [1.0, -3.0, 2.0, -5.0, 3.0, -7.0]],
        dtype=np.float32,
    )
    expected = _expected(base)

    reference = execute_reference(module, inputs=[base])
    loops = lower_to_loops(lower_to_cpu(module))
    cpu = execute_loop(borrow_inputs(loops), inputs=[base])
    native = compile_module(module, borrow_inputs=True, parallel=True)(inputs=[base])

    for actual in (reference, cpu, native):
        assert isinstance(actual, tuple)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    np.testing.assert_array_equal(
        base,
        np.array(
            [[-9.0, -8.0, -7.0, -6.0, -5.0, -4.0], [1.0, -3.0, 2.0, -5.0, 3.0, -7.0]],
            dtype=np.float32,
        ),
    )


def test_inplace_relu_advances_generation_and_stales_old_root_and_view():
    builder = GraphBuilder()
    base = builder.input((2, 4), dtype="int32")
    owned = base + 3
    target = owned.transpose((1, 0)).slice(axis=0, start=1, stop=4, step=2)
    fresh = owned.relu_into(target)
    module = builder.finish((fresh, owned, target))

    with pytest.raises(VerificationError, match="stale tensor view/alias"):
        verify(module)


def test_inplace_relu_rejects_input_and_constant_roots():
    builder = GraphBuilder()
    external = builder.input((4,), dtype="float32")
    with pytest.raises(ValueError, match="internal computed storage"):
        external.relu_into(external)

    literal = builder.tensor([-1.0, 2.0], dtype="float32")
    with pytest.raises(ValueError, match="internal computed storage"):
        literal.relu_into(literal)


def test_inplace_relu_requires_target_from_same_root():
    builder = GraphBuilder()
    x = builder.input((4,), dtype="int32")
    y = builder.input((4,), dtype="int32")
    lhs = x + 1
    rhs = y + 2
    with pytest.raises(ValueError, match="target must alias"):
        lhs.relu_into(rhs)


def test_generated_c_keeps_inplace_relu_serial_and_exposes_fresh_mutable_root():
    module = _module()
    loops = lower_to_loops(lower_to_cpu(module))
    source = generate_c(loops, parallel=True)
    effects = loops.relu_writes
    assert len(effects) == 1
    effect = effects[0]

    assert "#pragma omp parallel for schedule(static)" in source
    assert f"float *p{effect.output} = p{effect.root};" in source
    assert "isnan(value)" in source
    assert "fabsf(value)" in source

    effect_line = f"float *p{effect.output} = p{effect.root};"
    effect_position = source.index(effect_line)
    before_effect = source[:effect_position]
    assert before_effect.rfind("#pragma omp parallel for schedule(static)") < before_effect.rfind("}")


def test_inplace_relu_float_edges_match_existing_relu_semantics():
    _default_compiler_or_skip()
    builder = GraphBuilder()
    x = builder.input((4,), dtype="float32")
    owned = x + np.float32(0.5)
    fresh = owned.relu_into(owned.view((4,)))
    module = builder.finish(fresh)
    values = np.array([np.nan, -0.5, -2.5, np.inf], dtype=np.float32)

    expected = np.maximum(values + np.float32(0.5), np.float32(0.0))
    reference = execute_reference(module, inputs=[values])
    native = compile_module(module)(inputs=[values])

    np.testing.assert_array_equal(np.isnan(reference), np.isnan(expected))
    np.testing.assert_array_equal(np.isnan(native), np.isnan(expected))
    np.testing.assert_array_equal(reference[1:], expected[1:])
    np.testing.assert_array_equal(native[1:], expected[1:])
    assert not np.signbit(reference[1])
    assert not np.signbit(native[1])


def test_dynamic_inplace_relu_specializes_and_reuses_native_cache():
    _default_compiler_or_skip()
    batch = SymbolicDim("B")
    module = _module(batch)
    executable = compile_dynamic_module(module, borrow_inputs=True, parallel=True)

    for size in (3, 0, 5, 3):
        base = np.arange(size * 6, dtype=np.float32).reshape(size, 6) - np.float32(8.0)
        actual = executable(inputs=[base])
        expected = _expected(base)
        assert isinstance(actual, tuple)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    assert executable.cached_batch_sizes == (0, 3, 5)
