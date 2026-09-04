import os
import shutil

import numpy as np
import pytest

import tiny_tensor_compiler.compiler as compiler_module
from tiny_tensor_compiler import (
    BorrowedLoopProgram,
    GraphBuilder,
    IndexMap,
    LoopAlloc,
    LoopInput,
    LoopKernel,
    LoopProgram,
    LoopReturn,
    borrow_inputs,
    compile_module,
    execute_loop,
    execute_native,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
)
from tiny_tensor_compiler.backends import cpu as cpu_module
from tiny_tensor_compiler.ir import DType, TensorType


def _default_compiler_or_skip() -> None:
    executable = "cl" if os.name == "nt" else "cc"
    if shutil.which(executable) is None:
        pytest.skip(f"no platform default C compiler available: {executable}")


def _reused_input_slot_program() -> LoopProgram:
    type_ = TensorType((3,), DType.INT32)
    identity = IndexMap((0,))
    return LoopProgram(
        (
            LoopAlloc(0, type_),
            LoopAlloc(1, type_),
            LoopInput(0, 0),
            LoopKernel("relu", 1, (0,), (3,), (identity,)),
            LoopKernel("relu", 0, (1,), (3,), (identity,)),
            LoopReturn(0),
        )
    )


def test_borrow_inputs_splits_a_reused_input_lifetime_and_reverifies_loop_ir():
    program = _reused_input_slot_program()

    borrowed = borrow_inputs(program)

    assert isinstance(borrowed, BorrowedLoopProgram)
    assert borrowed.borrowed_inputs[0].buffer == 2
    assert [alloc.buffer for alloc in borrowed.allocations] == [0, 1, 2]
    assert borrowed.inputs == (LoopInput(2, 0),)
    assert borrowed.kernels[0].inputs == (2,)
    assert borrowed.kernels[1].output == 0
    assert borrowed.return_slots == (0,)


def test_cpu_borrowed_execution_binds_caller_array_without_input_copy(monkeypatch):
    borrowed = borrow_inputs(_reused_input_slot_program())
    values = np.array([-4, 0, 7], dtype=np.int32)
    values.setflags(write=False)

    def unexpected_copyto(*args, **kwargs):
        raise AssertionError("borrowed input must not be materialized through np.copyto")

    monkeypatch.setattr(cpu_module.np, "copyto", unexpected_copyto)

    actual = execute_loop(borrowed, inputs=[values])

    np.testing.assert_array_equal(actual, np.array([0, 0, 7], dtype=np.int32))
    np.testing.assert_array_equal(values, np.array([-4, 0, 7], dtype=np.int32))


def test_borrowed_codegen_uses_const_input_pointer_and_preserves_scratch_reuse():
    program = _reused_input_slot_program()
    copied_source = generate_c(program)
    borrowed_source = generate_c(borrow_inputs(program))

    assert "int32_t p0[3];" in borrowed_source
    assert "int32_t p1[3];" in borrowed_source
    assert "const int32_t *p2 = input0;" in borrowed_source
    assert "p0[n] = input0[n];" in copied_source
    assert "p2[n] = input0[n];" not in borrowed_source
    assert "p1[n] = p2[n]" in borrowed_source
    assert "p0[n] = p1[n]" in borrowed_source


def test_borrowed_runtime_inputs_reject_hidden_materialization_paths():
    borrowed = borrow_inputs(_reused_input_slot_program())

    with pytest.raises(TypeError, match="must be a numpy.ndarray"):
        execute_loop(borrowed, inputs=[[-1, 0, 2]])

    noncontiguous = np.arange(6, dtype=np.int32)[::2]
    assert not noncontiguous.flags.c_contiguous
    with pytest.raises(ValueError, match="must be C-contiguous"):
        execute_loop(borrowed, inputs=[noncontiguous])

    raw = np.zeros(13, dtype=np.uint8)
    misaligned = np.ndarray((3,), dtype=np.int32, buffer=raw, offset=1)
    assert not misaligned.flags.aligned
    with pytest.raises(ValueError, match="must be aligned"):
        execute_loop(borrowed, inputs=[misaligned])


def test_safe_input_slots_are_borrowed_in_place_without_extra_physical_slots():
    builder = GraphBuilder()
    lhs = builder.input((3,), dtype="float32")
    rhs = builder.input((3,), dtype="float32")
    module = builder.finish(lhs + rhs)
    loops = lower_to_loops(lower_to_cpu(module))

    borrowed = borrow_inputs(loops)

    assert len(borrowed.allocations) == len(loops.allocations)
    assert borrowed.borrowed_input_slots == frozenset(op.output for op in loops.inputs)
    source = generate_c(borrowed)
    for input_op in borrowed.inputs:
        assert f"*p{input_op.output} = input{input_op.index};" in source
        assert f"p{input_op.output}[n] = input{input_op.index}[n];" not in source


def test_high_level_compile_module_enables_verified_borrowing_only_when_requested(monkeypatch):
    builder = GraphBuilder()
    lhs = builder.input((2,), dtype="float32")
    rhs = builder.input((2,), dtype="float32")
    module = builder.finish(lhs + rhs)
    captured = []
    sentinel = object()

    def fake_compile_native(program, compiler=None, cache_dir=None):
        captured.append(program)
        return sentinel

    monkeypatch.setattr(compiler_module, "compile_native", fake_compile_native)

    assert compile_module(module) is sentinel
    assert compile_module(module, borrow_inputs=True) is sentinel
    assert not isinstance(captured[0], BorrowedLoopProgram)
    assert isinstance(captured[1], BorrowedLoopProgram)
    assert captured[1].borrowed_input_indices == frozenset({0, 1})


def test_borrowed_native_execution_matches_copy_path_for_reused_slot_and_multi_output():
    _default_compiler_or_skip()
    values = np.array([-5, 0, 9], dtype=np.int32)
    program = _reused_input_slot_program()

    copied = execute_native(program, inputs=[values])
    borrowed = execute_native(borrow_inputs(program), inputs=[values])

    np.testing.assert_array_equal(borrowed, copied)

    builder = GraphBuilder()
    lhs = builder.input((3,), dtype="float32")
    rhs = builder.input((3,), dtype="float32")
    module = builder.finish((lhs + rhs, lhs * rhs))
    loops = lower_to_loops(lower_to_cpu(module))
    borrowed_loops = borrow_inputs(loops)
    lhs_value = np.array([1.0, -2.0, 4.0], dtype=np.float32)
    rhs_value = np.array([3.0, 5.0, -1.0], dtype=np.float32)

    add_result, mul_result = execute_native(
        borrowed_loops,
        inputs=[lhs_value, rhs_value],
    )

    np.testing.assert_array_equal(add_result, lhs_value + rhs_value)
    np.testing.assert_array_equal(mul_result, lhs_value * rhs_value)
