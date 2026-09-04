import numpy as np
import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    execute_cpu,
    execute_loop,
    execute_reference,
    fuse_elementwise,
    generate_c,
    lower_to_cpu,
    lower_to_loops,
    plan_memory,
    verify,
)


def _assert_pair(actual, first, second):
    assert isinstance(actual, tuple)
    assert len(actual) == 2
    np.testing.assert_array_equal(actual[0], first)
    np.testing.assert_array_equal(actual[1], second)


def _build_returned_intermediate_module():
    builder = GraphBuilder()
    x = builder.input((3,), dtype="int32")
    summed = x + 1
    activated = summed.relu()
    return builder.finish((summed, activated))


def test_frontend_and_reference_execute_multiple_outputs():
    module = _build_returned_intermediate_module()
    verify(module)

    assert "  return %2, %3" in module.dump()

    inputs = [np.array([-2, 0, 5], dtype=np.int32)]
    result = execute_reference(module, inputs=inputs)
    _assert_pair(
        result,
        np.array([-1, 1, 6], dtype=np.int32),
        np.array([0, 1, 6], dtype=np.int32),
    )


def test_lowering_keeps_co_returned_values_live_in_distinct_physical_slots():
    program = lower_to_cpu(_build_returned_intermediate_module())
    assert len(program.return_slots) == 2

    plan = plan_memory(program)
    physical_returns = tuple(plan.physical_for(slot) for slot in program.return_slots)
    assert len(set(physical_returns)) == 2

    loops = lower_to_loops(program)
    assert loops.return_slots == physical_returns


def test_cpu_and_loop_execution_match_reference_for_multiple_outputs():
    module = _build_returned_intermediate_module()
    inputs = [np.array([-2, 0, 5], dtype=np.int32)]
    reference = execute_reference(module, inputs=inputs)
    program = lower_to_cpu(module)
    loops = lower_to_loops(program)

    cpu = execute_cpu(program, inputs=inputs)
    interpreted = execute_loop(loops, inputs=inputs)

    assert isinstance(reference, tuple)
    _assert_pair(cpu, reference[0], reference[1])
    _assert_pair(interpreted, reference[0], reference[1])


def test_fusion_does_not_consume_an_intermediate_that_is_also_returned():
    module = _build_returned_intermediate_module()
    inputs = [np.array([-2, 0, 5], dtype=np.int32)]
    loops = lower_to_loops(lower_to_cpu(module))
    fused = fuse_elementwise(loops)

    executable_opcodes = tuple(kernel.opcode for kernel in fused.kernels if kernel.opcode != "const")
    assert executable_opcodes == ("add", "relu")

    reference = execute_reference(module, inputs=inputs)
    result = execute_loop(fused, inputs=inputs)
    assert isinstance(reference, tuple)
    _assert_pair(result, reference[0], reference[1])


def test_single_output_execution_and_compatibility_properties_remain_unchanged():
    builder = GraphBuilder()
    x = builder.input((2,), dtype="float32")
    module = builder.finish(x.relu())
    inputs = [np.array([-1.0, 2.0], dtype=np.float32)]

    reference = execute_reference(module, inputs=inputs)
    program = lower_to_cpu(module)
    loops = lower_to_loops(program)

    assert isinstance(reference, np.ndarray)
    assert program.return_slots == (program.return_slot,)
    assert loops.return_slots == (loops.return_slot,)
    np.testing.assert_array_equal(execute_cpu(program, inputs=inputs), reference)
    np.testing.assert_array_equal(execute_loop(loops, inputs=inputs), reference)


def test_native_codegen_rejects_multi_output_until_abi_support_exists():
    loops = lower_to_loops(lower_to_cpu(_build_returned_intermediate_module()))

    with pytest.raises(
        RuntimeError,
        match="return_slot requires exactly one returned buffer, found 2",
    ):
        generate_c(loops)


def test_graph_builder_rejects_empty_result_sequence():
    builder = GraphBuilder()

    with pytest.raises(ValueError, match="graph must return at least one tensor"):
        builder.finish(())
