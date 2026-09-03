import numpy as np

from tiny_tensor_compiler import (
    GraphBuilder,
    execute_cpu,
    execute_reference,
    lower_to_cpu,
    plan_memory,
)


def test_liveness_plan_reuses_dead_same_type_buffers_deterministically():
    builder = GraphBuilder()
    x = builder.tensor([-1, 2, -3], dtype="int32")
    module = builder.finish(x.relu().relu().relu())
    program = lower_to_cpu(module)

    plan = plan_memory(program)

    assert plan.physical_count == 2
    assert [plan.physical_for(buffer) for buffer in range(4)] == [0, 1, 0, 1]
    assert plan.dump() == "\n".join(
        [
            "b0 -> p0 : tensor<3xi32>",
            "b1 -> p1 : tensor<3xi32>",
            "b2 -> p0 : tensor<3xi32>",
            "b3 -> p1 : tensor<3xi32>",
        ]
    )


def test_liveness_plan_never_aliases_buffers_live_in_the_same_kernel():
    builder = GraphBuilder()
    x = builder.tensor([1, 2, 3], dtype="int32")
    y = builder.tensor([4, 5, 6], dtype="int32")
    module = builder.finish(x + y)
    program = lower_to_cpu(module)

    plan = plan_memory(program)

    x_slot = plan.physical_for(0)
    y_slot = plan.physical_for(1)
    output_slot = plan.physical_for(2)
    assert len({x_slot, y_slot, output_slot}) == 3


def test_liveness_plan_requires_exact_tensor_type_for_reuse():
    builder = GraphBuilder()
    _dead_scalar = builder.tensor(7, dtype="int32")
    live_vector = builder.tensor([1, 2], dtype="int32")
    program = lower_to_cpu(builder.finish(live_vector))

    plan = plan_memory(program)

    assert plan.physical_count == 2
    assert plan.physical_for(0) != plan.physical_for(1)


def test_cpu_execution_uses_memory_plan_without_changing_results():
    builder = GraphBuilder()
    x = builder.tensor([-2.0, 1.5, 3.0], dtype="float32")
    module = builder.finish(x.relu().relu().relu())
    program = lower_to_cpu(module)
    plan = plan_memory(program)

    assert plan.physical_count < len(program.allocations)
    np.testing.assert_array_equal(execute_cpu(program), execute_reference(module))
