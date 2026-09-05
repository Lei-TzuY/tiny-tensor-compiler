import numpy as np

from tiny_tensor_compiler import GraphBuilder, execute_reference
from tiny_tensor_compiler.repro import (
    capture_repro_case,
    load_repro_case,
    replay_repro_case,
)
from tiny_tensor_compiler.serialization import deserialize_module, serialize_module


def _sum_op(module):
    return next(op for op in module.function.ops if op.opcode == "sum")


def test_axis_sum_round_trips_through_canonical_ir_serialization() -> None:
    builder = GraphBuilder()
    value = builder.input((2, 3, 4), dtype="int32")
    module = builder.finish(value.sum(axis=-2))

    document = serialize_module(module)
    restored = deserialize_module(document)

    assert _sum_op(module).attrs == {"axis": 1}
    assert _sum_op(restored).attrs == {"axis": 1}
    assert serialize_module(restored) == document

    runtime = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    np.testing.assert_array_equal(
        execute_reference(restored, inputs=[runtime]),
        execute_reference(module, inputs=[runtime]),
    )


def test_axis_sum_repro_capture_load_and_replay_preserve_axis_metadata() -> None:
    builder = GraphBuilder()
    value = builder.input((3, 4, 2), dtype="float32")
    module = builder.finish(value.reverse(1).sum(axis=1))
    runtime = np.arange(24, dtype=np.float32).reshape(3, 4, 2) - 7.0

    document = capture_repro_case(module, inputs=[runtime])
    case = load_repro_case(document)

    assert _sum_op(case.module).attrs == {"axis": 1}
    actual = replay_repro_case(document)
    expected = execute_reference(module, inputs=[runtime])
    np.testing.assert_array_equal(actual, expected)
