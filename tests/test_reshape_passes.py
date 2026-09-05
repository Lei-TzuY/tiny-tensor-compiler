import numpy as np

from tiny_tensor_compiler import (
    GraphBuilder,
    common_subexpression_eliminate,
    dead_code_eliminate,
    execute_reference,
    verify,
)


def test_dead_code_elimination_removes_unused_reshape_but_keeps_input_signature():
    builder = GraphBuilder()
    live = builder.input((2, 3), dtype="int32")
    dead = builder.input((2, 3), dtype="int32")
    _unused = dead.reshape((3, 2))
    module = builder.finish(live)

    assert dead_code_eliminate(module) == 1
    verify(module)
    assert [op.opcode for op in module.function.ops] == ["input", "input", "return"]


def test_cse_merges_identical_reshape_of_same_value_and_type():
    builder = GraphBuilder()
    value = builder.input((2, 3), dtype="int32")
    lhs = value.reshape((3, 2))
    rhs = value.reshape((3, 2))
    module = builder.finish(lhs + rhs)
    runtime = np.arange(6, dtype=np.int32).reshape(2, 3)
    before = execute_reference(module, inputs=[runtime])

    assert common_subexpression_eliminate(module) == 1
    verify(module)
    add = next(op for op in module.function.ops if op.opcode == "add")
    assert add.operands[0] is add.operands[1]
    assert [op.opcode for op in module.function.ops].count("reshape") == 1
    np.testing.assert_array_equal(execute_reference(module, inputs=[runtime]), before)
