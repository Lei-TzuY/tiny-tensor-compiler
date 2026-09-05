from tiny_tensor_compiler import IndexMap, LoopKernel
from tiny_tensor_compiler.simd_codegen import build_i32_sse2_plan


def _kernel(opcode: str, inputs: tuple[int, ...]) -> LoopKernel:
    identity = IndexMap((0,))
    return LoopKernel(
        opcode=opcode,
        output=9,
        inputs=inputs,
        iteration_shape=(9,),
        input_maps=(identity,) * len(inputs),
    )


def test_relu_chain_is_one_compositional_add_add_relu_plan():
    plan = build_i32_sse2_plan(_kernel("relu_chain_add_add", (0, 1, 2)))

    assert plan is not None
    assert plan.loads == (("lhs", 0), ("rhs", 1), ("tail", 2))
    assert [(step.opcode, step.output, step.inputs) for step in plan.steps] == [
        ("add", "inner", ("lhs", "rhs")),
        ("add", "result", ("inner", "tail")),
        ("relu", "relu", ("result",)),
    ]
    assert plan.result == "relu"


def test_add_tree_is_composed_from_three_fixed_width_add_steps():
    plan = build_i32_sse2_plan(_kernel("tree_add_add_add", (0, 1, 2, 3)))

    assert plan is not None
    assert [(step.opcode, step.output, step.inputs) for step in plan.steps] == [
        ("add", "left", ("a", "b")),
        ("add", "right", ("c", "d")),
        ("add", "result", ("left", "right")),
    ]
    assert plan.result == "result"


def test_plan_builder_does_not_expand_the_existing_sse2_capability_surface():
    assert build_i32_sse2_plan(_kernel("relu_mul", (0, 1))) is None
    assert build_i32_sse2_plan(_kernel("chain_add_mul", (0, 1, 2))) is None
    assert build_i32_sse2_plan(_kernel("tree_add_mul_add", (0, 1, 2, 3))) is None
