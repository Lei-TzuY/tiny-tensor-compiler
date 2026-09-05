from tiny_tensor_compiler.fused_expr import (
    BINARY_CHAIN_OPCODES,
    BINARY_TREE_OPCODES,
    CHAIN_TREE_OPCODES,
    FUSED_EXPRESSION_OPCODES,
    RELU_BINARY_CHAIN_OPCODES,
    RELU_BINARY_TREE_OPCODES,
    binary_chain_expression,
    binary_tree_expression,
    chain_tree_expression,
    describe_fused_opcode,
    encode_fused_opcode,
    with_terminal_relu,
)


def test_canonical_fused_expression_surface_is_complete_and_disjoint() -> None:
    families = (
        BINARY_CHAIN_OPCODES,
        RELU_BINARY_CHAIN_OPCODES,
        BINARY_TREE_OPCODES,
        RELU_BINARY_TREE_OPCODES,
        CHAIN_TREE_OPCODES,
    )
    assert tuple(map(len, families)) == (4, 4, 8, 8, 16)
    assert sum(map(len, families)) == len(FUSED_EXPRESSION_OPCODES) == 40
    assert set().union(*families) == set(FUSED_EXPRESSION_OPCODES)

    for opcode in FUSED_EXPRESSION_OPCODES:
        expression = describe_fused_opcode(opcode)
        assert expression is not None
        assert encode_fused_opcode(expression) == opcode
        assert expression.input_count in {3, 4, 5}
        assert expression.result == expression.steps[-1].output
        assert expression.terminal_relu == (expression.steps[-1].opcode == "relu")


def test_expression_builders_encode_legacy_spelling_only_at_boundary() -> None:
    chain = binary_chain_expression("add", "mul")
    assert encode_fused_opcode(chain) == "chain_add_mul"
    assert encode_fused_opcode(with_terminal_relu(chain)) == "relu_chain_add_mul"

    tree = binary_tree_expression("mul", "add", "mul")
    assert encode_fused_opcode(tree) == "tree_mul_add_mul"
    assert encode_fused_opcode(with_terminal_relu(tree)) == "relu_tree_mul_add_mul"

    chain_tree = chain_tree_expression("add", "mul", "add", "mul")
    assert encode_fused_opcode(chain_tree) == "chain_tree_add_mul_add_mul"


def test_relu_binary_chain_descriptor_preserves_fixed_width_step_order() -> None:
    expression = describe_fused_opcode("relu_chain_add_mul")
    assert expression is not None
    assert expression.family == "binary-chain"
    assert expression.input_names == ("lhs", "rhs", "tail")
    assert tuple(
        (step.opcode, step.output, step.inputs) for step in expression.steps
    ) == (
        ("add", "inner", ("lhs", "rhs")),
        ("mul", "value", ("inner", "tail")),
        ("relu", "relu", ("value",)),
    )
    assert expression.display_name == "integer ReLU binary-chain"


def test_binary_tree_and_chain_tree_descriptors_encode_dataflow() -> None:
    tree = describe_fused_opcode("tree_mul_add_mul")
    assert tree is not None
    assert tuple((step.opcode, step.inputs) for step in tree.steps) == (
        ("mul", ("a", "b")),
        ("add", ("c", "d")),
        ("mul", ("left", "right")),
    )

    chain_tree = describe_fused_opcode("chain_tree_add_mul_add_mul")
    assert chain_tree is not None
    assert chain_tree.input_names == (
        "first_lhs",
        "first_rhs",
        "left_tail",
        "right_lhs",
        "right_rhs",
    )
    assert tuple((step.output, step.inputs) for step in chain_tree.steps) == (
        ("inner", ("first_lhs", "first_rhs")),
        ("left", ("inner", "left_tail")),
        ("right", ("right_lhs", "right_rhs")),
        ("result", ("left", "right")),
    )


def test_non_fused_or_malformed_opcodes_have_no_descriptor() -> None:
    for opcode in (
        "add",
        "relu_add",
        "chain_add_div",
        "tree_add_add_div",
        "relu_chain_tree_add_add_add_add",
    ):
        assert describe_fused_opcode(opcode) is None
