from __future__ import annotations

from dataclasses import dataclass

_BINARY_OPCODES = ("add", "mul")

BINARY_CHAIN_OPCODES = frozenset(
    f"chain_{inner}_{outer}"
    for inner in _BINARY_OPCODES
    for outer in _BINARY_OPCODES
)
RELU_BINARY_CHAIN_OPCODES = frozenset(
    f"relu_{opcode}" for opcode in BINARY_CHAIN_OPCODES
)
BINARY_TREE_OPCODES = frozenset(
    f"tree_{left}_{right}_{root}"
    for left in _BINARY_OPCODES
    for right in _BINARY_OPCODES
    for root in _BINARY_OPCODES
)
RELU_BINARY_TREE_OPCODES = frozenset(
    f"relu_{opcode}" for opcode in BINARY_TREE_OPCODES
)
CHAIN_TREE_OPCODES = frozenset(
    f"chain_tree_{inner}_{left}_{right}_{root}"
    for inner in _BINARY_OPCODES
    for left in _BINARY_OPCODES
    for right in _BINARY_OPCODES
    for root in _BINARY_OPCODES
)
FUSED_EXPRESSION_OPCODES = frozenset(
    BINARY_CHAIN_OPCODES
    | RELU_BINARY_CHAIN_OPCODES
    | BINARY_TREE_OPCODES
    | RELU_BINARY_TREE_OPCODES
    | CHAIN_TREE_OPCODES
)


@dataclass(frozen=True)
class FusedExprStep:
    """One ordered scalar semantic step in a fused elementwise expression."""

    opcode: str
    output: str
    inputs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.opcode not in {"add", "mul", "relu"}:
            raise ValueError(f"unsupported fused expression step: {self.opcode}")
        expected_arity = 1 if self.opcode == "relu" else 2
        if len(self.inputs) != expected_arity:
            raise ValueError(
                f"fused {self.opcode} step requires {expected_arity} inputs, "
                f"got {len(self.inputs)}"
            )


@dataclass(frozen=True)
class FusedExpression:
    """Canonical semantics for one integer fused elementwise expression."""

    family: str
    input_names: tuple[str, ...]
    steps: tuple[FusedExprStep, ...]
    result: str
    terminal_relu: bool

    def __post_init__(self) -> None:
        available = set(self.input_names)
        if len(available) != len(self.input_names):
            raise ValueError("fused expression input names must be unique")
        for step in self.steps:
            if step.output in available:
                raise ValueError(f"duplicate fused expression value: {step.output}")
            missing = tuple(value for value in step.inputs if value not in available)
            if missing:
                raise ValueError(
                    f"fused expression step {step.output} references unavailable values {missing}"
                )
            available.add(step.output)
        if self.result not in available:
            raise ValueError(f"unknown fused expression result: {self.result}")
        if self.terminal_relu != (
            bool(self.steps) and self.steps[-1].opcode == "relu"
        ):
            raise ValueError("terminal_relu must match the final fused expression step")

    @property
    def input_count(self) -> int:
        return len(self.input_names)

    @property
    def display_name(self) -> str:
        if self.family == "binary-chain":
            base = "integer binary-chain"
        elif self.family == "binary-tree":
            base = "integer binary-tree"
        elif self.family == "chain-tree":
            base = "integer chain-tree"
        else:
            raise ValueError(f"unknown fused expression family: {self.family}")
        if self.terminal_relu:
            return base.replace("integer ", "integer ReLU ", 1)
        return base


def binary_chain_expression(
    inner_opcode: str,
    outer_opcode: str,
    *,
    terminal_relu: bool = False,
) -> FusedExpression:
    """Build the canonical three-input binary-chain expression."""
    _require_binary_opcode(inner_opcode)
    _require_binary_opcode(outer_opcode)
    steps = [FusedExprStep(inner_opcode, "inner", ("lhs", "rhs"))]
    if terminal_relu:
        steps.extend(
            (
                FusedExprStep(outer_opcode, "value", ("inner", "tail")),
                FusedExprStep("relu", "relu", ("value",)),
            )
        )
        result = "relu"
    else:
        steps.append(FusedExprStep(outer_opcode, "result", ("inner", "tail")))
        result = "result"
    return FusedExpression(
        family="binary-chain",
        input_names=("lhs", "rhs", "tail"),
        steps=tuple(steps),
        result=result,
        terminal_relu=terminal_relu,
    )


def binary_tree_expression(
    left_opcode: str,
    right_opcode: str,
    root_opcode: str,
    *,
    terminal_relu: bool = False,
) -> FusedExpression:
    """Build the canonical four-input binary-tree expression."""
    for opcode in (left_opcode, right_opcode, root_opcode):
        _require_binary_opcode(opcode)
    steps = [
        FusedExprStep(left_opcode, "left", ("a", "b")),
        FusedExprStep(right_opcode, "right", ("c", "d")),
    ]
    if terminal_relu:
        steps.extend(
            (
                FusedExprStep(root_opcode, "value", ("left", "right")),
                FusedExprStep("relu", "relu", ("value",)),
            )
        )
        result = "relu"
    else:
        steps.append(FusedExprStep(root_opcode, "result", ("left", "right")))
        result = "result"
    return FusedExpression(
        family="binary-tree",
        input_names=("a", "b", "c", "d"),
        steps=tuple(steps),
        result=result,
        terminal_relu=terminal_relu,
    )


def chain_tree_expression(
    inner_opcode: str,
    left_opcode: str,
    right_opcode: str,
    root_opcode: str,
) -> FusedExpression:
    """Build the canonical five-input chain-tree expression."""
    for opcode in (inner_opcode, left_opcode, right_opcode, root_opcode):
        _require_binary_opcode(opcode)
    return FusedExpression(
        family="chain-tree",
        input_names=(
            "first_lhs",
            "first_rhs",
            "left_tail",
            "right_lhs",
            "right_rhs",
        ),
        steps=(
            FusedExprStep(inner_opcode, "inner", ("first_lhs", "first_rhs")),
            FusedExprStep(left_opcode, "left", ("inner", "left_tail")),
            FusedExprStep(right_opcode, "right", ("right_lhs", "right_rhs")),
            FusedExprStep(root_opcode, "result", ("left", "right")),
        ),
        result="result",
        terminal_relu=False,
    )


def with_terminal_relu(expression: FusedExpression) -> FusedExpression:
    """Return the canonical terminal-ReLU form of a supported fused expression."""
    if expression.terminal_relu:
        return expression
    if expression.family == "binary-chain" and len(expression.steps) == 2:
        inner, outer = expression.steps
        expected = binary_chain_expression(inner.opcode, outer.opcode)
        if expression != expected:
            raise ValueError("cannot add terminal ReLU to a noncanonical binary-chain")
        return binary_chain_expression(
            inner.opcode,
            outer.opcode,
            terminal_relu=True,
        )
    if expression.family == "binary-tree" and len(expression.steps) == 3:
        left, right, root = expression.steps
        expected = binary_tree_expression(left.opcode, right.opcode, root.opcode)
        if expression != expected:
            raise ValueError("cannot add terminal ReLU to a noncanonical binary-tree")
        return binary_tree_expression(
            left.opcode,
            right.opcode,
            root.opcode,
            terminal_relu=True,
        )
    raise ValueError(f"terminal ReLU is unsupported for {expression.family}")


def encode_fused_opcode(expression: FusedExpression) -> str:
    """Encode a canonical structured expression using the legacy Loop IR spelling."""
    if expression.family == "binary-chain":
        binary_steps = expression.steps[:-1] if expression.terminal_relu else expression.steps
        if len(binary_steps) != 2:
            raise ValueError("noncanonical binary-chain cannot be encoded")
        inner, outer = binary_steps
        expected = binary_chain_expression(
            inner.opcode,
            outer.opcode,
            terminal_relu=expression.terminal_relu,
        )
        if expression != expected:
            raise ValueError("noncanonical binary-chain cannot be encoded")
        opcode = f"chain_{inner.opcode}_{outer.opcode}"
        return f"relu_{opcode}" if expression.terminal_relu else opcode

    if expression.family == "binary-tree":
        binary_steps = expression.steps[:-1] if expression.terminal_relu else expression.steps
        if len(binary_steps) != 3:
            raise ValueError("noncanonical binary-tree cannot be encoded")
        left, right, root = binary_steps
        expected = binary_tree_expression(
            left.opcode,
            right.opcode,
            root.opcode,
            terminal_relu=expression.terminal_relu,
        )
        if expression != expected:
            raise ValueError("noncanonical binary-tree cannot be encoded")
        opcode = f"tree_{left.opcode}_{right.opcode}_{root.opcode}"
        return f"relu_{opcode}" if expression.terminal_relu else opcode

    if expression.family == "chain-tree":
        if len(expression.steps) != 4 or expression.terminal_relu:
            raise ValueError("noncanonical chain-tree cannot be encoded")
        inner, left, right, root = expression.steps
        expected = chain_tree_expression(
            inner.opcode,
            left.opcode,
            right.opcode,
            root.opcode,
        )
        if expression != expected:
            raise ValueError("noncanonical chain-tree cannot be encoded")
        return (
            f"chain_tree_{inner.opcode}_{left.opcode}_{right.opcode}_{root.opcode}"
        )

    raise ValueError(f"unknown fused expression family: {expression.family}")


def describe_fused_opcode(opcode: str) -> FusedExpression | None:
    """Decode legacy Loop IR spelling into one canonical structured expression."""
    if opcode in BINARY_CHAIN_OPCODES | RELU_BINARY_CHAIN_OPCODES:
        terminal_relu = opcode in RELU_BINARY_CHAIN_OPCODES
        base = opcode.removeprefix("relu_") if terminal_relu else opcode
        _, inner_opcode, outer_opcode = base.split("_")
        return binary_chain_expression(
            inner_opcode,
            outer_opcode,
            terminal_relu=terminal_relu,
        )

    if opcode in BINARY_TREE_OPCODES | RELU_BINARY_TREE_OPCODES:
        terminal_relu = opcode in RELU_BINARY_TREE_OPCODES
        base = opcode.removeprefix("relu_") if terminal_relu else opcode
        _, left_opcode, right_opcode, root_opcode = base.split("_")
        return binary_tree_expression(
            left_opcode,
            right_opcode,
            root_opcode,
            terminal_relu=terminal_relu,
        )

    if opcode in CHAIN_TREE_OPCODES:
        _, _, inner_opcode, left_opcode, right_opcode, root_opcode = opcode.split("_")
        return chain_tree_expression(
            inner_opcode,
            left_opcode,
            right_opcode,
            root_opcode,
        )

    return None


def _require_binary_opcode(opcode: str) -> None:
    if opcode not in _BINARY_OPCODES:
        raise ValueError(f"unsupported fused binary opcode: {opcode}")
