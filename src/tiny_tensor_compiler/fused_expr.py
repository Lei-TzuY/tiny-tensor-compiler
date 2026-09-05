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
    """Canonical semantics for one integer fused LoopKernel opcode."""

    opcode: str
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


def describe_fused_opcode(opcode: str) -> FusedExpression | None:
    """Return the single canonical expression descriptor for a fused opcode."""
    if opcode in BINARY_CHAIN_OPCODES | RELU_BINARY_CHAIN_OPCODES:
        terminal_relu = opcode in RELU_BINARY_CHAIN_OPCODES
        base = opcode.removeprefix("relu_") if terminal_relu else opcode
        _, inner_opcode, outer_opcode = base.split("_")
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
            opcode=opcode,
            family="binary-chain",
            input_names=("lhs", "rhs", "tail"),
            steps=tuple(steps),
            result=result,
            terminal_relu=terminal_relu,
        )

    if opcode in BINARY_TREE_OPCODES | RELU_BINARY_TREE_OPCODES:
        terminal_relu = opcode in RELU_BINARY_TREE_OPCODES
        base = opcode.removeprefix("relu_") if terminal_relu else opcode
        _, left_opcode, right_opcode, root_opcode = base.split("_")
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
            opcode=opcode,
            family="binary-tree",
            input_names=("a", "b", "c", "d"),
            steps=tuple(steps),
            result=result,
            terminal_relu=terminal_relu,
        )

    if opcode in CHAIN_TREE_OPCODES:
        _, _, inner_opcode, left_opcode, right_opcode, root_opcode = opcode.split("_")
        return FusedExpression(
            opcode=opcode,
            family="chain-tree",
            input_names=("first_lhs", "first_rhs", "left_tail", "right_lhs", "right_rhs"),
            steps=(
                FusedExprStep(inner_opcode, "inner", ("first_lhs", "first_rhs")),
                FusedExprStep(left_opcode, "left", ("inner", "left_tail")),
                FusedExprStep(right_opcode, "right", ("right_lhs", "right_rhs")),
                FusedExprStep(root_opcode, "result", ("left", "right")),
            ),
            result="result",
            terminal_relu=False,
        )

    return None
