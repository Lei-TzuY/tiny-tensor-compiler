from __future__ import annotations

from dataclasses import dataclass

from .loop_ir import LoopKernel


@dataclass(frozen=True)
class I32SSE2Step:
    """One fixed-width vector expression step in a bounded SSE2 plan."""

    opcode: str
    output: str
    inputs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.opcode not in {"add", "relu"}:
            raise ValueError(f"unsupported SSE2 i32 step: {self.opcode}")
        expected_arity = 2 if self.opcode == "add" else 1
        if len(self.inputs) != expected_arity:
            raise ValueError(
                f"SSE2 i32 {self.opcode} requires {expected_arity} inputs, "
                f"got {len(self.inputs)}"
            )


@dataclass(frozen=True)
class I32SSE2Plan:
    """Compositional load/expression plan for one contiguous int32 kernel."""

    loads: tuple[tuple[str, int], ...]
    steps: tuple[I32SSE2Step, ...]
    result: str

    def __post_init__(self) -> None:
        available = {name for name, _ in self.loads}
        if len(available) != len(self.loads):
            raise ValueError("SSE2 i32 load names must be unique")
        for step in self.steps:
            if step.output in available:
                raise ValueError(f"duplicate SSE2 i32 value name: {step.output}")
            missing = [value for value in step.inputs if value not in available]
            if missing:
                raise ValueError(
                    f"SSE2 i32 step {step.output} references unavailable values {missing}"
                )
            available.add(step.output)
        if self.result not in available:
            raise ValueError(f"unknown SSE2 i32 plan result: {self.result}")


def build_i32_sse2_plan(op: LoopKernel) -> I32SSE2Plan | None:
    """Describe the existing SSE2 int32 family without encoding loop mechanics per opcode."""
    if op.opcode == "add":
        lhs, rhs = op.inputs
        return I32SSE2Plan(
            loads=(("lhs", lhs), ("rhs", rhs)),
            steps=(I32SSE2Step("add", "sum", ("lhs", "rhs")),),
            result="sum",
        )
    if op.opcode == "relu":
        (operand,) = op.inputs
        return I32SSE2Plan(
            loads=(("value", operand),),
            steps=(I32SSE2Step("relu", "relu", ("value",)),),
            result="relu",
        )
    if op.opcode == "relu_add":
        lhs, rhs = op.inputs
        return I32SSE2Plan(
            loads=(("lhs", lhs), ("rhs", rhs)),
            steps=(
                I32SSE2Step("add", "sum", ("lhs", "rhs")),
                I32SSE2Step("relu", "relu", ("sum",)),
            ),
            result="relu",
        )
    if op.opcode in {"chain_add_add", "relu_chain_add_add"}:
        lhs, rhs, tail = op.inputs
        steps = [
            I32SSE2Step("add", "inner", ("lhs", "rhs")),
            I32SSE2Step("add", "result", ("inner", "tail")),
        ]
        result = "result"
        if op.opcode == "relu_chain_add_add":
            steps.append(I32SSE2Step("relu", "relu", ("result",)))
            result = "relu"
        return I32SSE2Plan(
            loads=(("lhs", lhs), ("rhs", rhs), ("tail", tail)),
            steps=tuple(steps),
            result=result,
        )
    if op.opcode == "tree_add_add_add":
        a, b, c, d = op.inputs
        return I32SSE2Plan(
            loads=(("a", a), ("b", b), ("c", c), ("d", d)),
            steps=(
                I32SSE2Step("add", "left", ("a", "b")),
                I32SSE2Step("add", "right", ("c", "d")),
                I32SSE2Step("add", "result", ("left", "right")),
            ),
            result="result",
        )
    return None


def emit_i32_sse2_plan(plan: I32SSE2Plan, *, output: int, count: int) -> list[str]:
    """Emit one guarded SSE2 loop plus fixed-width scalar tail/fallback from a plan."""
    lines = [
        "        #if TINY_TENSOR_HAS_SSE2",
        "        int64_t n = 0;",
        f"        for (; n + 4 <= {count}; n += 4) {{",
    ]
    for name, buffer in plan.loads:
        lines.append(
            f"            __m128i {name} = "
            f"_mm_loadu_si128((const __m128i *)&p{buffer}[n]);"
        )
    for step in plan.steps:
        lines.extend(_emit_vector_step(step))
    lines.append(f"            _mm_storeu_si128((__m128i *)&p{output}[n], {plan.result});")
    lines.append("        }")
    lines.append(f"        for (; n < {count}; ++n) {{")
    lines.extend(_emit_scalar_plan(plan, output=output, indent="            "))
    lines.extend(
        [
            "        }",
            "        #else",
            "        TINY_TENSOR_VECTORIZE_LOOP",
            f"        for (int64_t n = 0; n < {count}; ++n) {{",
        ]
    )
    lines.extend(_emit_scalar_plan(plan, output=output, indent="            "))
    lines.extend(["        }", "        #endif"])
    return lines


def _emit_vector_step(step: I32SSE2Step) -> list[str]:
    if step.opcode == "add":
        lhs, rhs = step.inputs
        return [f"            __m128i {step.output} = _mm_add_epi32({lhs}, {rhs});"]

    (operand,) = step.inputs
    return [
        "            __m128i zero = _mm_setzero_si128();",
        f"            __m128i positive = _mm_cmpgt_epi32({operand}, zero);",
        f"            __m128i {step.output} = _mm_and_si128({operand}, positive);",
    ]


def _emit_scalar_plan(plan: I32SSE2Plan, *, output: int, indent: str) -> list[str]:
    load_refs = {name: f"p{buffer}[n]" for name, buffer in plan.loads}
    producers = {step.output: step for step in plan.steps}
    result_step = producers.get(plan.result)
    if result_step is not None and result_step.opcode == "relu":
        return _emit_scalar_relu_result(
            plan,
            relu_step=result_step,
            load_refs=load_refs,
            output=output,
            indent=indent,
        )

    lines, refs = _materialize_scalar_adds(
        plan,
        stop_before=plan.result,
        load_refs=load_refs,
        indent=indent,
    )
    if result_step is None or result_step.opcode != "add":
        raise ValueError("SSE2 i32 scalar result must be an add or relu step")
    lhs, rhs = result_step.inputs
    lines.append(
        f"{indent}p{output}[n] = ((int32_t){refs[lhs]} + (int32_t){refs[rhs]});"
    )
    return lines


def _emit_scalar_relu_result(
    plan: I32SSE2Plan,
    *,
    relu_step: I32SSE2Step,
    load_refs: dict[str, str],
    output: int,
    indent: str,
) -> list[str]:
    (operand,) = relu_step.inputs
    producers = {step.output: step for step in plan.steps}
    operand_step = producers.get(operand)

    if operand_step is not None and operand_step.opcode == "add":
        lines, refs = _materialize_scalar_adds(
            plan,
            stop_before=operand,
            load_refs=load_refs,
            indent=indent,
        )
        lhs, rhs = operand_step.inputs
        lines.append(
            f"{indent}int32_t value = ((int32_t){refs[lhs]} + (int32_t){refs[rhs]});"
        )
    elif operand in load_refs:
        lines = [f"{indent}int32_t value = (int32_t){load_refs[operand]};"]
    else:
        raise ValueError("SSE2 i32 relu operand must be a load or add result")

    lines.append(f"{indent}p{output}[n] = value < 0 ? 0 : value;")
    return lines


def _materialize_scalar_adds(
    plan: I32SSE2Plan,
    *,
    stop_before: str,
    load_refs: dict[str, str],
    indent: str,
) -> tuple[list[str], dict[str, str]]:
    lines: list[str] = []
    refs = dict(load_refs)
    for step in plan.steps:
        if step.output == stop_before:
            break
        if step.opcode == "relu":
            raise ValueError("SSE2 i32 plan only supports a terminal relu")
        lhs, rhs = step.inputs
        lines.append(
            f"{indent}int32_t {step.output} = "
            f"((int32_t){refs[lhs]} + (int32_t){refs[rhs]});"
        )
        refs[step.output] = step.output
    return lines, refs
