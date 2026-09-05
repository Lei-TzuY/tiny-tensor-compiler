from __future__ import annotations

from .c_codegen import _element_count, _emit_kernel, _select_i32_sse2_plan
from .ir import TensorType
from .loop_ir import LoopKernel

_OPENMP_PARALLEL_FOR = "#pragma omp parallel for schedule(static)"


def emit_parallel_kernel(
    op: LoopKernel,
    types: dict[int, TensorType],
    kernel_number: int,
) -> list[str]:
    """Emit one kernel with a barriered OpenMP loop when the scalar C path is safe."""
    lines = _emit_kernel(op, types, kernel_number)
    output_type = types[op.output]

    # The current SSE2 emitter owns a vector loop plus scalar tail. Keep that proven
    # implementation intact instead of stacking OpenMP directives onto its control flow.
    if _select_i32_sse2_plan(op, types) is not None:
        return lines
    if not op.iteration_shape or _element_count(output_type) == 0:
        return lines

    for index, line in enumerate(lines):
        stripped = line.strip()

        if stripped == "TINY_TENSOR_VECTORIZE_LOOP":
            loop_index = index + 1
            if loop_index >= len(lines) or not lines[loop_index].strip().startswith(
                "for (int64_t n ="
            ):
                raise RuntimeError(
                    "linearized kernel vectorization marker is not followed by its n loop"
                )
            _externalize_openmp_induction_variable(lines, loop_index, "n")
            lines[index] = f"{_indent_of(line)}int64_t n;"
            lines.insert(index + 1, f"{_indent_of(line)}{_OPENMP_PARALLEL_FOR}")
            return lines

        if stripped.startswith("for (int64_t i0 ="):
            indent = _indent_of(line)
            _externalize_openmp_induction_variable(lines, index, "i0")
            lines.insert(index, f"{indent}{_OPENMP_PARALLEL_FOR}")
            lines.insert(index, f"{indent}int64_t i0;")
            return lines

    raise RuntimeError("verified non-scalar kernel unexpectedly has no schedulable C loop")


def _externalize_openmp_induction_variable(
    lines: list[str],
    loop_index: int,
    variable: str,
) -> None:
    declaration = f"for (int64_t {variable} = 0;"
    replacement = f"for ({variable} = 0;"
    line = lines[loop_index]
    if declaration not in line:
        raise RuntimeError(f"OpenMP loop does not use the expected {variable} induction variable")
    lines[loop_index] = line.replace(declaration, replacement, 1)


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]
