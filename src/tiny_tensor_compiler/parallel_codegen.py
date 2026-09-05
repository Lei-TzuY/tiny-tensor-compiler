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
        indent = line[: len(line) - len(line.lstrip())]

        if stripped == "TINY_TENSOR_VECTORIZE_LOOP":
            if index + 1 >= len(lines) or not lines[index + 1].strip().startswith("for ("):
                raise RuntimeError("linearized kernel vectorization marker is not followed by a loop")
            lines[index] = f"{indent}{_OPENMP_PARALLEL_FOR}"
            return lines

        if stripped.startswith("for (int64_t i0 ="):
            lines.insert(index, f"{indent}{_OPENMP_PARALLEL_FOR}")
            return lines

    raise RuntimeError("verified non-scalar kernel unexpectedly has no schedulable C loop")
