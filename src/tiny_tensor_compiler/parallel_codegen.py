from __future__ import annotations

from dataclasses import replace

from .c_codegen import _c_type, _element_count, _emit_kernel, _select_i32_sse2_plan
from .effect_schedule import ParallelEffectGroup
from .ir import TensorType
from .layout import StorageLayout
from .loop_ir import (
    LoopBinaryInto,
    LoopCopyInto,
    LoopInplaceBinary,
    LoopKernel,
    LoopProgram,
    LoopView,
    _layout_is_non_overlapping,
)
from .write_codegen import (
    emit_binary_into,
    emit_copy_into,
    emit_effect_result_alias,
    emit_inplace_binary,
)

_OPENMP_PARALLEL_FOR = "#pragma omp parallel for schedule(static)"


def emit_parallel_kernel(
    op: LoopKernel,
    types: dict[int, TensorType],
    kernel_number: int,
    *,
    layouts: dict[int, StorageLayout] | None = None,
) -> list[str]:
    lines = _emit_kernel(op, types, kernel_number, layouts=layouts)
    output_type = types[op.output]
    if _select_i32_sse2_plan(op, types, layouts=layouts) is not None:
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


def emit_parallel_copy_into(
    op: LoopCopyInto,
    types: dict[int, TensorType],
    layouts: dict[int, StorageLayout],
) -> list[str]:
    lines = emit_copy_into(op, types, layouts)
    target_type = types[op.target]
    if not _layout_is_non_overlapping(target_type.shape, layouts[op.target]):
        return lines
    return _parallelize_effect_outer_loop(lines, target_type, effect_name="copy_into")


def emit_parallel_binary_into(
    op: LoopBinaryInto,
    types: dict[int, TensorType],
    layouts: dict[int, StorageLayout],
) -> list[str]:
    lines = emit_binary_into(op, types, layouts)
    return _parallelize_effect_outer_loop(lines, types[op.target], effect_name="binary_into")


def emit_parallel_effect_group(
    group: ParallelEffectGroup,
    program: LoopProgram,
    types: dict[int, TensorType],
    layouts: dict[int, StorageLayout],
) -> list[str]:
    """Emit region-independent effects as one barriered OpenMP sections region."""
    if len(group.effects) < 2:
        raise ValueError("parallel effect sections require at least two independent effects")

    transparent_by_output = {view.output: view for view in group.transparent_views}
    lines = ["    #pragma omp parallel sections", "    {"]
    for op in group.effects:
        lines.extend(("        #pragma omp section", "        {"))
        source_view = transparent_by_output.get(op.source)
        if source_view is not None:
            lines.extend(
                f"        {line}" for line in emit_view_alias(source_view, program, layouts) if line
            )
        canonical = replace(op, root=program.storage_root(op.root))
        body = _emit_serial_effect_body(canonical, types, layouts)
        lines.extend(f"        {line}" for line in body if line)
        lines.append("        }")
    lines.append("    }")

    effects_by_index = dict(zip(group.operation_indices, group.effects, strict=True))
    views_by_index = dict(
        zip(group.transparent_indices, group.transparent_views, strict=True)
    )
    for index in group.consumed_indices:
        effect = effects_by_index.get(index)
        if effect is not None:
            lines.extend(emit_effect_result_alias(effect))
            continue
        lines.extend(emit_view_alias(views_by_index[index], program, layouts))
    return lines


def emit_view_alias(
    op: LoopView,
    program: LoopProgram,
    layouts: dict[int, StorageLayout],
) -> list[str]:
    """Expose one logical view pointer directly from its canonical storage root."""
    root = program.storage_root(op.output)
    offset = layouts[op.output].offset
    pointer = f"p{root}" if offset == 0 else f"p{root} + {offset}"
    return [f"    const {_c_type(op.type.dtype)} *p{op.output} = {pointer};", ""]


def _emit_serial_effect_body(
    op: LoopCopyInto | LoopBinaryInto | LoopInplaceBinary,
    types: dict[int, TensorType],
    layouts: dict[int, StorageLayout],
) -> list[str]:
    if isinstance(op, LoopCopyInto):
        return emit_copy_into(op, types, layouts, expose_output=False)
    if isinstance(op, LoopBinaryInto):
        return emit_binary_into(op, types, layouts, expose_output=False)
    if isinstance(op, LoopInplaceBinary):
        return emit_inplace_binary(op, types, layouts, expose_output=False)
    raise TypeError("unsupported write effect in parallel sections")


def _parallelize_effect_outer_loop(
    lines: list[str],
    target_type: TensorType,
    *,
    effect_name: str,
) -> list[str]:
    if not target_type.shape or _element_count(target_type) == 0:
        return lines
    for index, line in enumerate(lines):
        if not line.strip().startswith("for (int64_t i0 ="):
            continue
        indent = _indent_of(line)
        _externalize_openmp_induction_variable(lines, index, "i0")
        lines.insert(index, f"{indent}{_OPENMP_PARALLEL_FOR}")
        lines.insert(index, f"{indent}int64_t i0;")
        return lines
    raise RuntimeError(f"verified {effect_name} unexpectedly has no schedulable target loop")


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
