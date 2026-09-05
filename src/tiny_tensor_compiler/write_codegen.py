from __future__ import annotations

from .ir import DType, TensorType
from .layout import StorageLayout
from .loop_ir import LoopCopyInto, LoopReluInto


def emit_copy_into(
    op: LoopCopyInto,
    types: dict[int, TensorType],
    layouts: dict[int, StorageLayout],
) -> list[str]:
    """Emit one serial copy into a verified target layout and expose the fresh root alias."""
    target_type = types[op.target]
    source_type = types[op.source]
    if target_type != source_type:
        raise RuntimeError("verified copy_into unexpectedly has mismatched source/target types")

    target_layout = layouts[op.target]
    source_layout = layouts[op.source]
    lines = ["    {"]

    if not target_type.shape:
        destination = _root_ref(op.root, target_layout.offset, ())
        source = f"p{op.source}[0]"
        lines.append(f"        {destination} = {source};")
    else:
        indent = "        "
        axes = tuple(range(len(target_type.shape)))
        for axis, bound in enumerate(target_type.shape):
            lines.append(
                f"{indent}for (int64_t i{axis} = 0; i{axis} < {bound}; ++i{axis}) {{"
            )
            indent += "    "
        destination = _root_ref(op.root, target_layout.offset, target_layout.strides)
        source_offset = _stride_offset(axes, source_layout.strides)
        lines.append(f"{indent}{destination} = p{op.source}[{source_offset}];")
        for _ in target_type.shape:
            indent = indent[:-4]
            lines.append(f"{indent}}}")

    lines.append("    }")
    lines.extend(_fresh_root_pointer(op.output, op.root, op.layout, op.type.dtype))
    return lines


def emit_relu_into(
    op: LoopReluInto,
    types: dict[int, TensorType],
    layouts: dict[int, StorageLayout],
) -> list[str]:
    """Emit one serial target-local ReLU mutation and expose the fresh root generation."""
    target_type = types[op.target]
    target_layout = layouts[op.target]
    dtype = target_type.dtype
    zero = _zero_literal(dtype)
    lines = ["    {"]

    if not target_type.shape:
        destination = _root_ref(op.root, target_layout.offset, ())
        lines.append(f"        {_c_type(dtype)} value = {destination};")
        lines.extend(_emit_relu_assignment(destination, dtype, zero, "        "))
    else:
        indent = "        "
        for axis, bound in enumerate(target_type.shape):
            lines.append(
                f"{indent}for (int64_t i{axis} = 0; i{axis} < {bound}; ++i{axis}) {{"
            )
            indent += "    "
        destination = _root_ref(op.root, target_layout.offset, target_layout.strides)
        lines.append(f"{indent}{_c_type(dtype)} value = {destination};")
        lines.extend(_emit_relu_assignment(destination, dtype, zero, indent))
        for _ in target_type.shape:
            indent = indent[:-4]
            lines.append(f"{indent}}}")

    lines.append("    }")
    lines.extend(_fresh_root_pointer(op.output, op.root, op.layout, op.type.dtype))
    return lines


def _fresh_root_pointer(
    output: int,
    root: int,
    layout: StorageLayout,
    dtype: DType,
) -> list[str]:
    root_offset = layout.offset
    pointer = f"p{root}" if root_offset == 0 else f"p{root} + {root_offset}"
    return [f"    {_c_type(dtype)} *p{output} = {pointer};", ""]


def _emit_relu_assignment(
    output_ref: str,
    dtype: DType,
    zero: str,
    indent: str,
) -> list[str]:
    if dtype in {DType.FLOAT32, DType.FLOAT64}:
        absolute = "fabsf" if dtype == DType.FLOAT32 else "fabs"
        return [
            f"{indent}if (isnan(value)) {{",
            f"{indent}    {output_ref} = value;",
            f"{indent}}} else if (value == {zero}) {{",
            f"{indent}    {output_ref} = {absolute}(value);",
            f"{indent}}} else if (value < {zero}) {{",
            f"{indent}    {output_ref} = {zero};",
            f"{indent}}} else {{",
            f"{indent}    {output_ref} = value;",
            f"{indent}}}",
        ]
    return [f"{indent}{output_ref} = value < {zero} ? {zero} : value;"]


def _root_ref(root: int, base_offset: int, strides: tuple[int, ...]) -> str:
    offset = _stride_offset(tuple(range(len(strides))), strides, base_offset=base_offset)
    return f"p{root}[{offset}]"


def _stride_offset(
    axes: tuple[int, ...],
    strides: tuple[int, ...],
    *,
    base_offset: int = 0,
) -> str:
    terms: list[str] = []
    if base_offset:
        terms.append(str(base_offset))
    for axis, stride in zip(axes, strides, strict=True):
        index = f"i{axis}"
        if stride == 1:
            terms.append(index)
        else:
            terms.append(f"({index} * {stride})")
    return " + ".join(terms) if terms else "0"


def _zero_literal(dtype: DType) -> str:
    if dtype == DType.FLOAT32:
        return "0.0f"
    if dtype == DType.FLOAT64:
        return "0.0"
    return "0"


def _c_type(dtype: DType) -> str:
    return {
        DType.INT32: "int32_t",
        DType.INT64: "int64_t",
        DType.FLOAT32: "float",
        DType.FLOAT64: "double",
    }[dtype]
