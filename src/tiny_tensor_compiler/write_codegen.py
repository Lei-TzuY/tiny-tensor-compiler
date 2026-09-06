from __future__ import annotations

from .ir import DType, TensorType
from .layout import StorageLayout
from .loop_ir import LoopBinaryInto, LoopCopyInto, LoopInplaceBinary


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
    c_type = _c_type(target_type.dtype)
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
    root_offset = op.layout.offset
    pointer = f"p{op.root}" if root_offset == 0 else f"p{op.root} + {root_offset}"
    lines.append(f"    {c_type} *p{op.output} = {pointer};")
    lines.append("")
    return lines


def emit_binary_into(
    op: LoopBinaryInto,
    types: dict[int, TensorType],
    layouts: dict[int, StorageLayout],
) -> list[str]:
    """Emit one serial exact-typed binary update through a verified target view."""
    target_type = types[op.target]
    source_type = types[op.source]
    if target_type != source_type:
        raise RuntimeError("verified binary_into unexpectedly has mismatched source/target types")
    if op.operator not in {"add", "mul"}:
        raise RuntimeError("verified binary_into unexpectedly has an unsupported operator")

    target_layout = layouts[op.target]
    source_layout = layouts[op.source]
    operator = "+" if op.operator == "add" else "*"
    c_type = _c_type(target_type.dtype)
    lines = ["    {"]

    if not target_type.shape:
        destination = _root_ref(op.root, target_layout.offset, ())
        lines.append(f"        {destination} = {destination} {operator} p{op.source}[0];")
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
        lines.append(
            f"{indent}{destination} = {destination} {operator} p{op.source}[{source_offset}];"
        )
        for _ in target_type.shape:
            indent = indent[:-4]
            lines.append(f"{indent}}}")

    lines.append("    }")
    root_offset = op.layout.offset
    pointer = f"p{op.root}" if root_offset == 0 else f"p{op.root} + {root_offset}"
    lines.append(f"    {c_type} *p{op.output} = {pointer};")
    lines.append("")
    return lines


def emit_inplace_binary(
    op: LoopInplaceBinary,
    types: dict[int, TensorType],
    layouts: dict[int, StorageLayout],
) -> list[str]:
    """Emit one serial exact-typed full-root binary update and expose its fresh handle."""
    root_type = types[op.root]
    source_type = types[op.source]
    if root_type != source_type or op.type != root_type:
        raise RuntimeError("verified binary_inplace unexpectedly has mismatched types")
    if op.operator not in {"add", "mul"}:
        raise RuntimeError("verified binary_inplace unexpectedly has an unsupported operator")

    root_layout = layouts[op.root]
    source_layout = layouts[op.source]
    operator = "+" if op.operator == "add" else "*"
    c_type = _c_type(root_type.dtype)
    lines = ["    {"]
    if not root_type.shape:
        destination = _root_ref(op.root, root_layout.offset, ())
        lines.append(f"        {destination} = {destination} {operator} p{op.source}[0];")
    else:
        indent = "        "
        axes = tuple(range(len(root_type.shape)))
        for axis, bound in enumerate(root_type.shape):
            lines.append(
                f"{indent}for (int64_t i{axis} = 0; i{axis} < {bound}; ++i{axis}) {{"
            )
            indent += "    "
        destination = _root_ref(op.root, root_layout.offset, root_layout.strides)
        source_offset = _stride_offset(axes, source_layout.strides)
        lines.append(
            f"{indent}{destination} = {destination} {operator} p{op.source}[{source_offset}];"
        )
        for _ in root_type.shape:
            indent = indent[:-4]
            lines.append(f"{indent}}}")
    lines.append("    }")
    lines.append(f"    {c_type} *p{op.output} = p{op.root};")
    lines.append("")
    return lines


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


def _c_type(dtype: DType) -> str:
    return {
        DType.INT32: "int32_t",
        DType.INT64: "int64_t",
        DType.FLOAT32: "float",
        DType.FLOAT64: "double",
    }[dtype]
