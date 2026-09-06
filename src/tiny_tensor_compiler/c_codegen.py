from __future__ import annotations

import math
from functools import reduce
from operator import mul
from typing import Any

import numpy as np

from .fused_expr import FusedExpression
from .ir import DType, TensorType
from .layout import StorageLayout
from .loop_ir import (
    IndexMap,
    LoopAlloc,
    LoopInput,
    LoopKernel,
    LoopProgram,
    LoopReturn,
    LoopView,
    fused_expression_for_kernel,
)
from .reduction import ReductionOperator, ReductionPlan
from .simd_codegen import I32SSE2Plan, build_i32_sse2_plan, emit_i32_sse2_plan


def generate_c(program: LoopProgram) -> str:
    """Generate deterministic C11 source for a verified explicit loop program."""
    types = program.value_types
    layouts = program.value_layouts
    return_type = types[program.return_slot]
    parameters = [f"{_c_type(return_type.dtype)} *out"]
    parameters.extend(
        f"const {_c_type(input_type.dtype)} *input{index}"
        for index, input_type in enumerate(program.input_types)
    )

    lines = [
        "#include <math.h>",
        "#include <stdint.h>",
        "",
        "#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)",
        "#include <emmintrin.h>",
        "#define TINY_TENSOR_HAS_SSE2 1",
        "#else",
        "#define TINY_TENSOR_HAS_SSE2 0",
        "#endif",
        "",
        "#if defined(_WIN32)",
        "#define TINY_TENSOR_EXPORT __declspec(dllexport)",
        "#else",
        "#define TINY_TENSOR_EXPORT",
        "#endif",
        "",
        "#if defined(_MSC_VER)",
        "#define TINY_TENSOR_VECTORIZE_LOOP __pragma(loop(ivdep))",
        "#elif defined(__clang__)",
        '#define TINY_TENSOR_VECTORIZE_LOOP _Pragma("clang loop vectorize(enable)")',
        "#elif defined(__GNUC__)",
        '#define TINY_TENSOR_VECTORIZE_LOOP _Pragma("GCC ivdep")',
        "#else",
        "#define TINY_TENSOR_VECTORIZE_LOOP",
        "#endif",
        "",
        f"TINY_TENSOR_EXPORT void tiny_tensor_run({', '.join(parameters)}) {{",
    ]

    for alloc in program.allocations:
        lines.append(f"    {_c_type(alloc.type.dtype)} p{alloc.buffer}[{_storage_size(alloc.type)}];")
    if program.allocations:
        lines.append("")

    kernel_number = 0
    for op in program.operations:
        if isinstance(op, LoopAlloc):
            continue
        if isinstance(op, LoopInput):
            lines.extend(_emit_input(op, types[op.output]))
            continue
        if isinstance(op, LoopView):
            root = program.storage_root(op.output)
            offset = layouts[op.output].offset
            pointer = f"p{root}" if offset == 0 else f"p{root} + {offset}"
            lines.append(f"    const {_c_type(op.type.dtype)} *p{op.output} = {pointer};")
            lines.append("")
            continue
        if isinstance(op, LoopReturn):
            lines.extend(_emit_return_copy(op.buffer, types[op.buffer], layouts[op.buffer], "out"))
            continue
        lines.extend(_emit_kernel(op, types, kernel_number, layouts=layouts))
        kernel_number += 1

    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_input(op: LoopInput, type_: TensorType) -> list[str]:
    count = _element_count(type_)
    return [
        f"    for (int64_t n = 0; n < {count}; ++n) {{",
        f"        p{op.output}[n] = input{op.index}[n];",
        "    }",
        "",
    ]


def _emit_kernel(
    op: LoopKernel,
    types: dict[int, TensorType],
    kernel_number: int,
    *,
    layouts: dict[int, StorageLayout] | None = None,
) -> list[str]:
    if layouts is None:
        layouts = {buffer: StorageLayout.contiguous(type_.shape) for buffer, type_ in types.items()}
    output_type = types[op.output]
    lines = ["    {"]

    literal_name: str | None = None
    if op.opcode == "const" and op.literal is not None and op.literal.ndim != 0:
        literal_name = f"literal_{kernel_number}"
        flat = np.asarray(op.literal).reshape(-1)
        values = ", ".join(_c_literal(value, output_type.dtype) for value in flat)
        if not values:
            values = _zero_literal(output_type.dtype)
        lines.append(
            f"        static const {_c_type(output_type.dtype)} {literal_name}"
            f"[{max(1, flat.size)}] = {{{values}}};"
        )

    reduction = op.reduction
    if reduction is not None:
        if len(op.inputs) != 1:
            raise RuntimeError("verified reduction loop unexpectedly has invalid arity")
        source = op.inputs[0]
        source_type = types[source]
        if reduction.operator is ReductionOperator.ARGMAX:
            return _emit_argmax_reduction(
                op,
                reduction,
                source,
                source_type,
                output_type,
                layouts[source],
                lines,
            )
        c_type = _c_type(output_type.dtype)
        value_name = f"{reduction.opcode}_value"
        identity = _reduction_identity_literal(reduction.operator.identity_number, output_type.dtype)
        operator = reduction.operator.c_operator
        axes = reduction.axes
        if axes is None:
            source_ref = _linear_input_ref(source, source_type, layouts[source], "n")
            lines.extend(
                [
                    f"        {c_type} {value_name} = {identity};",
                    f"        for (int64_t n = 0; n < {_element_count(source_type)}; ++n) {{",
                    f"            {value_name} = (({c_type}){value_name} {operator} ({c_type}){source_ref});",
                    "        }",
                    f"        p{op.output}[0] = {value_name};",
                    "    }",
                    "",
                ]
            )
            return lines

        indent = "        "
        for output_axis, bound in enumerate(output_type.shape):
            lines.append(
                f"{indent}for (int64_t i{output_axis} = 0; "
                f"i{output_axis} < {bound}; ++i{output_axis}) {{"
            )
            indent += "    "
        output_offset = _flat_offset(
            tuple(range(len(output_type.shape))), output_type.shape
        )
        lines.append(f"{indent}{c_type} {value_name} = {identity};")

        if len(axes) == 1:
            axis = axes[0]
            source_ref = _axis_reduction_input_ref(source, source_type, layouts[source], axis)
            lines.append(
                f"{indent}for (int64_t r = 0; r < {source_type.shape[axis]}; ++r) {{"
            )
            lines.append(
                f"{indent}    {value_name} = (({c_type}){value_name} {operator} ({c_type}){source_ref});"
            )
            lines.append(f"{indent}}}")
        else:
            source_ref = _multi_axis_reduction_input_ref(
                source,
                source_type,
                layouts[source],
                axes,
            )
            for reduction_position, axis in enumerate(axes):
                lines.append(
                    f"{indent}for (int64_t r{reduction_position} = 0; "
                    f"r{reduction_position} < {source_type.shape[axis]}; ++r{reduction_position}) {{"
                )
                indent += "    "
            lines.append(
                f"{indent}{value_name} = (({c_type}){value_name} {operator} ({c_type}){source_ref});"
            )
            for _ in axes:
                indent = indent[:-4]
                lines.append(f"{indent}}}")

        lines.append(f"{indent}p{op.output}[{output_offset}] = {value_name};")
        for _ in output_type.shape:
            indent = indent[:-4]
            lines.append(f"{indent}}}")
        lines.append("    }")
        lines.append("")
        return lines

    if op.opcode == "reshape":
        if len(op.inputs) != 1:
            raise RuntimeError("verified reshape loop unexpectedly has invalid arity")
        source = op.inputs[0]
        source_ref = _linear_input_ref(source, types[source], layouts[source], "n")
        lines.extend(
            [
                "        TINY_TENSOR_VECTORIZE_LOOP",
                f"        for (int64_t n = 0; n < {_element_count(output_type)}; ++n) {{",
                f"            p{op.output}[n] = {source_ref};",
                "        }",
                "    }",
                "",
            ]
        )
        return lines

    sse2_plan = _select_i32_sse2_plan(op, types, layouts=layouts)
    if sse2_plan is not None:
        lines.extend(
            emit_i32_sse2_plan(
                sse2_plan,
                output=op.output,
                count=_element_count(output_type),
            )
        )
        lines.append("    }")
        lines.append("")
        return lines

    linearized = _can_linearize_kernel(op, types, layouts=layouts)
    indent = "        "
    if linearized:
        lines.append(f"{indent}TINY_TENSOR_VECTORIZE_LOOP")
        lines.append(
            f"{indent}for (int64_t n = 0; n < {_element_count(output_type)}; ++n) {{"
        )
        indent += "    "
        output_offset = "n"
        loop_depth = 1
    else:
        for axis, bound in enumerate(op.iteration_shape):
            lines.append(f"{indent}for (int64_t i{axis} = 0; i{axis} < {bound}; ++i{axis}) {{")
            indent += "    "
        output_offset = _flat_offset(tuple(range(len(op.iteration_shape))), op.iteration_shape)
        loop_depth = len(op.iteration_shape)

    output_ref = f"p{op.output}[{output_offset}]"

    def input_ref(position: int) -> str:
        buffer = op.inputs[position]
        if linearized:
            return f"p{buffer}[n]"
        return _input_ref(buffer, op.input_maps[position], layouts[buffer])

    if op.opcode == "const":
        if op.literal is None:
            raise RuntimeError("verified const loop unexpectedly has no literal")
        if op.literal.ndim == 0:
            rhs = _c_literal(op.literal.item(), output_type.dtype)
        else:
            if literal_name is None:
                raise RuntimeError("non-scalar const loop unexpectedly has no literal storage")
            rhs = f"{literal_name}[{output_offset}]"
        lines.append(f"{indent}{output_ref} = {rhs};")
    elif op.opcode in {"add", "mul"}:
        lhs = input_ref(0)
        rhs = input_ref(1)
        c_type = _c_type(output_type.dtype)
        operator = "+" if op.opcode == "add" else "*"
        lines.append(f"{indent}{output_ref} = (({c_type}){lhs} {operator} ({c_type}){rhs});")
    elif op.opcode == "relu":
        operand = input_ref(0)
        c_type = _c_type(output_type.dtype)
        zero = _zero_literal(output_type.dtype)
        lines.append(f"{indent}{c_type} value = ({c_type}){operand};")
        lines.extend(_emit_relu_assignment(output_ref, output_type.dtype, zero, indent))
    elif op.opcode in {"relu_add", "relu_mul"}:
        lhs = input_ref(0)
        rhs = input_ref(1)
        c_type = _c_type(output_type.dtype)
        zero = _zero_literal(output_type.dtype)
        operator = "+" if op.opcode == "relu_add" else "*"
        lines.append(f"{indent}{c_type} value = (({c_type}){lhs} {operator} ({c_type}){rhs});")
        lines.extend(_emit_relu_assignment(output_ref, output_type.dtype, zero, indent))
    else:
        expression = fused_expression_for_kernel(op)
        if expression is None:
            raise RuntimeError(f"unsupported verified loop kernel: {op.opcode}")
        lines.extend(
            _emit_fused_expression(
                expression,
                output_ref=output_ref,
                output_type=output_type,
                input_ref=input_ref,
                indent=indent,
            )
        )

    for _ in range(loop_depth):
        indent = indent[:-4]
        lines.append(f"{indent}}}")
    lines.append("    }")
    lines.append("")
    return lines


def _emit_argmax_reduction(
    op: LoopKernel,
    reduction: ReductionPlan,
    source: int,
    source_type: TensorType,
    output_type: TensorType,
    source_layout: StorageLayout,
    lines: list[str],
) -> list[str]:
    if output_type.dtype is not DType.INT64:
        raise RuntimeError("verified argmax output must use int64 indices")
    source_c_type = _c_type(source_type.dtype)
    axes = reduction.axes
    if axes is None:
        first_ref = _linear_input_ref(source, source_type, source_layout, "0")
        source_ref = _linear_input_ref(source, source_type, source_layout, "n")
        lines.extend(
            [
                f"        {source_c_type} argmax_best = ({source_c_type}){first_ref};",
                "        int64_t argmax_index = 0;",
                f"        for (int64_t n = 1; n < {_element_count(source_type)}; ++n) {{",
                f"            {source_c_type} argmax_candidate = ({source_c_type}){source_ref};",
                f"            if ({_argmax_update_condition(source_type.dtype)}) {{",
                "                argmax_best = argmax_candidate;",
                "                argmax_index = n;",
                "            }",
                "        }",
                f"        p{op.output}[0] = argmax_index;",
                "    }",
                "",
            ]
        )
        return lines

    if len(axes) != 1:
        raise RuntimeError("verified argmax unexpectedly has a multi-axis domain")
    axis = axes[0]
    indent = "        "
    for output_axis, bound in enumerate(output_type.shape):
        lines.append(
            f"{indent}for (int64_t i{output_axis} = 0; "
            f"i{output_axis} < {bound}; ++i{output_axis}) {{"
        )
        indent += "    "
    output_offset = _flat_offset(tuple(range(len(output_type.shape))), output_type.shape)
    source_ref = _axis_reduction_input_ref(source, source_type, source_layout, axis)
    lines.extend(
        [
            f"{indent}int64_t r = 0;",
            f"{indent}{source_c_type} argmax_best = ({source_c_type}){source_ref};",
            f"{indent}int64_t argmax_index = 0;",
            f"{indent}for (r = 1; r < {source_type.shape[axis]}; ++r) {{",
            f"{indent}    {source_c_type} argmax_candidate = ({source_c_type}){source_ref};",
            f"{indent}    if ({_argmax_update_condition(source_type.dtype)}) {{",
            f"{indent}        argmax_best = argmax_candidate;",
            f"{indent}        argmax_index = r;",
            f"{indent}    }}",
            f"{indent}}}",
            f"{indent}p{op.output}[{output_offset}] = argmax_index;",
        ]
    )
    for _ in output_type.shape:
        indent = indent[:-4]
        lines.append(f"{indent}}}")
    lines.extend(["    }", ""])
    return lines


def _argmax_update_condition(dtype: DType) -> str:
    if dtype in {DType.FLOAT32, DType.FLOAT64}:
        return "!isnan(argmax_best) && (isnan(argmax_candidate) || argmax_candidate > argmax_best)"
    return "argmax_candidate > argmax_best"


def _emit_fused_expression(
    expression: FusedExpression,
    *,
    output_ref: str,
    output_type: TensorType,
    input_ref: Any,
    indent: str,
) -> list[str]:
    refs = {
        name: input_ref(position)
        for position, name in enumerate(expression.input_names)
    }
    c_type = _c_type(output_type.dtype)
    lines: list[str] = []

    for step in expression.steps:
        if step.opcode == "relu":
            operand = refs[step.inputs[0]]
            if operand != "value":
                lines.append(f"{indent}{c_type} value = ({c_type}){operand};")
            lines.extend(
                _emit_relu_assignment(
                    output_ref,
                    output_type.dtype,
                    _zero_literal(output_type.dtype),
                    indent,
                )
            )
            refs[step.output] = output_ref
            continue

        lhs, rhs = step.inputs
        operator = "+" if step.opcode == "add" else "*"
        expression_text = (
            f"(({c_type}){refs[lhs]} {operator} ({c_type}){refs[rhs]})"
        )
        if step.output == expression.result:
            lines.append(f"{indent}{output_ref} = {expression_text};")
            refs[step.output] = output_ref
        else:
            lines.append(f"{indent}{c_type} {step.output} = {expression_text};")
            refs[step.output] = step.output

    return lines


def _select_i32_sse2_plan(
    op: LoopKernel,
    types: dict[int, TensorType],
    *,
    layouts: dict[int, StorageLayout] | None = None,
) -> I32SSE2Plan | None:
    plan = build_i32_sse2_plan(op)
    if plan is None:
        return None
    if types[op.output].dtype != DType.INT32:
        return None
    if any(types[buffer].dtype != DType.INT32 for buffer in op.inputs):
        return None
    if not _can_linearize_kernel(op, types, layouts=layouts):
        return None
    return plan


def _can_linearize_kernel(
    op: LoopKernel,
    types: dict[int, TensorType],
    *,
    layouts: dict[int, StorageLayout] | None = None,
) -> bool:
    if not op.iteration_shape or _element_count(types[op.output]) == 0:
        return False
    if layouts is None:
        layouts = {buffer: StorageLayout.contiguous(type_.shape) for buffer, type_ in types.items()}
    identity = tuple(range(len(op.iteration_shape)))
    return all(
        types[buffer].shape == op.iteration_shape
        and index_map.axes == identity
        and layouts[buffer].is_contiguous(types[buffer].shape)
        for buffer, index_map in zip(op.inputs, op.input_maps, strict=True)
    )


def _emit_relu_assignment(output_ref: str, dtype: DType, zero: str, indent: str) -> list[str]:
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


def _emit_return_copy(
    buffer: int,
    type_: TensorType,
    layout: StorageLayout,
    output_name: str,
) -> list[str]:
    count = _element_count(type_)
    if not type_.shape:
        return [f"    {output_name}[0] = p{buffer}[0];"]
    if layout.is_contiguous(type_.shape):
        return [
            f"    for (int64_t r = 0; r < {count}; ++r) {{",
            f"        {output_name}[r] = p{buffer}[r];",
            "    }",
        ]

    lines: list[str] = []
    indent = "    "
    for axis, bound in enumerate(type_.shape):
        lines.append(f"{indent}for (int64_t r{axis} = 0; r{axis} < {bound}; ++r{axis}) {{")
        indent += "    "
    source_offset = _stride_offset(tuple(range(len(type_.shape))), layout.strides, prefix="r")
    output_offset = _flat_offset(tuple(range(len(type_.shape))), type_.shape, prefix="r")
    lines.append(f"{indent}{output_name}[{output_offset}] = p{buffer}[{source_offset}];")
    for _ in type_.shape:
        indent = indent[:-4]
        lines.append(f"{indent}}}")
    return lines


def _input_ref(buffer: int, index_map: IndexMap, layout: StorageLayout) -> str:
    return f"p{buffer}[{_stride_offset(index_map.axes, layout.strides)}]"


def _linear_input_ref(
    buffer: int,
    type_: TensorType,
    layout: StorageLayout,
    linear_index: str,
) -> str:
    if layout.is_contiguous(type_.shape):
        return f"p{buffer}[{linear_index}]"
    terms: list[str] = []
    logical_strides = StorageLayout.contiguous(type_.shape).strides
    for axis, (dim, logical_stride, physical_stride) in enumerate(
        zip(type_.shape, logical_strides, layout.strides, strict=True)
    ):
        if dim <= 1:
            continue
        coordinate = (
            f"(({linear_index} / {logical_stride}) % {dim})"
            if logical_stride != 1
            else f"({linear_index} % {dim})"
        )
        if physical_stride == 1:
            terms.append(coordinate)
        else:
            terms.append(f"({coordinate} * {physical_stride})")
    offset = " + ".join(terms) if terms else "0"
    return f"p{buffer}[{offset}]"


def _axis_reduction_input_ref(
    buffer: int,
    type_: TensorType,
    layout: StorageLayout,
    reduction_axis: int,
) -> str:
    terms: list[str] = []
    for source_axis, (dim, stride) in enumerate(
        zip(type_.shape, layout.strides, strict=True)
    ):
        if dim <= 1:
            continue
        if source_axis == reduction_axis:
            coordinate = "r"
        else:
            output_axis = source_axis if source_axis < reduction_axis else source_axis - 1
            coordinate = f"i{output_axis}"
        if stride == 1:
            terms.append(coordinate)
        elif stride != 0:
            terms.append(f"({coordinate} * {stride})")
    offset = " + ".join(terms) if terms else "0"
    return f"p{buffer}[{offset}]"


def _multi_axis_reduction_input_ref(
    buffer: int,
    type_: TensorType,
    layout: StorageLayout,
    reduction_axes: tuple[int, ...],
) -> str:
    reduced_positions = {axis: position for position, axis in enumerate(reduction_axes)}
    output_position = 0
    terms: list[str] = []
    for source_axis, (dim, stride) in enumerate(
        zip(type_.shape, layout.strides, strict=True)
    ):
        reduction_position = reduced_positions.get(source_axis)
        if reduction_position is None:
            coordinate = f"i{output_position}"
            output_position += 1
        else:
            coordinate = f"r{reduction_position}"
        if dim <= 1:
            continue
        if stride == 1:
            terms.append(coordinate)
        elif stride != 0:
            terms.append(f"({coordinate} * {stride})")
    offset = " + ".join(terms) if terms else "0"
    return f"p{buffer}[{offset}]"


def _flat_offset(
    axes: tuple[int | None, ...],
    shape: tuple[int, ...],
    *,
    prefix: str = "i",
) -> str:
    terms: list[str] = []
    for input_axis, output_axis in enumerate(axes):
        if output_axis is None:
            continue
        stride = reduce(mul, shape[input_axis + 1 :], 1)
        index = f"{prefix}{output_axis}"
        if stride == 1:
            terms.append(index)
        else:
            terms.append(f"({index} * {stride})")
    return " + ".join(terms) if terms else "0"


def _stride_offset(
    axes: tuple[int | None, ...],
    strides: tuple[int, ...],
    *,
    prefix: str = "i",
) -> str:
    terms: list[str] = []
    for input_axis, output_axis in enumerate(axes):
        if output_axis is None:
            continue
        stride = strides[input_axis]
        index = f"{prefix}{output_axis}"
        if stride == 1:
            terms.append(index)
        else:
            terms.append(f"({index} * {stride})")
    return " + ".join(terms) if terms else "0"


def _element_count(type_: TensorType) -> int:
    return reduce(mul, type_.shape, 1)


def _storage_size(type_: TensorType) -> int:
    return max(1, _element_count(type_))


def _c_type(dtype: DType) -> str:
    return {
        DType.INT32: "int32_t",
        DType.INT64: "int64_t",
        DType.FLOAT32: "float",
        DType.FLOAT64: "double",
    }[dtype]


def _zero_literal(dtype: DType) -> str:
    if dtype == DType.FLOAT32:
        return "0.0f"
    if dtype == DType.FLOAT64:
        return "0.0"
    return "0"


def _one_literal(dtype: DType) -> str:
    if dtype == DType.FLOAT32:
        return "1.0f"
    if dtype == DType.FLOAT64:
        return "1.0"
    return "1"


def _reduction_identity_literal(identity_number: int, dtype: DType) -> str:
    return _zero_literal(dtype) if identity_number == 0 else _one_literal(dtype)


def _c_literal(value: Any, dtype: DType) -> str:
    if dtype in {DType.INT32, DType.INT64}:
        return str(int(value))

    number = float(value)
    suffix = "f" if dtype == DType.FLOAT32 else ""
    if math.isnan(number):
        return "NAN"
    if math.isinf(number):
        return ("-" if number < 0 else "") + "INFINITY"

    literal = repr(number)
    if "." not in literal and "e" not in literal.lower():
        literal += ".0"
    return literal + suffix
