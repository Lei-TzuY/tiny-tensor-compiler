from __future__ import annotations

import math
from functools import reduce
from operator import mul
from typing import Any

import numpy as np

from .ir import DType, TensorType
from .loop_ir import IndexMap, LoopAlloc, LoopInput, LoopKernel, LoopProgram, LoopReturn


def generate_c(program: LoopProgram) -> str:
    """Generate deterministic C11 source for a verified explicit loop program."""
    types = {op.buffer: op.type for op in program.allocations}
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
        "#if defined(_WIN32)",
        "#define TINY_TENSOR_EXPORT __declspec(dllexport)",
        "#else",
        "#define TINY_TENSOR_EXPORT",
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
        if isinstance(op, LoopReturn):
            lines.extend(_emit_return(op, types[op.buffer]))
            continue
        lines.extend(_emit_kernel(op, types, kernel_number))
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
) -> list[str]:
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

    indent = "        "
    for axis, bound in enumerate(op.iteration_shape):
        lines.append(f"{indent}for (int64_t i{axis} = 0; i{axis} < {bound}; ++i{axis}) {{")
        indent += "    "

    output_offset = _flat_offset(tuple(range(len(op.iteration_shape))), op.iteration_shape)
    output_ref = f"p{op.output}[{output_offset}]"

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
        lhs = _input_ref(op.inputs[0], op.input_maps[0], types[op.inputs[0]])
        rhs = _input_ref(op.inputs[1], op.input_maps[1], types[op.inputs[1]])
        c_type = _c_type(output_type.dtype)
        operator = "+" if op.opcode == "add" else "*"
        lines.append(f"{indent}{output_ref} = (({c_type}){lhs} {operator} ({c_type}){rhs});")
    elif op.opcode == "relu":
        operand = _input_ref(op.inputs[0], op.input_maps[0], types[op.inputs[0]])
        c_type = _c_type(output_type.dtype)
        zero = _zero_literal(output_type.dtype)
        lines.append(f"{indent}{c_type} value = ({c_type}){operand};")
        lines.extend(_emit_relu_assignment(output_ref, output_type.dtype, zero, indent))
    elif op.opcode in {"relu_add", "relu_mul"}:
        lhs = _input_ref(op.inputs[0], op.input_maps[0], types[op.inputs[0]])
        rhs = _input_ref(op.inputs[1], op.input_maps[1], types[op.inputs[1]])
        c_type = _c_type(output_type.dtype)
        zero = _zero_literal(output_type.dtype)
        operator = "+" if op.opcode == "relu_add" else "*"
        lines.append(f"{indent}{c_type} value = (({c_type}){lhs} {operator} ({c_type}){rhs});")
        lines.extend(_emit_relu_assignment(output_ref, output_type.dtype, zero, indent))
    else:
        raise RuntimeError(f"unsupported verified loop kernel: {op.opcode}")

    for _ in op.iteration_shape:
        indent = indent[:-4]
        lines.append(f"{indent}}}")
    lines.append("    }")
    lines.append("")
    return lines


def _emit_relu_assignment(output_ref: str, dtype: DType, zero: str, indent: str) -> list[str]:
    if dtype in {DType.FLOAT32, DType.FLOAT64}:
        return [
            (
                f"{indent}{output_ref} = isnan(value) ? value : "
                f"(value <= {zero} ? {zero} : value);"
            )
        ]
    return [f"{indent}{output_ref} = value < {zero} ? {zero} : value;"]


def _emit_return(op: LoopReturn, type_: TensorType) -> list[str]:
    count = _element_count(type_)
    if type_.shape:
        return [
            f"    for (int64_t r = 0; r < {count}; ++r) {{",
            f"        out[r] = p{op.buffer}[r];",
            "    }",
        ]
    return [f"    out[0] = p{op.buffer}[0];"]


def _input_ref(buffer: int, index_map: IndexMap, type_: TensorType) -> str:
    return f"p{buffer}[{_flat_offset(index_map.axes, type_.shape)}]"


def _flat_offset(axes: tuple[int | None, ...], shape: tuple[int, ...]) -> str:
    terms: list[str] = []
    for input_axis, output_axis in enumerate(axes):
        if output_axis is None:
            continue
        stride = reduce(mul, shape[input_axis + 1 :], 1)
        if stride == 1:
            terms.append(f"i{output_axis}")
        else:
            terms.append(f"(i{output_axis} * {stride})")
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
