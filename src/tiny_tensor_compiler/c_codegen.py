from __future__ import annotations

import math
from functools import reduce
from operator import mul
from typing import Any

import numpy as np

from .ir import DType, TensorType
from .loop_ir import IndexMap, LoopAlloc, LoopInput, LoopKernel, LoopProgram, LoopReturn

_BINARY_CHAIN_OPERATORS = {
    "chain_add_add": ("+", "+"),
    "chain_add_mul": ("+", "*"),
    "chain_mul_add": ("*", "+"),
    "chain_mul_mul": ("*", "*"),
}
_RELU_BINARY_CHAIN_OPCODES = frozenset(f"relu_{opcode}" for opcode in _BINARY_CHAIN_OPERATORS)
_BINARY_TREE_OPERATORS = {
    f"tree_{left}_{right}_{root}": (
        "+" if left == "add" else "*",
        "+" if right == "add" else "*",
        "+" if root == "add" else "*",
    )
    for left in ("add", "mul")
    for right in ("add", "mul")
    for root in ("add", "mul")
}
_RELU_BINARY_TREE_OPCODES = frozenset(f"relu_{opcode}" for opcode in _BINARY_TREE_OPERATORS)
_CHAIN_TREE_OPERATORS = {
    f"chain_tree_{inner}_{left}_{right}_{root}": (
        "+" if inner == "add" else "*",
        "+" if left == "add" else "*",
        "+" if right == "add" else "*",
        "+" if root == "add" else "*",
    )
    for inner in ("add", "mul")
    for left in ("add", "mul")
    for right in ("add", "mul")
    for root in ("add", "mul")
}


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

    if _can_emit_sse2_i32(op, types):
        lines.extend(_emit_sse2_i32(op, output_type))
        lines.append("    }")
        lines.append("")
        return lines

    linearized = _can_linearize_kernel(op, types)
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
        return _input_ref(buffer, op.input_maps[position], types[buffer])

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
    elif op.opcode in _BINARY_CHAIN_OPERATORS or op.opcode in _RELU_BINARY_CHAIN_OPCODES:
        lhs = input_ref(0)
        rhs = input_ref(1)
        tail = input_ref(2)
        c_type = _c_type(output_type.dtype)
        relu_chain = op.opcode in _RELU_BINARY_CHAIN_OPCODES
        chain_opcode = op.opcode.removeprefix("relu_")
        inner_operator, outer_operator = _BINARY_CHAIN_OPERATORS[chain_opcode]
        lines.append(
            f"{indent}{c_type} inner = (({c_type}){lhs} {inner_operator} ({c_type}){rhs});"
        )
        if relu_chain:
            zero = _zero_literal(output_type.dtype)
            lines.append(
                f"{indent}{c_type} value = (({c_type})inner {outer_operator} ({c_type}){tail});"
            )
            lines.extend(_emit_relu_assignment(output_ref, output_type.dtype, zero, indent))
        else:
            lines.append(
                f"{indent}{output_ref} = (({c_type})inner {outer_operator} ({c_type}){tail});"
            )
    elif op.opcode in _BINARY_TREE_OPERATORS or op.opcode in _RELU_BINARY_TREE_OPCODES:
        left_lhs = input_ref(0)
        left_rhs = input_ref(1)
        right_lhs = input_ref(2)
        right_rhs = input_ref(3)
        c_type = _c_type(output_type.dtype)
        relu_tree = op.opcode in _RELU_BINARY_TREE_OPCODES
        tree_opcode = op.opcode.removeprefix("relu_")
        left_operator, right_operator, root_operator = _BINARY_TREE_OPERATORS[tree_opcode]
        lines.append(
            f"{indent}{c_type} left = "
            f"(({c_type}){left_lhs} {left_operator} ({c_type}){left_rhs});"
        )
        lines.append(
            f"{indent}{c_type} right = "
            f"(({c_type}){right_lhs} {right_operator} ({c_type}){right_rhs});"
        )
        if relu_tree:
            zero = _zero_literal(output_type.dtype)
            lines.append(
                f"{indent}{c_type} value = (({c_type})left {root_operator} ({c_type})right);"
            )
            lines.extend(_emit_relu_assignment(output_ref, output_type.dtype, zero, indent))
        else:
            lines.append(
                f"{indent}{output_ref} = (({c_type})left {root_operator} ({c_type})right);"
            )
    elif op.opcode in _CHAIN_TREE_OPERATORS:
        first_lhs = input_ref(0)
        first_rhs = input_ref(1)
        left_tail = input_ref(2)
        right_lhs = input_ref(3)
        right_rhs = input_ref(4)
        c_type = _c_type(output_type.dtype)
        inner_operator, left_operator, right_operator, root_operator = _CHAIN_TREE_OPERATORS[
            op.opcode
        ]
        lines.append(
            f"{indent}{c_type} inner = "
            f"(({c_type}){first_lhs} {inner_operator} ({c_type}){first_rhs});"
        )
        lines.append(
            f"{indent}{c_type} left = "
            f"(({c_type})inner {left_operator} ({c_type}){left_tail});"
        )
        lines.append(
            f"{indent}{c_type} right = "
            f"(({c_type}){right_lhs} {right_operator} ({c_type}){right_rhs});"
        )
        lines.append(
            f"{indent}{output_ref} = (({c_type})left {root_operator} ({c_type})right);"
        )
    else:
        raise RuntimeError(f"unsupported verified loop kernel: {op.opcode}")

    for _ in range(loop_depth):
        indent = indent[:-4]
        lines.append(f"{indent}}}")
    lines.append("    }")
    lines.append("")
    return lines


def _can_emit_sse2_i32(op: LoopKernel, types: dict[int, TensorType]) -> bool:
    return (
        op.opcode in {"add", "relu", "relu_add", "chain_add_add", "relu_chain_add_add"}
        and types[op.output].dtype == DType.INT32
        and all(types[buffer].dtype == DType.INT32 for buffer in op.inputs)
        and _can_linearize_kernel(op, types)
    )


def _emit_sse2_i32(op: LoopKernel, output_type: TensorType) -> list[str]:
    if op.opcode == "relu":
        return _emit_sse2_i32_relu(op, output_type)
    if op.opcode in {"chain_add_add", "relu_chain_add_add"}:
        return _emit_sse2_i32_chain_add_add(op, output_type)

    lhs, rhs = op.inputs
    count = _element_count(output_type)
    output = op.output
    relu = op.opcode == "relu_add"
    lines = [
        "        #if TINY_TENSOR_HAS_SSE2",
        "        int64_t n = 0;",
        f"        for (; n + 4 <= {count}; n += 4) {{",
        f"            __m128i lhs = _mm_loadu_si128((const __m128i *)&p{lhs}[n]);",
        f"            __m128i rhs = _mm_loadu_si128((const __m128i *)&p{rhs}[n]);",
        "            __m128i sum = _mm_add_epi32(lhs, rhs);",
    ]
    if relu:
        lines.extend(
            [
                "            __m128i zero = _mm_setzero_si128();",
                "            __m128i positive = _mm_cmpgt_epi32(sum, zero);",
                "            __m128i relu = _mm_and_si128(sum, positive);",
                f"            _mm_storeu_si128((__m128i *)&p{output}[n], relu);",
            ]
        )
    else:
        lines.append(f"            _mm_storeu_si128((__m128i *)&p{output}[n], sum);")
    lines.extend(["        }", f"        for (; n < {count}; ++n) {{"])
    if relu:
        lines.extend(
            [
                f"            int32_t value = ((int32_t)p{lhs}[n] + (int32_t)p{rhs}[n]);",
                f"            p{output}[n] = value < 0 ? 0 : value;",
            ]
        )
    else:
        lines.append(f"            p{output}[n] = ((int32_t)p{lhs}[n] + (int32_t)p{rhs}[n]);")
    lines.extend(["        }", "        #else", "        TINY_TENSOR_VECTORIZE_LOOP"])
    lines.append(f"        for (int64_t n = 0; n < {count}; ++n) {{")
    if relu:
        lines.extend(
            [
                f"            int32_t value = ((int32_t)p{lhs}[n] + (int32_t)p{rhs}[n]);",
                f"            p{output}[n] = value < 0 ? 0 : value;",
            ]
        )
    else:
        lines.append(f"            p{output}[n] = ((int32_t)p{lhs}[n] + (int32_t)p{rhs}[n]);")
    lines.extend(["        }", "        #endif"])
    return lines


def _emit_sse2_i32_chain_add_add(op: LoopKernel, output_type: TensorType) -> list[str]:
    lhs, rhs, tail = op.inputs
    count = _element_count(output_type)
    output = op.output
    relu = op.opcode == "relu_chain_add_add"
    lines = [
        "        #if TINY_TENSOR_HAS_SSE2",
        "        int64_t n = 0;",
        f"        for (; n + 4 <= {count}; n += 4) {{",
        f"            __m128i lhs = _mm_loadu_si128((const __m128i *)&p{lhs}[n]);",
        f"            __m128i rhs = _mm_loadu_si128((const __m128i *)&p{rhs}[n]);",
        f"            __m128i tail = _mm_loadu_si128((const __m128i *)&p{tail}[n]);",
        "            __m128i inner = _mm_add_epi32(lhs, rhs);",
        "            __m128i result = _mm_add_epi32(inner, tail);",
    ]
    if relu:
        lines.extend(
            [
                "            __m128i zero = _mm_setzero_si128();",
                "            __m128i positive = _mm_cmpgt_epi32(result, zero);",
                "            __m128i relu = _mm_and_si128(result, positive);",
                f"            _mm_storeu_si128((__m128i *)&p{output}[n], relu);",
            ]
        )
    else:
        lines.append(f"            _mm_storeu_si128((__m128i *)&p{output}[n], result);")
    lines.extend(
        [
            "        }",
            f"        for (; n < {count}; ++n) {{",
            f"            int32_t inner = ((int32_t)p{lhs}[n] + (int32_t)p{rhs}[n]);",
        ]
    )
    if relu:
        lines.extend(
            [
                f"            int32_t value = ((int32_t)inner + (int32_t)p{tail}[n]);",
                f"            p{output}[n] = value < 0 ? 0 : value;",
            ]
        )
    else:
        lines.append(f"            p{output}[n] = ((int32_t)inner + (int32_t)p{tail}[n]);")
    lines.extend(
        [
            "        }",
            "        #else",
            "        TINY_TENSOR_VECTORIZE_LOOP",
            f"        for (int64_t n = 0; n < {count}; ++n) {{",
            f"            int32_t inner = ((int32_t)p{lhs}[n] + (int32_t)p{rhs}[n]);",
        ]
    )
    if relu:
        lines.extend(
            [
                f"            int32_t value = ((int32_t)inner + (int32_t)p{tail}[n]);",
                f"            p{output}[n] = value < 0 ? 0 : value;",
            ]
        )
    else:
        lines.append(f"            p{output}[n] = ((int32_t)inner + (int32_t)p{tail}[n]);")
    lines.extend(["        }", "        #endif"])
    return lines


def _emit_sse2_i32_relu(op: LoopKernel, output_type: TensorType) -> list[str]:
    (operand,) = op.inputs
    count = _element_count(output_type)
    output = op.output
    return [
        "        #if TINY_TENSOR_HAS_SSE2",
        "        int64_t n = 0;",
        f"        for (; n + 4 <= {count}; n += 4) {{",
        f"            __m128i value = _mm_loadu_si128((const __m128i *)&p{operand}[n]);",
        "            __m128i zero = _mm_setzero_si128();",
        "            __m128i positive = _mm_cmpgt_epi32(value, zero);",
        "            __m128i relu = _mm_and_si128(value, positive);",
        f"            _mm_storeu_si128((__m128i *)&p{output}[n], relu);",
        "        }",
        f"        for (; n < {count}; ++n) {{",
        f"            int32_t value = (int32_t)p{operand}[n];",
        f"            p{output}[n] = value < 0 ? 0 : value;",
        "        }",
        "        #else",
        "        TINY_TENSOR_VECTORIZE_LOOP",
        f"        for (int64_t n = 0; n < {count}; ++n) {{",
        f"            int32_t value = (int32_t)p{operand}[n];",
        f"            p{output}[n] = value < 0 ? 0 : value;",
        "        }",
        "        #endif",
    ]


def _can_linearize_kernel(op: LoopKernel, types: dict[int, TensorType]) -> bool:
    if not op.iteration_shape or _element_count(types[op.output]) == 0:
        return False

    identity = tuple(range(len(op.iteration_shape)))
    return all(
        types[buffer].shape == op.iteration_shape and index_map.axes == identity
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