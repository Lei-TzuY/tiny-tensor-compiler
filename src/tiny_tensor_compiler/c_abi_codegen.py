from __future__ import annotations

from .c_codegen import _c_type, _element_count, _emit_input, _emit_kernel, _storage_size
from .input_binding import BorrowedLoopProgram
from .ir import TensorType
from .loop_ir import LoopAlloc, LoopInput, LoopProgram, LoopReturn
from .parallel_codegen import emit_parallel_kernel


def generate_c(
    program: LoopProgram | BorrowedLoopProgram,
    *,
    parallel: bool = False,
) -> str:
    """Generate deterministic C11 with one output pointer per returned tensor."""
    types = {op.buffer: op.type for op in program.allocations}
    return_types = tuple(types[slot] for slot in program.return_slots)
    output_names = (
        ("out",)
        if len(return_types) == 1
        else tuple(f"out{index}" for index in range(len(return_types)))
    )
    parameters = [
        f"{_c_type(return_type.dtype)} *{output_name}"
        for output_name, return_type in zip(output_names, return_types, strict=True)
    ]
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

    borrowed_by_slot = {
        binding.buffer: binding for binding in getattr(program, "borrowed_inputs", ())
    }
    for alloc in program.allocations:
        binding = borrowed_by_slot.get(alloc.buffer)
        if binding is None:
            lines.append(f"    {_c_type(alloc.type.dtype)} p{alloc.buffer}[{_storage_size(alloc.type)}];")
        else:
            lines.append(
                f"    const {_c_type(alloc.type.dtype)} *p{alloc.buffer} = input{binding.index};"
            )
    if program.allocations:
        lines.append("")

    kernel_number = 0
    return_number = 0
    for op in program.operations:
        if isinstance(op, LoopAlloc):
            continue
        if isinstance(op, LoopInput):
            if op.output not in borrowed_by_slot:
                lines.extend(_emit_input(op, types[op.output]))
            continue
        if isinstance(op, LoopReturn):
            lines.extend(
                _emit_return(
                    op,
                    types[op.buffer],
                    output_names[return_number],
                )
            )
            return_number += 1
            continue
        emitter = emit_parallel_kernel if parallel else _emit_kernel
        lines.extend(emitter(op, types, kernel_number))
        kernel_number += 1

    if return_number != len(output_names):
        raise RuntimeError("verified loop IR return count changed during C generation")

    lines.append("}")
    return "\n".join(lines) + "\n"


def _emit_return(op: LoopReturn, type_: TensorType, output_name: str) -> list[str]:
    count = _element_count(type_)
    if type_.shape:
        return [
            f"    for (int64_t r = 0; r < {count}; ++r) {{",
            f"        {output_name}[r] = p{op.buffer}[r];",
            "    }",
        ]
    return [f"    {output_name}[0] = p{op.buffer}[0];"]
