from __future__ import annotations

from .avx2_codegen import (
    avx2_support_prelude,
    emit_i32_avx2_dispatch,
    emit_i32_avx2_helper,
)
from .c_codegen import (
    _c_type,
    _element_count,
    _emit_input,
    _emit_kernel,
    _emit_return_copy,
    _select_i32_sse2_plan,
    _storage_size,
)
from .input_binding import BorrowedLoopProgram
from .loop_ir import (
    LoopAlloc,
    LoopBinaryInto,
    LoopCopyInto,
    LoopInplaceBinary,
    LoopInput,
    LoopKernel,
    LoopProgram,
    LoopReturn,
    LoopView,
)
from .parallel_codegen import emit_parallel_binary_into, emit_parallel_kernel
from .simd_codegen import I32SSE2Plan, emit_i32_sse2_plan
from .write_codegen import emit_binary_into, emit_copy_into, emit_inplace_binary


def generate_c(
    program: LoopProgram | BorrowedLoopProgram,
    *,
    parallel: bool = False,
) -> str:
    """Generate deterministic C11 with one output pointer per returned tensor."""
    types = program.value_types
    layouts = program.value_layouts
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

    avx2_plans: dict[int, I32SSE2Plan] = {}
    if not parallel:
        for kernel_number, kernel in enumerate(
            op for op in program.operations if isinstance(op, LoopKernel)
        ):
            plan = _select_i32_sse2_plan(kernel, types, layouts=layouts)
            if plan is not None:
                avx2_plans[kernel_number] = plan

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
    ]
    if avx2_plans:
        lines.extend(avx2_support_prelude())
    lines.extend(
        [
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
        ]
    )
    for kernel_number, plan in avx2_plans.items():
        lines.extend(
            emit_i32_avx2_helper(
                plan,
                helper_name=f"tiny_tensor_avx2_kernel_{kernel_number}",
            )
        )
    lines.append(f"TINY_TENSOR_EXPORT void tiny_tensor_run({', '.join(parameters)}) {{")

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
        if isinstance(op, LoopView):
            root = program.storage_root(op.output)
            offset = layouts[op.output].offset
            pointer = f"p{root}" if offset == 0 else f"p{root} + {offset}"
            lines.append(f"    const {_c_type(op.type.dtype)} *p{op.output} = {pointer};")
            lines.append("")
            continue
        if isinstance(op, LoopCopyInto):
            lines.extend(emit_copy_into(op, types, layouts))
            continue
        if isinstance(op, LoopBinaryInto):
            emitter = emit_parallel_binary_into if parallel else emit_binary_into
            lines.extend(emitter(op, types, layouts))
            continue
        if isinstance(op, LoopInplaceBinary):
            lines.extend(emit_inplace_binary(op, types, layouts))
            continue
        if isinstance(op, LoopReturn):
            lines.extend(
                _emit_return_copy(
                    op.buffer,
                    types[op.buffer],
                    layouts[op.buffer],
                    output_names[return_number],
                )
            )
            return_number += 1
            continue

        plan = avx2_plans.get(kernel_number)
        if plan is not None:
            lines.append("    {")
            count = _element_count(types[op.output])
            lines.extend(
                emit_i32_avx2_dispatch(
                    plan,
                    helper_name=f"tiny_tensor_avx2_kernel_{kernel_number}",
                    output=op.output,
                    count=count,
                    sse2_lines=emit_i32_sse2_plan(
                        plan,
                        output=op.output,
                        count=count,
                    ),
                )
            )
            lines.extend(["    }", ""])
        else:
            emitter = emit_parallel_kernel if parallel else _emit_kernel
            lines.extend(emitter(op, types, kernel_number, layouts=layouts))
        kernel_number += 1

    if return_number != len(output_names):
        raise RuntimeError("verified loop IR return count changed during C generation")

    lines.append("}")
    return "\n".join(lines) + "\n"