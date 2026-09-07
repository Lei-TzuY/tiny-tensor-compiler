from __future__ import annotations

from .simd_codegen import I32SSE2Plan, I32SSE2Step


def avx2_support_prelude() -> list[str]:
    """Emit x86 AVX2 compile/runtime capability detection without raising the TU baseline."""
    return [
        "#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)",
        "#define TINY_TENSOR_X86 1",
        "#else",
        "#define TINY_TENSOR_X86 0",
        "#endif",
        "",
        "#if TINY_TENSOR_X86 && defined(_MSC_VER)",
        "#include <intrin.h>",
        "#include <immintrin.h>",
        "#define TINY_TENSOR_CAN_COMPILE_AVX2 1",
        "#define TINY_TENSOR_AVX2_TARGET",
        "static void tiny_tensor_cpuid(unsigned leaf, unsigned subleaf, unsigned *a, unsigned *b, unsigned *c, unsigned *d) {",
        "    int regs[4];",
        "    __cpuidex(regs, (int)leaf, (int)subleaf);",
        "    *a = (unsigned)regs[0];",
        "    *b = (unsigned)regs[1];",
        "    *c = (unsigned)regs[2];",
        "    *d = (unsigned)regs[3];",
        "}",
        "static uint64_t tiny_tensor_xgetbv0(void) { return (uint64_t)_xgetbv(0); }",
        "#elif TINY_TENSOR_X86 && (defined(__GNUC__) || defined(__clang__))",
        "#include <cpuid.h>",
        "#include <immintrin.h>",
        "#define TINY_TENSOR_CAN_COMPILE_AVX2 1",
        '#define TINY_TENSOR_AVX2_TARGET __attribute__((target("avx2")))',
        "static void tiny_tensor_cpuid(unsigned leaf, unsigned subleaf, unsigned *a, unsigned *b, unsigned *c, unsigned *d) {",
        "    __cpuid_count(leaf, subleaf, *a, *b, *c, *d);",
        "}",
        "static uint64_t tiny_tensor_xgetbv0(void) {",
        "    unsigned eax;",
        "    unsigned edx;",
        '    __asm__ volatile ("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));',
        "    return ((uint64_t)edx << 32) | (uint64_t)eax;",
        "}",
        "#else",
        "#define TINY_TENSOR_CAN_COMPILE_AVX2 0",
        "#define TINY_TENSOR_AVX2_TARGET",
        "#endif",
        "",
        "#if TINY_TENSOR_CAN_COMPILE_AVX2",
        "static int tiny_tensor_cpu_has_avx2(void) {",
        "#if defined(TINY_TENSOR_DISABLE_AVX2)",
        "    return 0;",
        "#else",
        "    unsigned eax;",
        "    unsigned ebx;",
        "    unsigned ecx;",
        "    unsigned edx;",
        "    tiny_tensor_cpuid(0, 0, &eax, &ebx, &ecx, &edx);",
        "    if (eax < 7) return 0;",
        "    tiny_tensor_cpuid(1, 0, &eax, &ebx, &ecx, &edx);",
        "    if ((ecx & (1u << 27)) == 0 || (ecx & (1u << 28)) == 0) return 0;",
        "    if ((tiny_tensor_xgetbv0() & 0x6u) != 0x6u) return 0;",
        "    tiny_tensor_cpuid(7, 0, &eax, &ebx, &ecx, &edx);",
        "    return (ebx & (1u << 5)) != 0;",
        "#endif",
        "}",
        "TINY_TENSOR_EXPORT int tiny_tensor_runtime_has_avx2(void) {",
        "    return tiny_tensor_cpu_has_avx2();",
        "}",
        "#endif",
        "",
    ]


def emit_i32_avx2_helper(
    plan: I32SSE2Plan,
    *,
    helper_name: str,
) -> list[str]:
    """Emit one AVX2-only helper for an existing add/ReLU semantic plan."""
    parameters = ["int32_t *out"]
    parameters.extend(
        f"const int32_t *in{index}" for index, _ in enumerate(plan.loads)
    )
    parameters.append("int64_t count")
    lines = [
        "#if TINY_TENSOR_CAN_COMPILE_AVX2",
        f"static TINY_TENSOR_AVX2_TARGET void {helper_name}({', '.join(parameters)}) {{",
        "    int64_t n = 0;",
        "    for (; n + 8 <= count; n += 8) {",
    ]
    for index, (name, _buffer) in enumerate(plan.loads):
        lines.append(
            f"        __m256i {name} = _mm256_loadu_si256((const __m256i *)&in{index}[n]);"
        )
    for step in plan.steps:
        lines.extend(_emit_avx2_step(step))
    lines.extend(
        [
            f"        _mm256_storeu_si256((__m256i *)&out[n], {plan.result});",
            "    }",
            "    for (; n < count; ++n) {",
        ]
    )
    lines.extend(_emit_scalar_tail(plan, indent="        "))
    lines.extend(["    }", "}", "#endif", ""])
    return lines


def emit_i32_avx2_dispatch(
    plan: I32SSE2Plan,
    *,
    helper_name: str,
    output: int,
    count: int,
    sse2_lines: list[str],
) -> list[str]:
    """Call AVX2 only after runtime CPU/OS validation, else execute the existing SSE2 path."""
    arguments = [f"p{output}"]
    arguments.extend(f"p{buffer}" for _name, buffer in plan.loads)
    arguments.append(str(count))
    lines = [
        "        #if TINY_TENSOR_CAN_COMPILE_AVX2",
        "        if (tiny_tensor_cpu_has_avx2()) {",
        f"            {helper_name}({', '.join(arguments)});",
        "        } else",
        "        #endif",
        "        {",
    ]
    lines.extend(f"    {line}" if line else line for line in sse2_lines)
    lines.append("        }")
    return lines


def _emit_avx2_step(step: I32SSE2Step) -> list[str]:
    if step.opcode == "add":
        lhs, rhs = step.inputs
        return [f"        __m256i {step.output} = _mm256_add_epi32({lhs}, {rhs});"]

    (operand,) = step.inputs
    zero = f"zero_{step.output}"
    positive = f"positive_{step.output}"
    return [
        f"        __m256i {zero} = _mm256_setzero_si256();",
        f"        __m256i {positive} = _mm256_cmpgt_epi32({operand}, {zero});",
        f"        __m256i {step.output} = _mm256_and_si256({operand}, {positive});",
    ]


def _emit_scalar_tail(plan: I32SSE2Plan, *, indent: str) -> list[str]:
    refs = {name: f"in{index}[n]" for index, (name, _buffer) in enumerate(plan.loads)}
    lines: list[str] = []
    for step in plan.steps:
        if step.opcode == "add":
            lhs, rhs = step.inputs
            lines.append(
                f"{indent}int32_t {step.output} = ((int32_t){refs[lhs]} + (int32_t){refs[rhs]});"
            )
            refs[step.output] = step.output
            continue

        (operand,) = step.inputs
        lines.append(
            f"{indent}int32_t {step.output} = ((int32_t){refs[operand]} < 0 ? 0 : (int32_t){refs[operand]});"
        )
        refs[step.output] = step.output

    lines.append(f"{indent}out[n] = (int32_t){refs[plan.result]};")
    return lines
