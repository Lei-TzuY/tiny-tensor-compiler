from __future__ import annotations

import shutil
import sys

import numpy as np
import pytest

from tiny_tensor_compiler import execute_reference
from tiny_tensor_compiler.cross_compiler_metamorphic import (
    CROSS_COMPILER_CONFIGURATIONS,
    CompilerConfiguration,
    run_cross_compiler_metamorphic_campaign,
)
from tiny_tensor_compiler.repro import load_repro_case, repro_case_sha256


def test_default_cross_compiler_order_is_gcc_then_clang():
    assert CROSS_COMPILER_CONFIGURATIONS == (
        CompilerConfiguration("gcc", "gcc"),
        CompilerConfiguration("clang", "clang"),
    )


def test_reference_runner_is_deterministically_clean_and_ordered():
    seen: list[str] = []

    def reference_runner(configuration, module, inputs):
        seen.append(configuration.name)
        return execute_reference(module, inputs=inputs)

    first = run_cross_compiler_metamorphic_campaign(
        start_seed=0,
        cases=4,
        compiler_runner=reference_runner,
    )
    first_seen = tuple(seen)
    seen.clear()
    second = run_cross_compiler_metamorphic_campaign(
        start_seed=0,
        cases=4,
        compiler_runner=reference_runner,
    )

    assert first == second
    assert first.passed
    assert first.checked_cases == 4
    assert first.failure is None
    assert first_seen == tuple(configuration.name for configuration in CROSS_COMPILER_CONFIGURATIONS) * 4
    assert tuple(seen) == first_seen


@pytest.mark.skipif(sys.platform == "win32", reason="GCC/Clang same-host evidence is verified on Ubuntu CI")
def test_real_gcc_clang_campaign_executes_both_toolchains(tmp_path):
    gcc = shutil.which("gcc")
    clang = shutil.which("clang")
    assert gcc is not None, "Ubuntu cross-compiler verification requires executable gcc"
    assert clang is not None, "Ubuntu cross-compiler verification requires executable clang"

    configurations = (
        CompilerConfiguration("gcc", gcc),
        CompilerConfiguration("clang", clang),
    )
    result = run_cross_compiler_metamorphic_campaign(
        start_seed=0,
        cases=4,
        configurations=configurations,
        cache_dir=tmp_path,
    )

    signature = None if result.failure is None else result.failure.signature
    assert result.passed, f"unexpected GCC/Clang divergence: {signature}"
    assert result.checked_cases == 4
    assert result.failure is None


def test_compiler_mismatch_shrinks_deterministically_to_canonical_repro():
    def wrong_clang(configuration, module, inputs):
        result = execute_reference(module, inputs=inputs)
        if configuration.name == "clang":
            array = np.asarray(result)
            return np.zeros_like(array)
        return result

    first = run_cross_compiler_metamorphic_campaign(
        start_seed=5,
        cases=1,
        compiler_runner=wrong_clang,
    )
    second = run_cross_compiler_metamorphic_campaign(
        start_seed=5,
        cases=1,
        compiler_runner=wrong_clang,
    )

    assert first == second
    assert not first.passed
    assert first.failure is not None
    failure = first.failure
    assert failure.seed == 5
    assert failure.baseline_compiler == "gcc"
    assert failure.failing_compiler == "clang"
    assert failure.signature.startswith("compiler:gcc->clang:mismatch:")
    assert failure.original_operation_count >= failure.minimized_operation_count
    assert failure.shrink_evaluations > 0
    assert second.failure is not None
    assert repro_case_sha256(failure.minimized_repro) == repro_case_sha256(
        second.failure.minimized_repro
    )
    case = load_repro_case(failure.minimized_repro)
    assert case.module.function.name == "differential"


def test_compiler_exception_signature_excludes_unstable_message_text():
    def failing_clang(configuration, module, inputs):
        if configuration.name == "clang":
            raise RuntimeError("temporary-path /tmp/build-1234/lib.so should not be stable")
        return execute_reference(module, inputs=inputs)

    result = run_cross_compiler_metamorphic_campaign(
        start_seed=0,
        cases=1,
        compiler_runner=failing_clang,
    )

    assert result.failure is not None
    assert result.failure.baseline_compiler == "gcc"
    assert result.failure.failing_compiler == "clang"
    assert result.failure.signature == "compiler:gcc->clang:exception:builtins.RuntimeError"
    assert "temporary-path" not in result.failure.signature


def test_cross_compiler_campaign_fails_closed():
    with pytest.raises(TypeError, match="seed"):
        run_cross_compiler_metamorphic_campaign(start_seed=True, cases=1)
    with pytest.raises(ValueError, match="positive"):
        run_cross_compiler_metamorphic_campaign(start_seed=0, cases=0)
    with pytest.raises(ValueError, match="64-bit"):
        run_cross_compiler_metamorphic_campaign(start_seed=(1 << 64) - 1, cases=2)

    def reference_runner(configuration, module, inputs):
        return execute_reference(module, inputs=inputs)

    with pytest.raises(ValueError, match="default native"):
        run_cross_compiler_metamorphic_campaign(
            start_seed=0,
            cases=1,
            compiler_runner=reference_runner,
            cache_dir="cache",
        )

    with pytest.raises(ValueError, match="exactly two"):
        run_cross_compiler_metamorphic_campaign(
            start_seed=0,
            cases=1,
            configurations=(CompilerConfiguration("gcc", "gcc"),),  # type: ignore[arg-type]
            compiler_runner=reference_runner,
        )
    with pytest.raises(ValueError, match="names must be distinct"):
        run_cross_compiler_metamorphic_campaign(
            start_seed=0,
            cases=1,
            configurations=(
                CompilerConfiguration("same", "gcc"),
                CompilerConfiguration("same", "clang"),
            ),
            compiler_runner=reference_runner,
        )
    with pytest.raises(ValueError, match="commands must be distinct"):
        run_cross_compiler_metamorphic_campaign(
            start_seed=0,
            cases=1,
            configurations=(
                CompilerConfiguration("gcc-a", "gcc"),
                CompilerConfiguration("gcc-b", "gcc"),
            ),
            compiler_runner=reference_runner,
        )


def test_compiler_configuration_validation_rejects_ambiguous_values():
    with pytest.raises(ValueError, match="non-empty"):
        CompilerConfiguration("", "gcc")
    with pytest.raises(ValueError, match="command"):
        CompilerConfiguration("gcc", "")
