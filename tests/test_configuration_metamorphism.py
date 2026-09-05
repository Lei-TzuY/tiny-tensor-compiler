from __future__ import annotations

import numpy as np
import pytest

from tiny_tensor_compiler import execute_reference
from tiny_tensor_compiler.configuration_metamorphic import (
    NATIVE_CONFIGURATIONS,
    NativeConfiguration,
    run_configuration_metamorphic_campaign,
)
from tiny_tensor_compiler.repro import load_repro_case, repro_case_sha256


def test_native_configuration_order_is_fixed_and_semantic():
    assert NATIVE_CONFIGURATIONS == (
        NativeConfiguration("serial-copied", parallel=False, borrow_inputs=False),
        NativeConfiguration("parallel-copied", parallel=True, borrow_inputs=False),
        NativeConfiguration("serial-borrowed", parallel=False, borrow_inputs=True),
        NativeConfiguration("parallel-borrowed", parallel=True, borrow_inputs=True),
    )


def test_reference_configuration_runner_is_deterministically_clean_and_ordered():
    seen: list[str] = []

    def reference_runner(configuration, module, inputs):
        seen.append(configuration.name)
        return execute_reference(module, inputs=inputs)

    first = run_configuration_metamorphic_campaign(
        start_seed=0,
        cases=4,
        configuration_runner=reference_runner,
    )
    first_seen = tuple(seen)
    seen.clear()
    second = run_configuration_metamorphic_campaign(
        start_seed=0,
        cases=4,
        configuration_runner=reference_runner,
    )

    assert first == second
    assert first.passed
    assert first.checked_cases == 4
    assert first.failure is None
    assert first_seen == tuple(configuration.name for configuration in NATIVE_CONFIGURATIONS) * 4
    assert tuple(seen) == first_seen


def test_real_native_campaign_agrees_across_all_four_configurations(tmp_path):
    result = run_configuration_metamorphic_campaign(
        start_seed=0,
        cases=4,
        cache_dir=tmp_path,
    )

    signature = None if result.failure is None else result.failure.signature
    assert result.passed, f"unexpected configuration metamorphic failure: {signature}"
    assert result.checked_cases == 4
    assert result.failure is None


def test_configuration_mismatch_shrinks_deterministically_to_one_canonical_repro():
    def wrong_parallel_borrowed(configuration, module, inputs):
        result = execute_reference(module, inputs=inputs)
        if configuration.name == "parallel-borrowed":
            array = np.asarray(result)
            return np.zeros_like(array)
        return result

    first = run_configuration_metamorphic_campaign(
        start_seed=5,
        cases=1,
        configuration_runner=wrong_parallel_borrowed,
    )
    second = run_configuration_metamorphic_campaign(
        start_seed=5,
        cases=1,
        configuration_runner=wrong_parallel_borrowed,
    )

    assert first == second
    assert not first.passed
    assert first.failure is not None
    failure = first.failure
    assert failure.seed == 5
    assert failure.baseline_configuration == "serial-copied"
    assert failure.failing_configuration == "parallel-borrowed"
    assert failure.signature.startswith(
        "configuration:serial-copied->parallel-borrowed:mismatch:"
    )
    assert failure.original_operation_count >= failure.minimized_operation_count
    assert failure.shrink_evaluations > 0
    assert repro_case_sha256(failure.minimized_repro) == repro_case_sha256(
        second.failure.minimized_repro
    )
    case = load_repro_case(failure.minimized_repro)
    assert case.module.function.name == "differential"


def test_configuration_exception_signature_excludes_unstable_message_text():
    def failing_parallel_copied(configuration, module, inputs):
        if configuration.name == "parallel-copied":
            raise RuntimeError("temporary-path C:/random/build-1234.dll should not be stable")
        return execute_reference(module, inputs=inputs)

    result = run_configuration_metamorphic_campaign(
        start_seed=0,
        cases=1,
        configuration_runner=failing_parallel_copied,
    )

    assert result.failure is not None
    assert result.failure.baseline_configuration == "serial-copied"
    assert result.failure.failing_configuration == "parallel-copied"
    assert result.failure.signature == (
        "configuration:serial-copied->parallel-copied:exception:builtins.RuntimeError"
    )
    assert "temporary-path" not in result.failure.signature


def test_campaign_configuration_fails_closed():
    with pytest.raises(TypeError, match="seed"):
        run_configuration_metamorphic_campaign(start_seed=True, cases=1)
    with pytest.raises(ValueError, match="positive"):
        run_configuration_metamorphic_campaign(start_seed=0, cases=0)
    with pytest.raises(ValueError, match="64-bit"):
        run_configuration_metamorphic_campaign(start_seed=(1 << 64) - 1, cases=2)

    def reference_runner(configuration, module, inputs):
        return execute_reference(module, inputs=inputs)

    with pytest.raises(ValueError, match="default native"):
        run_configuration_metamorphic_campaign(
            start_seed=0,
            cases=1,
            configuration_runner=reference_runner,
            compiler="cc",
        )
    with pytest.raises(ValueError, match="default native"):
        run_configuration_metamorphic_campaign(
            start_seed=0,
            cases=1,
            configuration_runner=reference_runner,
            cache_dir="cache",
        )


def test_native_configuration_validation_rejects_ambiguous_values():
    with pytest.raises(ValueError, match="non-empty"):
        NativeConfiguration("", parallel=False, borrow_inputs=False)
    with pytest.raises(TypeError, match="parallel"):
        NativeConfiguration("bad", parallel=1, borrow_inputs=False)
    with pytest.raises(TypeError, match="borrow_inputs"):
        NativeConfiguration("bad", parallel=False, borrow_inputs=1)
