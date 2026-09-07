import numpy as np
import pytest

from tiny_tensor_compiler.frontend import GraphBuilder
from tiny_tensor_compiler.repro import (
    ReproMismatchError,
    capture_repro_case,
    load_repro_case,
    replay_repro_case,
)
from tiny_tensor_compiler.repro_minimizer import (
    ReproMinimizationError,
    minimize_native_mismatch,
    minimize_repro_case,
)


def _module(shape=(8,)):
    builder = GraphBuilder()
    value = builder.input(shape, dtype="int32")
    return builder.finish(value + value)


def _document(values):
    array = np.asarray(values, dtype=np.int32)
    return capture_repro_case(_module(array.shape), (array,))


def test_minimizer_deterministically_reduces_nonzero_runtime_support():
    document = _document(np.arange(1, 9, dtype=np.int32))

    def still_interesting(candidate):
        case = load_repro_case(candidate)
        return np.count_nonzero(case.inputs[0]) >= 2

    first = minimize_repro_case(document, still_interesting)
    second = minimize_repro_case(document, still_interesting)

    assert first == second
    assert first.changed is True
    assert first.exhausted is False
    assert first.original_nonzero == 8
    assert first.minimized_nonzero == 2
    minimized = load_repro_case(first.document)
    assert minimized.inputs[0].tolist() == [0, 0, 0, 0, 0, 0, 7, 8]
    assert minimized.expected_outputs[0].tolist() == [0, 0, 0, 0, 0, 0, 14, 16]
    replay_repro_case(first.document, backend="reference")


def test_minimizer_preserves_shapes_dtypes_and_module_digest():
    document = _document([4, 3, 2, 1])
    original = load_repro_case(document)

    result = minimize_repro_case(
        document,
        lambda candidate: np.count_nonzero(load_repro_case(candidate).inputs[0]) >= 1,
    )
    minimized = load_repro_case(result.document)

    assert minimized.module_digest == original.module_digest
    assert minimized.inputs[0].shape == original.inputs[0].shape
    assert minimized.inputs[0].dtype == original.inputs[0].dtype
    assert minimized.expected_outputs[0].shape == original.expected_outputs[0].shape
    assert minimized.expected_outputs[0].dtype == original.expected_outputs[0].dtype
    assert np.count_nonzero(minimized.inputs[0]) == 1
    replay_repro_case(result.document, backend="reference")


def test_minimizer_processes_multiple_inputs_in_stable_declaration_order():
    builder = GraphBuilder()
    lhs = builder.input((4,), dtype="int32")
    rhs = builder.input((4,), dtype="int32")
    module = builder.finish(lhs + rhs)
    document = capture_repro_case(
        module,
        (
            np.array([1, 2, 3, 4], dtype=np.int32),
            np.array([5, 6, 7, 8], dtype=np.int32),
        ),
    )

    result = minimize_repro_case(
        document,
        lambda candidate: sum(
            np.count_nonzero(value) for value in load_repro_case(candidate).inputs
        )
        >= 1,
    )
    minimized = load_repro_case(result.document)

    assert minimized.inputs[0].tolist() == [0, 0, 0, 0]
    assert minimized.inputs[1].tolist() == [0, 0, 0, 8]
    assert result.minimized_nonzero == 1


def test_evaluation_budget_returns_best_valid_candidate_deterministically():
    document = _document(np.arange(1, 9, dtype=np.int32))

    result = minimize_repro_case(
        document,
        lambda candidate: np.count_nonzero(load_repro_case(candidate).inputs[0]) >= 1,
        max_evaluations=2,
    )

    assert result.evaluations == 2
    assert result.exhausted is True
    assert result.changed is True
    assert load_repro_case(result.document).inputs[0].tolist() == [1, 2, 3, 4, 0, 0, 0, 0]


def test_original_case_must_satisfy_predicate_and_predicate_must_return_bool():
    document = _document([1, 2, 3, 4])

    with pytest.raises(ReproMinimizationError, match="original repro does not satisfy"):
        minimize_repro_case(document, lambda candidate: False)

    with pytest.raises(TypeError, match="predicate must return bool"):
        minimize_repro_case(document, lambda candidate: np.bool_(True))


def test_minimizer_validates_evaluation_budget():
    document = _document([1, 2, 3, 4])

    for invalid in (0, -1, True, 1.5):
        with pytest.raises((TypeError, ValueError)):
            minimize_repro_case(document, lambda candidate: True, max_evaluations=invalid)


def test_native_mismatch_adapter_reduces_only_while_mismatch_reproduces(monkeypatch):
    import tiny_tensor_compiler.repro_minimizer as minimizer

    document = _document(np.arange(1, 9, dtype=np.int32))
    calls = []

    def fake_replay(candidate, *, backend, compiler=None, cache_dir=None, parallel=False):
        assert backend == "native"
        case = load_repro_case(candidate)
        support = int(np.count_nonzero(case.inputs[0]))
        calls.append((support, compiler, cache_dir, parallel))
        if support >= 2:
            raise ReproMismatchError("simulated native divergence")
        return case.expected_outputs[0]

    monkeypatch.setattr(minimizer, "replay_repro_case", fake_replay)

    result = minimize_native_mismatch(
        document,
        compiler="fake-cc",
        cache_dir="cache",
        parallel=True,
    )

    assert result.minimized_nonzero == 2
    assert calls
    assert all(call[1:] == ("fake-cc", "cache", True) for call in calls)


def test_native_mismatch_adapter_rejects_non_reproducing_original(monkeypatch):
    import tiny_tensor_compiler.repro_minimizer as minimizer

    document = _document([1, 2, 3, 4])
    monkeypatch.setattr(
        minimizer,
        "replay_repro_case",
        lambda *args, **kwargs: load_repro_case(args[0]).expected_outputs[0],
    )

    with pytest.raises(ReproMinimizationError, match="original repro does not satisfy"):
        minimize_native_mismatch(document)
