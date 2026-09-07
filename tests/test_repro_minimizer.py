import sys

import pytest

from tiny_tensor_compiler.frontend import GraphBuilder
from tiny_tensor_compiler.repro_minimizer import (
    ReproMinimizationError,
    main,
    minimize_operations,
    minimize_return_roots,
)
from tiny_tensor_compiler.serialization import deserialize_module, serialize_module


def _multi_output_module():
    builder = GraphBuilder()
    first = builder.input((2,), dtype="float32").relu()
    target = builder.input((3,), dtype="float32").relu()
    third = builder.input((4,), dtype="float32").relu()
    return builder.finish((first, target, third))


def _returned_shapes(module):
    return tuple(value.type.shape for value in module.function.ops[-1].operands)


def _target_shape_predicate(module):
    return (3,) in _returned_shapes(module)


def test_minimizer_finds_deterministic_one_minimal_return_root():
    first = minimize_return_roots(_multi_output_module(), _target_shape_predicate)
    second = minimize_return_roots(_multi_output_module(), _target_shape_predicate)

    assert first.module_json == second.module_json
    assert first.original_return_count == 3
    assert first.minimized_return_count == 1
    assert first.attempts == 3
    assert first.accepted_reductions == 2

    minimized = deserialize_module(first.module_json)
    assert _returned_shapes(minimized) == ((3,),)
    assert serialize_module(minimized) == first.module_json
    assert [op.attrs["index"] for op in minimized.function.ops if op.opcode == "input"] == [0, 1, 2]
    assert [op.opcode for op in minimized.function.ops].count("relu") == 1


def test_minimizer_requires_initial_reproduction_and_boolean_predicate():
    with pytest.raises(ReproMinimizationError, match="initial module does not satisfy"):
        minimize_return_roots(_multi_output_module(), lambda module: False)

    with pytest.raises(TypeError, match="predicate must return a bool"):
        minimize_return_roots(_multi_output_module(), lambda module: 1)


def test_single_return_is_already_one_minimal():
    builder = GraphBuilder()
    value = builder.input((3,), dtype="float32")
    module = builder.finish(value.relu())

    result = minimize_return_roots(module, lambda candidate: True)

    assert result.original_return_count == 1
    assert result.minimized_return_count == 1
    assert result.attempts == 0
    assert result.accepted_reductions == 0
    assert result.module_json == serialize_module(module)


def test_minimizer_fails_closed_on_effectful_module():
    builder = GraphBuilder()
    lhs = builder.input((4,), dtype="float32")
    rhs = builder.input((4,), dtype="float32")
    source = builder.input((4,), dtype="float32")
    root = lhs + rhs
    updated = root.copy_into(root, source)
    module = builder.finish(updated)

    with pytest.raises(ReproMinimizationError, match="effectful opcode"):
        minimize_return_roots(module, lambda candidate: True)


def test_operation_minimizer_is_deterministic_and_reduces_exact_typed_chain():
    builder = GraphBuilder()
    lhs = builder.input((3,), dtype="float32")
    rhs = builder.input((3,), dtype="float32")
    module = builder.finish((lhs + rhs).relu())

    first = minimize_operations(module, lambda candidate: True)
    second = minimize_operations(module, lambda candidate: True)

    assert first.module_json == second.module_json
    assert first.original_operation_count == 2
    assert first.minimized_operation_count == 0
    assert first.attempts == 2
    assert first.accepted_reductions == 2

    minimized = first.module
    assert [op.opcode for op in minimized.function.ops] == ["input", "input", "return"]
    assert [op.attrs["index"] for op in minimized.function.ops if op.opcode == "input"] == [0, 1]
    assert minimized.function.ops[-1].operands[0] is minimized.function.ops[1].results[0]


def test_operation_minimizer_obeys_predicate_and_reaches_one_minimal_result():
    builder = GraphBuilder()
    lhs = builder.input((3,), dtype="float32")
    rhs = builder.input((3,), dtype="float32")
    module = builder.finish((lhs + rhs).relu())

    def keep_relu(candidate):
        returned = candidate.function.ops[-1].operands[0]
        return returned.producer is not None and returned.producer.opcode == "relu"

    result = minimize_operations(module, keep_relu)

    assert result.original_operation_count == 2
    assert result.minimized_operation_count == 1
    assert result.attempts == 3
    assert result.accepted_reductions == 1
    minimized = result.module
    assert [op.opcode for op in minimized.function.ops] == ["input", "input", "relu", "return"]
    assert minimized.function.ops[-1].operands[0] is minimized.function.ops[2].results[0]
    assert minimized.function.ops[2].operands[0] is minimized.function.ops[1].results[0]


def test_operation_minimizer_only_substitutes_exact_result_type_operands():
    builder = GraphBuilder()
    lhs = builder.input((2, 1), dtype="float32")
    rhs = builder.input((1, 3), dtype="float32")
    module = builder.finish(lhs + rhs)

    result = minimize_operations(module, lambda candidate: True)

    assert result.original_operation_count == 1
    assert result.minimized_operation_count == 1
    assert result.attempts == 0
    assert result.accepted_reductions == 0
    assert result.module_json == serialize_module(module)


def test_operation_minimizer_fails_closed_on_effectful_module():
    builder = GraphBuilder()
    lhs = builder.input((4,), dtype="float32")
    rhs = builder.input((4,), dtype="float32")
    source = builder.input((4,), dtype="float32")
    root = lhs + rhs
    module = builder.finish(root.copy_into(root, source))

    with pytest.raises(ReproMinimizationError, match="effectful opcode"):
        minimize_operations(module, lambda candidate: True)


def test_cli_runs_external_predicate_without_a_shell(tmp_path, capsys):
    module_path = tmp_path / "module.json"
    output_path = tmp_path / "minimized.json"
    predicate_path = tmp_path / "predicate.py"
    module_path.write_text(serialize_module(_multi_output_module()), encoding="utf-8")
    predicate_path.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "from tiny_tensor_compiler.serialization import deserialize_module\n"
        "module = deserialize_module(Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
        "shapes = [value.type.shape for value in module.function.ops[-1].operands]\n"
        "raise SystemExit(0 if (3,) in shapes else 1)\n",
        encoding="utf-8",
    )

    assert main(
        [
            str(module_path),
            str(output_path),
            "--predicate",
            sys.executable,
            str(predicate_path),
        ]
    ) == 0

    minimized = deserialize_module(output_path.read_text(encoding="utf-8"))
    assert _returned_shapes(minimized) == ((3,),)
    output = capsys.readouterr().out
    assert "returns: 3 -> 1" in output
    assert "accepted reductions: 2" in output


def test_cli_can_apply_operation_reduction_with_external_predicate(tmp_path, capsys):
    builder = GraphBuilder()
    lhs = builder.input((3,), dtype="float32")
    rhs = builder.input((3,), dtype="float32")
    module = builder.finish((lhs + rhs).relu())
    module_path = tmp_path / "module.json"
    output_path = tmp_path / "minimized.json"
    predicate_path = tmp_path / "predicate.py"
    module_path.write_text(serialize_module(module), encoding="utf-8")
    predicate_path.write_text("raise SystemExit(0)\n", encoding="utf-8")

    assert main(
        [
            str(module_path),
            str(output_path),
            "--reduce-operations",
            "--predicate",
            sys.executable,
            str(predicate_path),
        ]
    ) == 0

    minimized = deserialize_module(output_path.read_text(encoding="utf-8"))
    assert [op.opcode for op in minimized.function.ops] == ["input", "input", "return"]
    output = capsys.readouterr().out
    assert "operations: 2 -> 0" in output
    assert "operation reductions: 2" in output


def test_cli_distinguishes_nonreproduction_from_predicate_failure(tmp_path):
    module_path = tmp_path / "module.json"
    output_path = tmp_path / "minimized.json"
    helper = tmp_path / "predicate.py"
    module_path.write_text(serialize_module(_multi_output_module()), encoding="utf-8")

    helper.write_text("raise SystemExit(1)\n", encoding="utf-8")
    assert main(
        [str(module_path), str(output_path), "--predicate", sys.executable, str(helper)]
    ) == 1
    assert not output_path.exists()

    helper.write_text("raise SystemExit(2)\n", encoding="utf-8")
    assert main(
        [str(module_path), str(output_path), "--predicate", sys.executable, str(helper)]
    ) == 2
    assert not output_path.exists()
