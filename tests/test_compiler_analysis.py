import json

import pytest

from tiny_tensor_compiler.analysis import analyze_module
from tiny_tensor_compiler.frontend import GraphBuilder
from tiny_tensor_compiler.ir import SymbolicDim


def test_report_measures_actual_fusion_pipeline():
    builder = GraphBuilder("fusion-report")
    lhs = builder.input((8,), "int32")
    rhs = builder.input((8,), "int32")
    module = builder.finish((lhs + rhs).relu())

    report = analyze_module(module)

    assert report.function_name == "fusion-report"
    assert report.input_count == 2
    assert report.output_count == 1
    assert dict(report.tensor_op_counts) == {"add": 1, "input": 2, "relu": 1, "return": 1}
    assert dict(report.pre_fusion_kernel_counts) == {"add": 1, "relu": 1}
    assert report.post_fusion_kernel_count == 1
    assert report.fused_kernel_count == 1
    assert report.kernels_eliminated_by_fusion == 1


def test_report_counts_aliases_without_double_counting_storage():
    builder = GraphBuilder("alias-report")
    source = builder.input((2, 4), "int32")
    module = builder.finish(source.view((4, 2)))

    report = analyze_module(module)

    assert report.logical_value_count == 2
    assert report.alias_value_count == 1
    assert report.physical_storage_count == 1
    assert report.logical_tensor_bytes == 64
    assert report.planned_owning_storage_bytes == 32
    assert len(report.storage_slots) == 1
    assert report.storage_slots[0].byte_count == 32
    assert report.storage_slots[0].dtype == "i32"
    assert report.storage_slots[0].shape == (2, 4)


def test_report_observes_mutation_effects_and_reductions_without_changing_them():
    builder = GraphBuilder("effect-report")
    source = builder.input((2, 4), "int32")
    delta = builder.input((2, 2), "int32")
    one = builder.tensor(1, dtype="int32")
    root = source + one
    target = root.slice(axis=1, start=0, stop=4, step=2)
    updated = root.add_into(target, delta)
    reduced = updated.transpose((1, 0)).sum(axis=1)
    module = builder.finish((updated, reduced))

    report = analyze_module(module)

    assert dict(report.effect_counts) == {"binary_into": 1}
    assert report.view_count == 2
    assert report.output_count == 2
    assert dict(report.post_fusion_kernel_counts)["sum"] == 1
    assert report.alias_value_count >= 3


def test_report_json_is_canonical_and_symbolic_modules_require_specialization():
    builder = GraphBuilder("stable-report")
    source = builder.input((4,), "float32")
    module = builder.finish(source.relu())

    first = analyze_module(module).to_json()
    second = analyze_module(module).to_json()

    assert first == second
    decoded = json.loads(first)
    assert decoded["format"] == "tiny-tensor-compiler-report"
    assert decoded["version"] == 1
    assert decoded["function_name"] == "stable-report"

    dynamic_builder = GraphBuilder("dynamic-report")
    batch = SymbolicDim("B")
    dynamic = dynamic_builder.finish(dynamic_builder.input((batch, 4), "float32").relu())

    with pytest.raises(ValueError, match="concrete tensor shapes"):
        analyze_module(dynamic)
