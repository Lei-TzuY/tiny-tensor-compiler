from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .fusion_planner import fuse_elementwise
from .ir import Module, TensorType
from .layout import element_count
from .loop_ir import (
    LoopBinaryInto,
    LoopCopyInto,
    LoopInplaceBinary,
    LoopKernel,
    LoopProgram,
    fused_expression_for_kernel,
    lower_to_loops,
)
from .lowering import lower_to_cpu, plan_memory
from .symbolic import has_symbolic_shapes

_REPORT_FORMAT = "tiny-tensor-compiler-report"
_REPORT_VERSION = 1


@dataclass(frozen=True)
class StorageSlotReport:
    """One compiler-owned physical storage root in the concrete memory plan."""

    slot: int
    shape: tuple[int, ...]
    dtype: str
    byte_count: int


@dataclass(frozen=True)
class CompilerReport:
    """Deterministic structural facts from the concrete compiler pipeline.

    Byte counts describe planned compiler-owned tensor storage, not process RSS or
    a runtime peak-memory measurement. Fusion counts are structural transformation
    facts and are not a performance claim.
    """

    function_name: str
    input_count: int
    output_count: int
    tensor_op_counts: tuple[tuple[str, int], ...]
    logical_value_count: int
    logical_tensor_bytes: int
    physical_storage_count: int
    planned_owning_storage_bytes: int
    storage_slots: tuple[StorageSlotReport, ...]
    alias_value_count: int
    view_count: int
    effect_counts: tuple[tuple[str, int], ...]
    pre_fusion_kernel_count: int
    pre_fusion_kernel_counts: tuple[tuple[str, int], ...]
    post_fusion_kernel_count: int
    post_fusion_kernel_counts: tuple[tuple[str, int], ...]
    fused_kernel_count: int
    kernels_eliminated_by_fusion: int
    format: str = _REPORT_FORMAT
    version: int = _REPORT_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible deterministic report mapping."""
        return asdict(self)

    def to_json(self) -> str:
        """Return canonical JSON suitable for snapshots and exact comparisons."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def analyze_module(module: Module) -> CompilerReport:
    """Analyze one verified concrete module through the real pre-native pipeline."""
    if not isinstance(module, Module):
        raise TypeError("analyze_module requires a Module")
    if has_symbolic_shapes(module):
        raise ValueError(
            "analyze_module requires concrete tensor shapes; specialize symbolic modules first"
        )

    cpu = lower_to_cpu(module)
    memory = plan_memory(cpu)
    pre_fusion = lower_to_loops(cpu)
    post_fusion = fuse_elementwise(pre_fusion)

    storage_slots = tuple(
        StorageSlotReport(
            slot=slot,
            shape=type_.shape,
            dtype=type_.dtype.value,
            byte_count=_tensor_bytes(type_),
        )
        for slot, type_ in enumerate(memory.physical_types)
    )
    pre_kernels = tuple(pre_fusion.kernels)
    post_kernels = tuple(post_fusion.kernels)
    eliminated = len(pre_kernels) - len(post_kernels)
    if eliminated < 0:
        raise RuntimeError("fusion unexpectedly increased the number of loop kernels")

    return CompilerReport(
        function_name=module.function.name,
        input_count=len(pre_fusion.inputs),
        output_count=len(post_fusion.return_slots),
        tensor_op_counts=_histogram(op.opcode for op in module.function.ops),
        logical_value_count=len(cpu.allocations),
        logical_tensor_bytes=sum(_tensor_bytes(op.type) for op in cpu.allocations),
        physical_storage_count=memory.physical_count,
        planned_owning_storage_bytes=sum(slot.byte_count for slot in storage_slots),
        storage_slots=storage_slots,
        alias_value_count=len(memory.aliases),
        view_count=len(pre_fusion.views),
        effect_counts=_effect_histogram(pre_fusion),
        pre_fusion_kernel_count=len(pre_kernels),
        pre_fusion_kernel_counts=_histogram(kernel.opcode for kernel in pre_kernels),
        post_fusion_kernel_count=len(post_kernels),
        post_fusion_kernel_counts=_histogram(kernel.opcode for kernel in post_kernels),
        fused_kernel_count=sum(
            fused_expression_for_kernel(kernel) is not None for kernel in post_kernels
        ),
        kernels_eliminated_by_fusion=eliminated,
    )


def _effect_histogram(program: LoopProgram) -> tuple[tuple[str, int], ...]:
    effects: list[str] = []
    for op in program.operations:
        if isinstance(op, LoopCopyInto):
            effects.append("copy_into")
        elif isinstance(op, LoopBinaryInto):
            effects.append("binary_into")
        elif isinstance(op, LoopInplaceBinary):
            effects.append("binary_inplace")
    return _histogram(effects)


def _histogram(values: Iterable[str]) -> tuple[tuple[str, int], ...]:
    counts = Counter(values)
    return tuple(sorted(counts.items()))


def _tensor_bytes(type_: TensorType) -> int:
    shape = type_.shape
    if any(not isinstance(dim, int) or isinstance(dim, bool) for dim in shape):
        raise ValueError("compiler analysis storage accounting requires concrete tensor shapes")
    return element_count(shape) * type_.dtype.to_numpy().itemsize
