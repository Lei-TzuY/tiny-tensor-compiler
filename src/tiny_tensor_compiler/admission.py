from __future__ import annotations

from dataclasses import dataclass

from .analysis import CompilerReport, analyze_module
from .ir import Module


@dataclass(frozen=True)
class CompileBudget:
    """Optional structural limits checked before native compilation.

    The storage limit uses ``CompilerReport.planned_owning_storage_bytes`` from the
    ordinary concrete pre-native memory plan. It is not a process-RSS, heap-peak,
    stack-peak, or borrow-adjusted runtime-memory limit. The kernel limit uses the
    post-fusion Loop IR kernel count and is not a runtime-cost estimate.
    """

    max_planned_storage_bytes: int | None = None
    max_post_fusion_kernels: int | None = None

    def __post_init__(self) -> None:
        _validate_limit("max_planned_storage_bytes", self.max_planned_storage_bytes)
        _validate_limit("max_post_fusion_kernels", self.max_post_fusion_kernels)


class CompileBudgetExceeded(RuntimeError):
    """Raised when a concrete compiler report exceeds one configured limit."""

    def __init__(
        self,
        *,
        metric: str,
        limit: int,
        actual: int,
        report: CompilerReport,
    ) -> None:
        self.metric = metric
        self.limit = limit
        self.actual = actual
        self.report = report
        super().__init__(
            f"compile budget exceeded: {metric} actual {actual} exceeds limit {limit}"
        )


def enforce_compile_budget(module: Module, budget: CompileBudget) -> CompilerReport:
    """Analyze one concrete module and fail closed if a configured limit is exceeded."""
    if not isinstance(module, Module):
        raise TypeError("enforce_compile_budget requires a Module")
    if not isinstance(budget, CompileBudget):
        raise TypeError("enforce_compile_budget requires a CompileBudget")

    report = analyze_module(module)
    _check_limit(
        report,
        metric="planned_owning_storage_bytes",
        limit=budget.max_planned_storage_bytes,
        actual=report.planned_owning_storage_bytes,
    )
    _check_limit(
        report,
        metric="post_fusion_kernel_count",
        limit=budget.max_post_fusion_kernels,
        actual=report.post_fusion_kernel_count,
    )
    return report


def _validate_limit(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer or None")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _check_limit(
    report: CompilerReport,
    *,
    metric: str,
    limit: int | None,
    actual: int,
) -> None:
    if limit is None or actual <= limit:
        return
    raise CompileBudgetExceeded(
        metric=metric,
        limit=limit,
        actual=actual,
        report=report,
    )
