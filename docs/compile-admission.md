# Compile admission budgets

`CompileBudget` turns the compiler's existing deterministic structural analysis into an optional fail-closed admission gate before native compilation begins.

The policy is intentionally small and evidence-backed. It does not introduce a second lowering pipeline: `enforce_compile_budget()` consumes the same `CompilerReport` produced by `analyze_module()`, which already runs the repository's verified concrete path through CPU lowering, memory planning, Loop IR, and elementwise fusion.

## Supported limits

Two inclusive limits are currently available:

- `max_planned_storage_bytes`: upper bound on `CompilerReport.planned_owning_storage_bytes`.
- `max_post_fusion_kernels`: upper bound on `CompilerReport.post_fusion_kernel_count`.

Each limit is either `None` or a non-negative Python integer. Boolean values are rejected even though `bool` is an `int` subclass.

If a metric exceeds its configured limit, compilation raises `CompileBudgetExceeded`. The exception exposes `metric`, `limit`, `actual`, and the exact `CompilerReport` used for the decision.

## Concrete compilation

`compile_module(module, budget=...)` checks the concrete module before native compilation or native-cache lookup is invoked. A rejected module therefore does not create, load, or publish a native artifact.

When `budget=None`, the historical compilation path is preserved: no compiler report is computed solely for admission, and existing native lowering/call boundaries are unchanged.

## Dynamic specialization

`compile_dynamic_module(module, budget=...)` freezes the policy in the returned `DynamicExecutable`. Runtime symbolic inputs are solved and concretized as usual; each distinct concrete specialization is then checked against the same budget before its native executable is created.

A rejected concrete binding is not inserted into `DynamicExecutable.cached_bindings`. Other bindings may still be admitted and cached independently.

## Evidence boundary

These metrics are structural compiler facts, not runtime resource measurements:

- `planned_owning_storage_bytes` counts compiler-owned physical storage roots in the ordinary pre-native memory plan. It is not process RSS, peak heap/stack usage, allocator overhead, operating-system working set, OpenMP runtime memory, or a borrow-adjusted runtime footprint.
- `post_fusion_kernel_count` counts Loop IR kernels after the existing fusion planner. It is not a wall-clock cost, instruction count, profitability score, or performance estimate.

Accordingly, `CompileBudget` is a deterministic compiler admission policy, not a security sandbox, denial-of-service guarantee, memory limiter, or benchmark. Stronger runtime-resource enforcement would require separate measurement and execution controls.

## Phase boundary

This phase closes the first report-backed compilation-decision slice. Adding more report counters as budget knobs without a distinct operational need would be low-value farming. Future policy work should add a qualitatively new verified decision capability or move to another compiler/runtime frontier.