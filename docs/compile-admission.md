# Compile admission budgets

`CompileBudget` turns the compiler's existing deterministic structural analysis into an optional fail-closed admission gate before native compilation begins.

The policy is intentionally small and evidence-backed. It does not introduce a second lowering pipeline: `enforce_compile_budget()` consumes the same `CompilerReport` produced by `analyze_module()`, which already runs the repository's verified concrete path through CPU lowering, memory planning, Loop IR, and elementwise fusion.

## Supported limits

Two inclusive limits are currently available:

- `max_planned_storage_bytes`: upper bound on `CompilerReport.planned_owning_storage_bytes`.
- `max_post_fusion_kernels`: upper bound on `CompilerReport.post_fusion_kernel_count`.

Each limit is either `None` or a non-negative Python integer. Boolean values are rejected even though `bool` is an `int` subclass.

If a metric exceeds its configured limit, compilation raises `CompileBudgetExceeded`. The exception exposes `metric`, `limit`, `actual`, and the exact `CompilerReport` used for the decision.

## Concrete fail-closed compilation

`compile_module(module, budget=...)` checks the concrete module before native compilation or native-cache lookup is invoked. A rejected module therefore does not create, load, or publish a native artifact.

When `budget=None`, the historical compilation path is preserved: no compiler report is computed solely for admission, and existing native lowering/call boundaries are unchanged.

This fail-closed contract remains unchanged by adaptive compilation. Callers that require rejection on budget excess should continue to use `compile_module(..., budget=...)`.

## Explicit adaptive execution

`tiny_tensor_compiler.compiler.compile_adaptive_module(module, budget=...)` is a separate opt-in policy. It evaluates the same `CompileBudget` against the same deterministic `CompilerReport` before choosing one of two already-verified execution paths:

- within budget: eagerly compile the ordinary native executable;
- budget exceeded: retain the ordinary verified post-lowering Loop program and execute it through the existing CPU Loop interpreter.

The returned `AdaptiveExecutable` exposes `backend`, `report`, and `budget_exceeded`, so the decision is observable rather than hidden. `backend` is exactly `"native"` or `"loop"`. The loop path is selected only when `enforce_compile_budget()` raises `CompileBudgetExceeded`; verifier failures, symbolic-shape errors, native compiler/load failures, and runtime input validation errors are not swallowed or converted into fallback.

`borrow_inputs=True` is applied through the same verified input-lifetime transform on either selected path. `parallel=True` affects only an admitted native specialization; the Loop CPU fallback does not pretend to provide OpenMP execution.

The first adaptive facade intentionally has no `out=` argument. Native preallocated-output support therefore does not become a backend-dependent API whose semantics disappear when a budget triggers Loop fallback. Adaptive execution currently promises the common `inputs -> result` contract, including ordered multiple outputs.

## Dynamic specialization

`compile_dynamic_module(module, budget=...)` keeps its historical fail-closed behavior. Runtime symbolic inputs are solved and concretized as usual; each distinct concrete specialization is checked against the same budget before its native executable is created. A rejected concrete binding is not inserted into `DynamicExecutable.cached_bindings`.

`tiny_tensor_compiler.compiler.compile_adaptive_dynamic_module(module, budget=...)` instead returns `AdaptiveDynamicExecutable`. Each concrete symbolic binding is solved, specialized, analyzed, and selected independently. Within-budget bindings cache native `AdaptiveExecutable` values; over-budget bindings cache Loop fallback values. `cached_bindings` records all selected concrete bindings and `cached_binding_backends` exposes the deterministic backend selected for each binding.

A budget decision is made once per cached binding. Repeated use of the same binding reuses the same adaptive specialization rather than retrying native compilation or changing backend opportunistically.

## Evidence boundary

These metrics are structural compiler facts, not runtime resource measurements:

- `planned_owning_storage_bytes` counts compiler-owned physical storage roots in the ordinary pre-native memory plan. It is not process RSS, peak heap/stack usage, allocator overhead, operating-system working set, OpenMP runtime memory, or a borrow-adjusted runtime footprint.
- `post_fusion_kernel_count` counts Loop IR kernels after the existing fusion planner. It is not a wall-clock cost, instruction count, profitability score, or performance estimate.

Accordingly, `CompileBudget` is a deterministic compiler admission policy, not a security sandbox, denial-of-service guarantee, memory limiter, or benchmark. Adaptive Loop fallback also does not claim to reduce runtime memory or improve performance; it only avoids native compilation for the explicit over-budget policy case while preserving verified executable semantics.

## Phase boundary

The fail-closed admission phase and the first explicit adaptive native-or-Loop execution phase are now separate, composable policies. Adding more report counters, fallback modes, or retry heuristics without a distinct operational requirement would be low-value farming. A later runtime-policy milestone should introduce a qualitatively new enforceable resource or execution-control capability, such as bounded external compiler-process control with evidence, rather than disguising more structural heuristics as runtime guarantees.
