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

The returned `AdaptiveExecutable` exposes `backend`, `report`, and `budget_exceeded`, so the decision is observable rather than hidden. `backend` is exactly `"native"` or `"loop"`. The loop path is selected only when `enforce_compile_budget()` raises `CompileBudgetExceeded`; verifier failures, symbolic-shape errors, native compiler/load failures, compiler timeouts, and runtime input validation errors are not swallowed or converted into fallback.

`borrow_inputs=True` is applied through the same verified input-lifetime transform on either selected path. `parallel=True` affects only an admitted native specialization; the Loop CPU fallback does not pretend to provide OpenMP execution.

The first adaptive facade intentionally has no `out=` argument. Native preallocated-output support therefore does not become a backend-dependent API whose semantics disappear when a budget triggers Loop fallback. Adaptive execution currently promises the common `inputs -> result` contract, including ordered multiple outputs.

## Dynamic specialization

`compile_dynamic_module(module, budget=...)` keeps its historical fail-closed behavior. Runtime symbolic inputs are solved and concretized as usual; each distinct concrete specialization is checked against the same budget before its native executable is created. A rejected concrete binding is not inserted into `DynamicExecutable.cached_bindings`.

`tiny_tensor_compiler.compiler.compile_adaptive_dynamic_module(module, budget=...)` instead returns `AdaptiveDynamicExecutable`. Each concrete symbolic binding is solved, specialized, analyzed, and selected independently. Within-budget bindings cache native `AdaptiveExecutable` values; over-budget bindings cache Loop fallback values. `cached_bindings` records all selected concrete bindings and `cached_binding_backends` exposes the deterministic backend selected for each binding.

A budget decision is made once per cached binding. Repeated use of the same binding reuses the same adaptive specialization rather than retrying native compilation or changing backend opportunistically.

## External compiler subprocess timeout

Native compilation APIs now accept the optional `compiler_timeout` policy. It is either `None` or a positive finite number of seconds. Boolean, zero, negative, NaN, and infinite values are rejected before compiler lookup.

The policy bounds only one launched external C compiler subprocess. It is threaded consistently through serial and OpenMP native compilation plus concrete, dynamic, and adaptive high-level compilation. A timed-out compiler raises `NativeCompilationTimeout`, which remains a `NativeCompilationError` and exposes the exact command plus configured timeout for diagnostics.

A timeout never becomes an adaptive Loop fallback. Adaptive fallback remains reserved exclusively for an explicit `CompileBudgetExceeded` decision made before native compilation begins.

Timeout cleanup uses the existing artifact durability boundaries:

- transient build directories are removed after timeout;
- persistent-cache temporary build directories are removed and no library/manifest is published from a timed-out build;
- timeout is not part of native artifact identity, so an existing process or persistent cache hit reuses the already-compiled artifact without launching a compiler merely because the caller supplies a different timeout;
- reusable native/dynamic handles retain the timeout policy for any later compilation that is actually required after a cache miss.

The timeout is intentionally **not** a total compilation deadline. Time spent waiting for a persistent-cache lease is outside this first policy because no compiler subprocess has started yet. It also does not limit native execution after compilation.

## Evidence boundary

These metrics are structural compiler facts, not runtime resource measurements:

- `planned_owning_storage_bytes` counts compiler-owned physical storage roots in the ordinary pre-native memory plan. It is not process RSS, peak heap/stack usage, allocator overhead, operating-system working set, OpenMP runtime memory, or a borrow-adjusted runtime footprint.
- `post_fusion_kernel_count` counts Loop IR kernels after the existing fusion planner. It is not a wall-clock cost, instruction count, profitability score, or performance estimate.

Accordingly, `CompileBudget` is a deterministic compiler admission policy, not a security sandbox, denial-of-service guarantee, memory limiter, or benchmark. Adaptive Loop fallback also does not claim to reduce runtime memory or improve performance; it only avoids native compilation for the explicit over-budget policy case while preserving verified executable semantics.

`compiler_timeout` is likewise a bounded wait for the directly launched compiler subprocess, not a process-tree sandbox, CPU quota, memory limit, trusted cancellation protocol, runtime timeout, cache-lock deadline, or general denial-of-service guarantee. No security/isolation claim is implied by subprocess timeout enforcement.

## Phase boundary

The fail-closed admission phase, adaptive native-or-Loop policy, and bounded external compiler-subprocess timeout are now separate, composable controls. Adding more report counters, fallback modes, timeout aliases, or retry heuristics without a distinct operational requirement would be low-value farming. The next runtime-control promotion should require genuinely new enforceable semantics—such as an explicit total compilation deadline/process-tree cancellation model with cross-platform evidence, or move to another architectural frontier—rather than disguising additional thresholds as a new subsystem.
