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

The returned `AdaptiveExecutable` exposes `backend`, `report`, and `budget_exceeded`, so the decision is observable rather than hidden. `backend` is exactly `"native"` or `"loop"`. The loop path is selected only when `enforce_compile_budget()` raises `CompileBudgetExceeded`; verifier failures, symbolic-shape errors, native compiler/load failures, compiler/deadline timeouts, and runtime input validation errors are not swallowed or converted into fallback.

`borrow_inputs=True` is applied through the same verified input-lifetime transform on either selected path. `parallel=True` affects only an admitted native specialization; the Loop CPU fallback does not pretend to provide OpenMP execution.

The first adaptive facade intentionally has no `out=` argument. Native preallocated-output support therefore does not become a backend-dependent API whose semantics disappear when a budget triggers Loop fallback. Adaptive execution currently promises the common `inputs -> result` contract, including ordered multiple outputs.

## Dynamic specialization

`compile_dynamic_module(module, budget=...)` keeps its historical fail-closed behavior. Runtime symbolic inputs are solved and concretized as usual; each distinct concrete specialization is checked against the same budget before its native executable is created. A rejected concrete binding is not inserted into `DynamicExecutable.cached_bindings`.

`tiny_tensor_compiler.compiler.compile_adaptive_dynamic_module(module, budget=...)` instead returns `AdaptiveDynamicExecutable`. Each concrete symbolic binding is solved, specialized, analyzed, and selected independently. Within-budget bindings cache native `AdaptiveExecutable` values; over-budget bindings cache Loop fallback values. `cached_bindings` records all selected concrete bindings and `cached_binding_backends` exposes the deterministic backend selected for each binding.

A budget decision is made once per cached binding. Repeated use of the same binding reuses the same adaptive specialization rather than retrying native compilation or changing backend opportunistically.

## External compiler timeout and total compilation deadline

Native compilation exposes two distinct optional wall-clock controls. Both are either `None` or a positive finite number of seconds; Boolean, zero, negative, NaN, and infinite values are rejected before compiler lookup.

`compiler_timeout` bounds only one launched external C compiler stage. A timeout raises `NativeCompilationTimeout`, which exposes the exact command plus configured compiler-stage timeout.

`compilation_timeout` starts one monotonic deadline for the native artifact transaction. It is threaded through serial and OpenMP native compilation plus concrete, dynamic, and adaptive high-level compilation. The deadline covers cooperative checks around compiler lookup/source generation/artifact identity, waiting for the process-local native-artifact cache lock, waiting for a persistent-cache lease, the compiler process, artifact publication/staging, and shared-library load. A deadline expiration raises `NativeCompilationDeadlineExceeded`, which exposes the configured total timeout, the phase where expiry was observed, and the compiler command when the compiler stage was active.

When both controls are configured, the earlier remaining bound governs the compiler stage. If the compiler-stage timeout expires first the exception remains `NativeCompilationTimeout`; if the total transaction deadline expires first it remains `NativeCompilationDeadlineExceeded`. The two policies therefore compose without changing their diagnostic meaning.

A timeout never becomes an adaptive Loop fallback. Adaptive fallback remains reserved exclusively for an explicit `CompileBudgetExceeded` decision made before native compilation begins.

Deadline cleanup uses the existing artifact durability boundaries and adds bounded cancellation semantics:

- transient build directories are removed after timeout;
- persistent-cache temporary build directories are removed, and a compiler killed before publication cannot publish a library/manifest;
- process-local cache-lock and persistent-cache lease waits obey the same total deadline instead of blocking indefinitely under the total-deadline policy;
- timed-out compiler descendants are terminated as a process tree: POSIX launches the compiler in a new session and kills its process group, while Windows launches a new process group and uses `taskkill /T /F` with direct-kill fallback;
- timeout values are not part of native artifact identity, so a valid process or persistent cache hit can reuse the already-compiled artifact without relaunching a compiler merely because the caller supplies a different timeout;
- reusable native/dynamic handles retain both timeout policies for later artifact acquisition that is actually required after a cache miss.

The total deadline is deliberately scoped to native artifact acquisition, not arbitrary caller work before that transaction. High-level concrete/dynamic/adaptive APIs propagate the relative `compilation_timeout` into each native artifact transaction; tensor analysis/lowering that occurs before the native facade is not retroactively charged to a hidden global stopwatch.

The deadline also does not limit native execution after an artifact is ready. Filesystem operations such as source writes, hashing, atomic publication, staging copies, or dynamic-library loading are guarded by checks before/after relevant phases but are not forcibly preempted mid-system-call. The policy is therefore a bounded cooperative compilation transaction plus compiler-process-tree cancellation, not a general operating-system cancellation primitive.

## Evidence boundary

These metrics are structural compiler facts, not runtime resource measurements:

- `planned_owning_storage_bytes` counts compiler-owned physical storage roots in the ordinary pre-native memory plan. It is not process RSS, peak heap/stack usage, allocator overhead, operating-system working set, OpenMP runtime memory, or a borrow-adjusted runtime footprint.
- `post_fusion_kernel_count` counts Loop IR kernels after the existing fusion planner. It is not a wall-clock cost, instruction count, profitability score, or performance estimate.

Accordingly, `CompileBudget` is a deterministic compiler admission policy, not a security sandbox, denial-of-service guarantee, memory limiter, or benchmark. Adaptive Loop fallback also does not claim to reduce runtime memory or improve performance; it only avoids native compilation for the explicit over-budget policy case while preserving verified executable semantics.

`compiler_timeout` and `compilation_timeout` are likewise operational cancellation bounds, not a CPU quota, memory limit, trusted sandbox, native-execution timeout, hostile-filesystem preemption mechanism, or general denial-of-service guarantee. Process-tree termination is durability behavior for compiler descendants, not a claim that arbitrary untrusted compiler code is securely isolated.

## Phase boundary

The fail-closed admission phase, adaptive native-or-Loop policy, direct compiler-stage timeout, and total native artifact-transaction deadline are now separate, composable controls. The total-deadline phase closes the previously identified cache-wait/process-tree cancellation gap with cross-platform executable evidence. Adding more timeout aliases, phase counters, retries, or fallback modes without a distinct operational requirement would be low-value farming; the next milestone should promote to a new enforceable compiler/runtime capability rather than rename or subdivide the same deadline policy.
