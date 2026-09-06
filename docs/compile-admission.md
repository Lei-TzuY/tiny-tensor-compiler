# Compile admission budgets

`CompileBudget` turns deterministic compiler facts and dynamic-handle state into optional fail-closed admission gates before new native compilation begins.

The policy is intentionally small and evidence-backed. It does not introduce a second lowering pipeline: concrete structural checks consume the same `CompilerReport` produced by `analyze_module()`, which already runs the repository's verified concrete path through CPU lowering, memory planning, Loop IR, and elementwise fusion. Dynamic specialization cardinality is enforced separately by the reusable dynamic handle because it is a property of that handle's accumulated successful bindings, not of one concrete module.

## Supported limits

Three non-negative limits are currently available:

- `max_planned_storage_bytes`: upper bound on `CompilerReport.planned_owning_storage_bytes`.
- `max_post_fusion_kernels`: upper bound on `CompilerReport.post_fusion_kernel_count`.
- `max_dynamic_specializations`: maximum number of distinct successfully cached complete symbolic bindings retained by one `DynamicExecutable` or `AdaptiveDynamicExecutable`.

Each limit is either `None` or a non-negative Python integer. Boolean values are rejected even though `bool` is an `int` subclass. A dynamic-specialization limit of `0` therefore permits cache hits only if a specialization somehow already exists; a newly created handle cannot admit its first binding.

If a concrete structural metric exceeds its configured limit, compilation raises `CompileBudgetExceeded`. The exception exposes `metric`, `limit`, `actual`, and the exact `CompilerReport` used for the decision.

If an unseen runtime binding would exceed `max_dynamic_specializations`, the dynamic handle instead raises `DynamicSpecializationBudgetExceeded` before concrete specialization or backend selection. The exception exposes `limit`, the deterministic `attempted_binding`, and the sorted `cached_bindings` that already consume the handle's capacity.

## Concrete fail-closed compilation

`compile_module(module, budget=...)` checks the concrete structural limits before native compilation or native-cache lookup is invoked. A rejected module therefore does not create, load, or publish a native artifact.

`max_dynamic_specializations` has no meaning for one already-concrete module and is intentionally ignored by `enforce_compile_budget()`. It is enforced only by reusable dynamic handles that can accumulate multiple concrete bindings over time.

When `budget=None`, the historical compilation path is preserved: no compiler report is computed solely for admission, and existing native lowering/call boundaries are unchanged.

This fail-closed contract remains unchanged by adaptive compilation. Callers that require rejection on concrete structural budget excess should continue to use `compile_module(..., budget=...)`.

## Explicit adaptive execution

`tiny_tensor_compiler.compiler.compile_adaptive_module(module, budget=...)` is a separate opt-in policy. It evaluates the concrete structural portion of the same `CompileBudget` against the same deterministic `CompilerReport` before choosing one of two already-verified execution paths:

- within budget: eagerly compile the ordinary native executable;
- concrete structural budget exceeded: retain the ordinary verified post-lowering Loop program and execute it through the existing CPU Loop interpreter.

The returned `AdaptiveExecutable` exposes `backend`, `report`, and `budget_exceeded`, so the decision is observable rather than hidden. `backend` is exactly `"native"` or `"loop"`. The loop path is selected only when `enforce_compile_budget()` raises `CompileBudgetExceeded`; dynamic-specialization admission failures, verifier failures, symbolic-shape errors, native compiler/load failures, compiler timeouts, and runtime input validation errors are not swallowed or converted into fallback.

`borrow_inputs=True` is applied through the same verified input-lifetime transform on either selected path. `parallel=True` affects only an admitted native specialization; the Loop CPU fallback does not pretend to provide OpenMP execution.

The first adaptive facade intentionally has no `out=` argument. Native preallocated-output support therefore does not become a backend-dependent API whose semantics disappear when a structural budget triggers Loop fallback. Adaptive execution currently promises the common `inputs -> result` contract, including ordered multiple outputs.

## Dynamic specialization

`compile_dynamic_module(module, budget=...)` keeps the existing per-concrete fail-closed behavior and may additionally cap the number of successful runtime bindings retained by the returned handle. Runtime symbolic inputs are solved to a complete deterministic binding first. Cache hits are checked before the cap, so an already-admitted binding remains reusable even after the handle reaches its configured maximum.

For an unseen binding, `max_dynamic_specializations` is checked while holding the same specialization lock and **before** `specialize_module()` or `compile_module()` is invoked. The first `N` successful distinct bindings may therefore populate a handle configured with limit `N`; the next distinct binding is rejected without a new concrete clone, compiler invocation, or native-cache side effect. A specialization that fails before being inserted into the cache does not consume capacity.

The cap counts complete bindings, not individual symbolic dimensions. Multi-symbol diagnostics use the executable's canonical symbol order, so callers receive deterministic entries such as `(("B", 2), ("W", 3))` regardless of mapping insertion order.

`tiny_tensor_compiler.compiler.compile_adaptive_dynamic_module(module, budget=...)` applies the same cardinality gate. Each admitted concrete symbolic binding is solved, specialized, analyzed, and selected independently. Within-structural-budget bindings cache native `AdaptiveExecutable` values; structurally over-budget bindings cache Loop fallback values. Either backend decision counts as one successful specialization because both retain per-binding executable state. `cached_bindings` records all selected concrete bindings and `cached_binding_backends` exposes the deterministic backend selected for each binding.

A backend/budget decision is made once per cached binding. Repeated use of the same binding reuses the same specialization rather than retrying native compilation, changing backend opportunistically, or consuming another cardinality slot.

The first cardinality policy is deliberately fail-closed rather than LRU. Removing a Python dictionary entry would not prove that process-local native artifacts, loaded shared libraries, persistent cache files, or other backend resources had actually been released. Eviction therefore remains a separate future resource-lifecycle problem rather than being implied by this admission cap.

## External compiler subprocess timeout

Native compilation APIs accept the optional `compiler_timeout` policy. It is either `None` or a positive finite number of seconds. Boolean, zero, negative, NaN, and infinite values are rejected before compiler lookup.

The policy bounds one launched external C compiler invocation and, when that explicit timeout is enabled, also terminates the ordinary descendant process tree created by that compiler. It is threaded consistently through serial and OpenMP native compilation plus concrete, dynamic, and adaptive high-level compilation. A timed-out compiler raises `NativeCompilationTimeout`, which remains a `NativeCompilationError` and exposes the exact command plus configured timeout for diagnostics.

With `compiler_timeout=None`, the historical `subprocess.run(...)` compilation path is preserved. With an explicit timeout, POSIX launches the compiler in a new session and kills its process group on timeout; Windows launches a new process group and uses `taskkill /T /F` for the compiler PID before reaping the parent process. Cross-platform regression coverage uses a fake compiler that spawns a delayed child writer and proves the descendant does not survive the timeout boundary.

A timeout never becomes an adaptive Loop fallback. Adaptive fallback remains reserved exclusively for an explicit concrete `CompileBudgetExceeded` decision made before native compilation begins.

Timeout cleanup uses the existing artifact durability boundaries:

- transient build directories are removed after timeout;
- persistent-cache temporary build directories are removed and no library/manifest is published from a timed-out build;
- timeout is not part of native artifact identity, so an existing process or persistent cache hit reuses the already-compiled artifact without launching a compiler merely because the caller supplies a different timeout;
- reusable native/dynamic handles retain the timeout policy for any later compilation that is actually required after a cache miss.

The timeout is intentionally **not** a total compilation deadline. Time spent waiting for a persistent-cache lease is outside this policy because no compiler process tree has started yet. It also does not limit native execution after compilation. Process-tree cancellation covers ordinary descendants that remain in the launched compiler's POSIX process group or Windows process tree; it is not a sandbox guarantee for a deliberately detached process that escapes those operating-system relationships.

## Evidence boundary

The concrete report metrics are structural compiler facts, not runtime resource measurements:

- `planned_owning_storage_bytes` counts compiler-owned physical storage roots in the ordinary pre-native memory plan. It is not process RSS, peak heap/stack usage, allocator overhead, operating-system working set, OpenMP runtime memory, or a borrow-adjusted runtime footprint.
- `post_fusion_kernel_count` counts Loop IR kernels after the existing fusion planner. It is not a wall-clock cost, instruction count, profitability score, or performance estimate.

`max_dynamic_specializations` is likewise a per-handle admission count, not a measurement or hard bound on process memory, generated code bytes, number of shared libraries, persistent-cache size, global native-cache entries, compiler subprocesses, or total specializations created by other handles. It prevents one configured dynamic handle from admitting unbounded new binding identities; it does not claim that rejecting the next binding reclaims resources already created for earlier bindings.

Accordingly, `CompileBudget` is a deterministic compiler admission policy, not a security sandbox, denial-of-service guarantee, memory limiter, or benchmark. Adaptive Loop fallback also does not claim to reduce runtime memory or improve performance; it only avoids native compilation for the explicit concrete over-budget policy case while preserving verified executable semantics.

`compiler_timeout` is likewise a bounded wait plus best-effort operating-system process-tree cancellation for the launched compiler invocation, not a CPU quota, memory limit, trusted sandbox, runtime timeout, cache-lock deadline, or general denial-of-service guarantee. It does not guarantee termination of intentionally detached descendants that escape the managed POSIX process group or Windows process tree. No security/isolation claim is implied by timeout enforcement.

## Phase boundary

Concrete structural admission, adaptive native-or-Loop policy, per-handle dynamic-specialization cardinality, and bounded external compiler process-tree timeout are now separate, composable controls. Adding more counters, cache-size aliases, eviction heuristics, fallback modes, timeout aliases, or retry knobs without a distinct enforceable lifecycle requirement would be low-value farming. The next runtime-control promotion should require genuinely new semantics—such as resource-accounted specialization eviction with proven native-artifact release, a total compilation deadline that also bounds persistent-cache lease waiting and other pre/post-compiler phases, or another architectural frontier—rather than disguising another threshold as a subsystem.
