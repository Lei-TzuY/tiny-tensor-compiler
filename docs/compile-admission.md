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

The returned `AdaptiveExecutable` exposes `backend`, `report`, and `budget_exceeded`, so the decision is observable rather than hidden. `backend` is exactly `"native"` or `"loop"`. The loop path is selected only when `enforce_compile_budget()` raises `CompileBudgetExceeded`; dynamic-specialization admission failures, verifier failures, symbolic-shape errors, native compiler/load failures, compiler timeouts, total compile-deadline failures, and runtime input validation errors are not swallowed or converted into fallback.

`borrow_inputs=True` is applied through the same verified input-lifetime transform on either selected path. `parallel=True` affects only an admitted native specialization; the Loop CPU fallback does not pretend to provide OpenMP execution.

The first adaptive facade intentionally has no `out=` argument. Native preallocated-output support therefore does not become a backend-dependent API whose semantics disappear when a structural budget triggers Loop fallback. Adaptive execution currently promises the common `inputs -> result` contract, including ordered multiple outputs.

## Dynamic specialization

`compile_dynamic_module(module, budget=...)` keeps the existing per-concrete fail-closed behavior and may additionally cap the number of successful runtime bindings retained by the returned handle. Runtime symbolic inputs are solved to a complete deterministic binding first. Cache hits are checked before the cap, so an already-admitted binding remains reusable even after the handle reaches its configured maximum.

For an unseen binding, `max_dynamic_specializations` is checked while holding the same specialization lock and **before** `specialize_module()` or `compile_module()` is invoked. The first `N` successful distinct bindings may therefore populate a handle configured with limit `N`; the next distinct binding is rejected without a new concrete clone, compiler invocation, or native-cache side effect. A specialization that fails before being inserted into the cache does not consume capacity.

The cap counts complete bindings, not individual symbolic dimensions. Multi-symbol diagnostics use the executable's canonical symbol order, so callers receive deterministic entries such as `(("B", 2), ("W", 3))` regardless of mapping insertion order.

`tiny_tensor_compiler.compiler.compile_adaptive_dynamic_module(module, budget=...)` applies the same cardinality gate. Each admitted concrete symbolic binding is solved, specialized, analyzed, and selected independently. Within-structural-budget bindings cache native `AdaptiveExecutable` values; structurally over-budget bindings cache Loop fallback values. Either backend decision counts as one successful specialization because both retain per-binding executable state. `cached_bindings` records all selected concrete bindings and `cached_binding_backends` exposes the deterministic backend selected for each binding.

A backend/budget decision is made once per cached binding. Repeated use of the same binding reuses the same specialization rather than retrying native compilation, changing backend opportunistically, or consuming another cardinality slot.

## Resource-managed specialization retention

Admission and retention are separate policies. The ordinary `max_dynamic_specializations` cap remains fail-closed and does not imply reclamation. Callers that require bounded retained specialization state can instead opt into:

- `tiny_tensor_compiler.specialization_cache.compile_resource_managed_dynamic_module(...)`;
- `tiny_tensor_compiler.specialization_cache.compile_resource_managed_adaptive_dynamic_module(...)`.

These facades add `max_cached_specializations`, a non-negative per-handle retention limit with deterministic least-recently-used ordering. A cache hit refreshes the binding to most-recently-used position. `max_cached_specializations=0` is valid: the newly created executable is returned to the caller but immediately removed from the managed handle's specialization map.

Managed serial native specializations coordinate ownership by the same exact process-local native-artifact identity already used by the native cache: compiler command, generated C source, and optional persistent-cache identity. Ordinary and adaptive-native resource-managed handles that retain the same identity share the loaded artifact. Evicting one specialization removes only that handle's ownership reference while another managed owner remains; only eviction of the final managed owner removes the process-cache entry, unloads the shared library, and deletes its process-owned staging directory. The accounting is per owner and per retained specialization rather than a boolean set, so one handle can safely retain more than one specialization resolving to the same identity.

An external reference to an evicted `NativeExecutable` remains usable but is not counted as managed retention ownership. After the final managed owner releases an identity, such a reference reacquires the same artifact on its next execution. With a persistent cache, the durable artifact is preserved and can be reloaded without another compiler invocation.

Adaptive managed specialization preserves backend truthfulness: evicting `backend="native"` releases one managed native ownership reference and unloads only if that was the final managed owner, while evicting `backend="loop"` releases only the retained specialization decision and does not increment native-release accounting. The handles expose `retained_bindings_lru`, `eviction_count`, and `released_native_artifact_count`; the release counter advances only for an actual native-cache removal/unload.

`clear_native_cache()` remains an explicit process-wide override. Managed ownership does not block it. If the ordinary process cache was already cleared while owners remain, later managed eviction safely drops ownership without falsely reporting an unload that no longer exists; retained executable objects can reacquire through the existing native path.

Managed eviction still rejects `parallel=True` because Windows OpenMP generated DLLs are intentionally process-pinned; cross-handle serial ownership does not prove those worker-referenced libraries can be unloaded safely. The managed policy is also deliberately not composable with `CompileBudget.max_dynamic_specializations`, because silently combining a lifetime admission cap with an evicting retention cap would make it ambiguous whether an evicted binding still consumes lifetime admission capacity.

This is coordinated targeted lifecycle release, not a global memory bound. `max_cached_specializations` remains per handle. The ownership registry coordinates only resource-managed serial native retention; it does not bound process RSS, generated-code bytes, persistent-cache size, compiler subprocesses, artifacts retained solely through unmanaged executable references, or OpenMP process-pinned libraries. See `docs/specialization-eviction.md` for the complete lifecycle and evidence boundary.

## Native compilation time controls

Native compilation exposes two distinct, composable wall-clock policies.

`compiler_timeout` bounds **one launched external C compiler process**. It is either `None` or a positive finite number of seconds. Boolean, zero, negative, NaN, and infinite values are rejected before compiler lookup. When enabled, POSIX launches the compiler in a new session and kills its process group on timeout; Windows launches a new process group and uses `taskkill /T /F` for the compiler PID before reaping the parent process. A timeout raises `NativeCompilationTimeout`, which remains a `NativeCompilationError` and exposes the exact command plus configured process timeout.

`compile_deadline` instead bounds **one native artifact acquisition attempt after a process-cache miss**. It is also either `None` or a positive finite number of seconds. The implementation converts the configured duration to one absolute monotonic deadline and carries that same deadline through persistent-cache lease acquisition and, when compilation is required, into the external compiler process. Time already spent waiting for the cross-process cache lease is therefore consumed from the compiler's remaining budget instead of starting a fresh timer.

The total deadline deliberately starts at the artifact-acquisition boundary, not at Python graph lowering, compiler-command lookup, or native execution. Process-local native-cache hits return the existing loaded artifact immediately and do not start a new deadline. A persistent on-disk cache hit still has to acquire the digest lease within the remaining total deadline, but once the verified artifact is staged no compiler is launched.

If both controls are configured, the launched compiler receives the tighter of `compiler_timeout` and the remaining `compile_deadline`. The exception type identifies which policy actually exhausted:

- `NativeCompilationTimeout`: the explicit per-process `compiler_timeout` was tighter;
- `NativeCompilationDeadlineExceeded(stage="compiler", ...)`: the remaining total deadline was tighter;
- `NativeCompilationDeadlineExceeded(stage="persistent-cache lease", ...)`: the total deadline expired before the lease could be acquired.

The deadline error remains a `NativeCompilationError` and exposes the configured total duration, failing stage, and compiler command when a compiler process had been selected. A deadline exhausted at the cache-lease stage does not launch the compiler.

Without `compile_deadline`, historical persistent-cache lease behavior is preserved. POSIX may block on the operating-system lease as before; Windows retains its independent 300-second lease-safety timeout. With an explicit total deadline, POSIX and Windows both poll non-blockingly so the caller's absolute deadline can expire first. The Windows 300-second safety bound still applies if it is tighter than a longer caller deadline.

Both controls are threaded through serial and OpenMP native compilation plus concrete, dynamic, and adaptive high-level compilation. Reusable dynamic handles retain the configured policy for each future specialization that actually requires a new native artifact. An adaptive structural-budget fallback occurs before native artifact acquisition and therefore does not consume or convert a native compilation deadline into Loop fallback.

Timeout/deadline cleanup keeps the existing artifact durability boundaries:

- transient build directories are removed after a timed-out compiler;
- persistent-cache temporary build directories are removed and no library/manifest is published from an incomplete build;
- compiler timeout and total deadline are policy, not native artifact identity, so changing either does not create another cache key;
- a verified process or persistent cache hit reuses the already-compiled artifact without launching a compiler merely because the caller supplies a tighter control.

Neither policy limits native execution after compilation. Process-tree cancellation covers ordinary descendants that remain in the launched compiler's POSIX process group or Windows process tree; it is not a sandbox guarantee for a deliberately detached process that escapes those operating-system relationships.

## Evidence boundary

The concrete report metrics are structural compiler facts, not runtime resource measurements:

- `planned_owning_storage_bytes` counts compiler-owned physical storage roots in the ordinary pre-native memory plan. It is not process RSS, peak heap/stack usage, allocator overhead, operating-system working set, OpenMP runtime memory, or a borrow-adjusted runtime footprint.
- `post_fusion_kernel_count` counts Loop IR kernels after the existing fusion planner. It is not a wall-clock cost, instruction count, profitability score, or performance estimate.

`max_dynamic_specializations` is likewise a per-handle admission count, not a measurement or hard bound on process memory, generated code bytes, number of shared libraries, persistent-cache size, global native-cache entries, compiler subprocesses, or total specializations created by other handles. It prevents one configured dynamic handle from admitting unbounded new binding identities; it does not claim that rejecting the next binding reclaims resources already created for earlier bindings.

`max_cached_specializations` is a separate per-handle retention count backed by coordinated serial artifact ownership. It proves that multiple resource-managed handles retaining one native identity do not unload that shared artifact prematurely, and that final managed-owner eviction removes the currently loaded process artifact and staging directory when the artifact is still present. It still does not establish a process-wide RSS/code-byte bound, persistent-cache bound, or permanent reclamation: unmanaged/external executable references may reacquire the identity later, and OpenMP artifacts remain outside the evicting policy.

Accordingly, `CompileBudget` is a deterministic compiler admission policy, not a security sandbox, denial-of-service guarantee, memory limiter, or benchmark. Adaptive Loop fallback also does not claim to reduce runtime memory or improve performance; it only avoids native compilation for the explicit concrete over-budget policy case while preserving verified executable semantics.

`compiler_timeout` and `compile_deadline` are wall-clock control policies, not CPU quotas, memory limits, trusted sandboxes, runtime execution timeouts, or general denial-of-service guarantees. The total deadline covers persistent-cache lease waiting and a required compiler process after a process-cache miss; it does not bound Python lowering, filesystem operations before artifact lookup, native library execution, or deliberately detached compiler descendants that escape the managed process tree. No security/isolation or performance claim is implied by either policy.

## Phase boundary

Concrete structural admission, adaptive native-or-Loop policy, lifetime dynamic-specialization admission, cross-handle managed serial artifact ownership/release, per-process compiler cancellation, and a shared lease-plus-compiler artifact-acquisition deadline are now distinct runtime controls with explicit evidence boundaries.

This closes the first cross-handle managed native-ownership phase. More retention counts, replacement-policy aliases, registry-inspection knobs, timer variants, or retry knobs would be low-value policy farming. A future runtime-control promotion should require qualitatively stronger lifecycle semantics, such as safe reclaim of process-pinned OpenMP artifacts or explicit ownership/release semantics covering ordinary unmanaged native handles; otherwise the project should move to another architectural frontier.
