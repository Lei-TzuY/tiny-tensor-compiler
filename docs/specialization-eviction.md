# Resource-managed specialization eviction

Dynamic specialization admission and dynamic specialization retention are separate policies.

`CompileBudget.max_dynamic_specializations` remains the existing fail-closed lifetime admission cap: once a handle has admitted its configured number of distinct complete bindings, a new binding is rejected. It does not reclaim resources.

The resource-managed dynamic facades add a separate `max_cached_specializations` retention limit:

- `tiny_tensor_compiler.specialization_cache.compile_resource_managed_dynamic_module(...)`
- `tiny_tensor_compiler.specialization_cache.compile_resource_managed_dynamic_gradient_module(...)`
- `tiny_tensor_compiler.specialization_cache.compile_resource_managed_dynamic_linearization(...)`
- `tiny_tensor_compiler.specialization_cache.compile_resource_managed_dynamic_vjp_module(...)`
- `tiny_tensor_compiler.specialization_cache.compile_resource_managed_dynamic_hvp_module(...)`
- `tiny_tensor_compiler.specialization_cache.compile_resource_managed_dynamic_jvp_module(...)`
- `tiny_tensor_compiler.specialization_cache.compile_resource_managed_adaptive_dynamic_module(...)`
- `tiny_tensor_compiler.specialization_cache.compile_resource_managed_adaptive_dynamic_gradient_module(...)`
- `tiny_tensor_compiler.specialization_cache.compile_resource_managed_adaptive_dynamic_linearization(...)`
- `tiny_tensor_compiler.specialization_cache.compile_resource_managed_adaptive_dynamic_vjp_module(...)`
- `tiny_tensor_compiler.specialization_cache.compile_resource_managed_adaptive_dynamic_hvp_module(...)`
- `tiny_tensor_compiler.specialization_cache.compile_resource_managed_adaptive_dynamic_jvp_module(...)`

The returned handles retain at most the configured number of specialization decisions in deterministic least-recently-used order. A cache hit refreshes that binding to most-recently-used position. `max_cached_specializations=0` is valid: a specialization may be returned to the caller while the managed handle retains no binding afterward.

## What eviction releases

Managed serial native specializations now coordinate ownership by the same process-local artifact identity already used by the native cache: compiler command, generated C source, and optional persistent-cache identity. No second artifact key is introduced.

When one resource-managed handle retains a native specialization, that handle registers one ownership reference for the artifact identity. If another ordinary or adaptive resource-managed handle retains the same identity, both owners share the already-loaded artifact. Evicting one retained specialization removes only that owner's reference while another managed owner remains; the process-cache entry, loaded shared library, and process-owned staging directory remain intact. Only eviction of the final managed owner removes the process-cache entry, closes the loaded shared library, and removes its staging directory.

Ownership is counted per handle and per retained specialization rather than as a boolean set. This keeps the lifecycle correct even if multiple bindings retained by one handle ever resolve to the same native artifact identity.

An external reference to an evicted `NativeExecutable` remains valid but is deliberately not a managed-retention owner. After the final managed owner releases an identity, such a reference reacquires that artifact on its next invocation. Without a persistent cache this may compile again; with a configured persistent cache it reloads the durable artifact without invoking the compiler. Eviction never deletes the user-configured persistent cache artifact.

Dynamic-gradient, dynamic-VJP, dynamic-HVP, and dynamic-JVP handles use the same native artifact identity and ownership registry after runtime shape specialization and autodiff. Evicting a retained gradient, VJP, HVP, or JVP specialization therefore releases the generated transformed artifact under the same final-owner rule, while an external `NativeExecutable` reference remains usable and may reacquire the artifact later.

Adaptive dynamic execution, including adaptive dynamic gradients, runtime-seeded VJPs, single-input HVPs, and forward-mode JVPs, preserves the same distinction:

- an evicted `backend="native"` specialization releases one managed native ownership reference and unloads only when it was the final managed owner;
- an evicted `backend="loop"` specialization releases only the retained backend decision because no native artifact exists.

The handles expose `retained_bindings_lru`, `eviction_count`, and `released_native_artifact_count`. The release counter increments only for an actual native-cache removal/unload, not merely because one shared managed owner was evicted.

## Managed concrete native ownership

Concrete serial native execution can now opt into the same ownership protocol without changing the historical `NativeExecutable` contract:

- `compile_resource_managed_module(...)` compiles one concrete module through the ordinary native pipeline and returns a `ResourceManagedNativeExecutable` lease.
- `manage_native_executable(existing)` adopts an already-created ordinary serial `NativeExecutable` into one explicit managed ownership reference without inventing a new artifact identity or recompiling it.

A managed concrete lease is callable like the wrapped native executable and also supports a context manager. `close()` is explicit and idempotent. Closing one of several managed owners releases only that owner's reference; closing the final managed owner removes the current process-cache entry, unloads the shared library, and removes the process-owned staging directory. A closed lease fails closed on later execution and does not silently reacquire the artifact.

The wrapped ordinary `NativeExecutable` remains an ordinary handle. If a managed lease adopted it and later performed the final managed release, the ordinary handle keeps its historical behavior and may reacquire the artifact on a later call. This separation lets callers choose deterministic managed ownership without retroactively changing every existing native handle into an owning resource object.

Managed concrete owners and resource-managed dynamic/adaptive specialization owners participate in the same identity registry. For example, evicting a dynamic specialization does not unload an artifact while a concrete lease still owns the same identity; final close of that lease then performs the real unload.

Persistent-cache artifacts are not ownership targets. Final managed close releases only process-local staging/library state; the durable artifact under `cache_dir` remains available for compiler-free reacquisition.

## Explicit global cache clearing

`clear_native_cache()` remains an explicit process-wide override. Managed ownership accounting does not prevent a caller from clearing the ordinary native process cache. If the cache was cleared while managed owners still retain specialization decisions or a concrete lease remains open, later release safely drops ownership without falsely reporting an unload that already happened. An open retained executable or concrete lease remains able to reacquire its artifact through the existing native execution path; a subsequent final managed release unloads that reacquired process artifact.

## Deliberate boundaries

Resource-managed eviction and concrete managed leases still reject `parallel=True` / `ParallelNativeExecutable`. Windows OpenMP artifacts are process-pinned because unloading generated DLL code while the OpenMP runtime may retain worker references previously caused an access violation. Cross-handle serial ownership does not prove that those worker-referenced artifacts can be reclaimed safely.

The managed policy remains separate from `CompileBudget.max_dynamic_specializations`. Combining the lifetime admission cap with `max_cached_specializations` is rejected rather than silently redefining whether an evicted binding still consumes lifetime admission capacity.

The retention limit remains per dynamic handle even though serial artifact ownership is coordinated across managed handles. This is not a process-wide bound on loaded libraries, resident memory, generated-code bytes, persistent-cache size, compiler subprocesses, or ordinary unmanaged executable references. Weak managed-owner bookkeeping prevents dead handle identities from being retained indefinitely, but handle garbage collection is not advertised as an eager artifact-unload API; explicit eviction, concrete `close()`, and `clear_native_cache()` are the proven release boundaries.

## Executable evidence

Regression coverage proves the lifecycle instead of inferring it from cache length:

- LRU ordering and refresh on cache hits;
- `max_cached_specializations=0` immediate eviction;
- process-owned serial staging directories disappear when the final managed specialization owner is evicted;
- two resource-managed dynamic/adaptive handles can retain one shared artifact and the first eviction leaves it loaded;
- dynamic and adaptive-native managed handles participate in the same ownership protocol;
- `released_native_artifact_count` increments only on the final-owner actual unload;
- explicit `clear_native_cache()` followed by managed evictions is safe and does not falsely report release;
- an externally retained evicted executable can reacquire and execute again;
- a persistent-cache-backed executable reloads after final-owner eviction without another compiler invocation;
- adaptive Loop eviction does not register or release native ownership;
- native and adaptive dynamic VJP specializations obey the same LRU, final-owner unload, external-handle reacquisition, and truthful Loop-eviction accounting as ordinary and gradient specializations;
- native retained-linearization bindings account for every primal/pushforward/pullback artifact identity under one logical LRU specialization, unload each identity only on final managed ownership, and keep adaptive Loop bundles at zero native ownership;
- a resource-managed concrete handle unloads its serial artifact on final `close()` and fails closed afterward;
- two concrete owners share one artifact and only final close unloads it;
- concrete and dynamic managed owners coordinate on one shared artifact identity;
- adopting an ordinary executable preserves that ordinary handle's later reacquire behavior;
- an open concrete lease can survive explicit global cache clearing, reacquire on execution, and then release the reacquired artifact;
- persistent-cache-backed concrete close keeps the durable artifact intact;
- invalid limits, unsupported OpenMP ownership, invalid adoption types, and ambiguous admission/retention policy combinations fail closed.

The production exact heads are verified on Ubuntu and Windows with Python 3.11 and 3.13, including actual MSVC shared-library retention, cross-owner survival, final-owner unload, ordinary-handle reacquisition, and persistent-cache reuse. CI timing is not used as performance evidence.

## Phase boundary

This closes explicit managed serial-native ownership across ordinary dynamic execution, gradient/VJP/HVP/JVP transforms, concrete native leases, and retained-linearization specializations whose one logical binding may own multiple native artifact identities. `released_native_artifact_count` continues to count actual process-cache removals/unloads rather than logical binding evictions; therefore one retained bundle eviction can release several identities, none when those identities are still shared, or zero for an adaptive Loop bundle.

Raising retention counts, adding `close()` aliases, exposing the registry as another cache-control surface, or adding replacement-policy variants would be lifecycle-policy farming rather than a new architecture milestone. Safe reclaim of process-pinned OpenMP artifacts still requires evidence that worker/runtime references can no longer execute generated code before unload. Portable retained-linearization bundle-set packaging carries one symbolic binding as a coordinated primal/tape + pushforward + pullback unit with canonical role paths, child manifest/ABI hashes, and a cross-component retained-state ABI contract. Deterministic single-file archive transport for that multi-artifact bundle-set kind is now also closed under the existing compiler-free archive verification boundary, including canonical-entry enforcement, eager child verification, atomic publication, private extraction, and cleanup. The promoted deployment frontier is content-addressed registry integration for retained-linearization archives; publisher authorization, release-channel freshness/rollback, threshold policy, and transparency remain downstream layers rather than part of archive or lifecycle ownership.
