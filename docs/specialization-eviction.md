# Resource-managed specialization eviction

Dynamic specialization admission and dynamic specialization retention are separate policies.

`CompileBudget.max_dynamic_specializations` remains the existing fail-closed lifetime admission cap: once a handle has admitted its configured number of distinct complete bindings, a new binding is rejected. It does not reclaim resources.

The resource-managed dynamic facades add a separate `max_cached_specializations` retention limit:

- `tiny_tensor_compiler.specialization_cache.compile_resource_managed_dynamic_module(...)`
- `tiny_tensor_compiler.specialization_cache.compile_resource_managed_adaptive_dynamic_module(...)`

The returned handles retain at most the configured number of specialization decisions in deterministic least-recently-used order. A cache hit refreshes that binding to most-recently-used position. `max_cached_specializations=0` is valid: a specialization may be returned to the caller while the managed handle retains no binding afterward.

## What eviction releases

Managed serial native specializations now coordinate ownership by the same process-local artifact identity already used by the native cache: compiler command, generated C source, and optional persistent-cache identity. No second artifact key is introduced.

When one resource-managed handle retains a native specialization, that handle registers one ownership reference for the artifact identity. If another ordinary or adaptive resource-managed handle retains the same identity, both owners share the already-loaded artifact. Evicting one retained specialization removes only that owner's reference while another managed owner remains; the process-cache entry, loaded shared library, and process-owned staging directory remain intact. Only eviction of the final managed owner removes the process-cache entry, closes the loaded shared library, and removes its staging directory.

Ownership is counted per handle and per retained specialization rather than as a boolean set. This keeps the lifecycle correct even if multiple bindings retained by one handle ever resolve to the same native artifact identity.

An external reference to an evicted `NativeExecutable` remains valid but is deliberately not a managed-retention owner. After the final managed owner releases an identity, such a reference reacquires that artifact on its next invocation. Without a persistent cache this may compile again; with a configured persistent cache it reloads the durable artifact without invoking the compiler. Eviction never deletes the user-configured persistent cache artifact.

Adaptive dynamic execution preserves the same distinction:

- an evicted `backend="native"` specialization releases one managed native ownership reference and unloads only when it was the final managed owner;
- an evicted `backend="loop"` specialization releases only the retained backend decision because no native artifact exists.

The handles expose `retained_bindings_lru`, `eviction_count`, and `released_native_artifact_count`. The release counter increments only for an actual native-cache removal/unload, not merely because one shared managed owner was evicted.

## Explicit global cache clearing

`clear_native_cache()` remains an explicit process-wide override. Managed ownership accounting does not prevent a caller from clearing the ordinary native process cache. If the cache was cleared while managed owners still retain specialization decisions, later owner eviction safely drops ownership without falsely reporting an unload that already happened. A retained executable remains able to reacquire its artifact through the existing native execution path.

## Deliberate boundaries

Resource-managed eviction still rejects `parallel=True`. Windows OpenMP artifacts are process-pinned because unloading generated DLL code while the OpenMP runtime may retain worker references previously caused an access violation. Cross-handle serial ownership does not prove that those worker-referenced artifacts can be reclaimed safely.

The managed policy remains separate from `CompileBudget.max_dynamic_specializations`. Combining the lifetime admission cap with `max_cached_specializations` is rejected rather than silently redefining whether an evicted binding still consumes lifetime admission capacity.

The retention limit remains per dynamic handle even though serial artifact ownership is coordinated across managed handles. This is not a process-wide bound on loaded libraries, resident memory, generated-code bytes, persistent-cache size, compiler subprocesses, or artifacts reachable only through ordinary unmanaged/external executable references. Weak managed-owner bookkeeping prevents dead handle identities from being retained indefinitely, but handle garbage collection is not advertised as an eager artifact-unload API; explicit eviction and `clear_native_cache()` remain the proven release boundaries.

## Executable evidence

Regression coverage proves the lifecycle instead of inferring it from cache length:

- LRU ordering and refresh on cache hits;
- `max_cached_specializations=0` immediate eviction;
- process-owned serial staging directories disappear when the final managed specialization owner is evicted;
- two ordinary resource-managed handles can retain one shared artifact and the first eviction leaves it loaded;
- ordinary and adaptive-native managed handles participate in the same ownership protocol;
- `released_native_artifact_count` increments only on the final-owner actual unload;
- explicit `clear_native_cache()` followed by managed evictions is safe and does not falsely report release;
- an externally retained evicted executable can reacquire and execute again;
- a persistent-cache-backed executable reloads after final-owner eviction without another compiler invocation;
- adaptive Loop eviction does not register or release native ownership;
- invalid limits, unsupported OpenMP eviction, and ambiguous admission/retention policy combinations fail closed.

The production exact head was verified on Ubuntu and Windows with Python 3.11 and 3.13, including actual MSVC shared-library retention and final-owner unload. CI timing is not used as performance evidence.

## Phase boundary

This closes the first cross-handle managed native-ownership phase. Raising retention counts, exposing the registry as another cache-control surface, or adding replacement-policy aliases would be policy farming rather than a new architecture milestone.

A later resource-control promotion should require qualitatively stronger lifecycle evidence, such as safe reclaim of process-pinned OpenMP artifacts or explicit ownership/release semantics that also cover ordinary unmanaged native handles. Otherwise the project should promote to another subsystem frontier.
