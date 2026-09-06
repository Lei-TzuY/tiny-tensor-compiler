# Resource-managed specialization eviction

Dynamic specialization admission and dynamic specialization retention are separate policies.

`CompileBudget.max_dynamic_specializations` remains the existing fail-closed lifetime admission cap: once a handle has admitted its configured number of distinct complete bindings, a new binding is rejected. It does not reclaim resources.

The resource-managed dynamic facades add a separate `max_cached_specializations` retention limit:

- `tiny_tensor_compiler.specialization_cache.compile_resource_managed_dynamic_module(...)`
- `tiny_tensor_compiler.specialization_cache.compile_resource_managed_adaptive_dynamic_module(...)`

The returned handles retain at most the configured number of specialization decisions in deterministic least-recently-used order. A cache hit refreshes that binding to most-recently-used position. `max_cached_specializations=0` is valid: a specialization may be returned to the caller while the managed handle retains no binding afterward.

## What eviction releases

For an ordinary serial native specialization, eviction does more than remove a Python dictionary entry. The handle derives the exact process-local native-artifact cache identity from the evicted `NativeExecutable`, removes that entry under the native-cache lock, closes the loaded shared library, and removes its process-owned staging directory.

An external reference to the evicted `NativeExecutable` remains valid. On its next invocation it reacquires the same artifact identity. Without a persistent cache this may compile again; with a configured persistent cache it reloads the durable artifact without invoking the compiler. Eviction never deletes the user-configured persistent cache artifact.

Adaptive dynamic execution preserves the same distinction:

- an evicted `backend="native"` specialization releases its currently loaded serial process-local artifact;
- an evicted `backend="loop"` specialization releases only the retained backend decision because no native artifact exists.

The handles expose `retained_bindings_lru`, `eviction_count`, and `released_native_artifact_count` so those decisions remain observable.

## Deliberate boundaries

Resource-managed eviction currently rejects `parallel=True`. Windows OpenMP artifacts are process-pinned because unloading generated DLL code while the OpenMP runtime may retain worker references previously caused an access violation. Pretending that removing the specialization entry releases those artifacts would therefore be false. A future parallel-aware lifecycle policy must first provide evidence that its generated code can be reclaimed safely.

This first resource-managed policy is also kept separate from `CompileBudget.max_dynamic_specializations`. Combining the lifetime admission cap with `max_cached_specializations` is rejected in this phase rather than silently redefining whether an evicted binding still consumes lifetime admission capacity.

The retention limit is per dynamic handle. It is not a process-wide bound on loaded libraries, resident memory, generated-code bytes, persistent-cache size, compiler subprocesses, or artifacts reachable through other handles. An external executable reference can reacquire an evicted artifact, so process resources may grow again later. The policy therefore proves targeted lifecycle release; it is not a memory sandbox or denial-of-service guarantee.

## Executable evidence

Regression coverage proves the lifecycle instead of inferring it from cache length:

- LRU ordering and refresh on cache hits;
- `max_cached_specializations=0` immediate eviction;
- process-owned serial staging directories disappear when their specialization is evicted;
- an externally retained evicted executable can reacquire and execute again;
- a persistent-cache-backed executable reloads after eviction without another compiler invocation;
- adaptive native eviction releases its process artifact while adaptive Loop eviction does not increment native-release accounting;
- invalid limits, unsupported OpenMP eviction, and ambiguous admission/retention policy combinations fail closed.

The production/adaptive exact head was verified on Ubuntu and Windows with Python 3.11 and 3.13, including actual MSVC shared-library unload and reacquisition. CI timing is not used as performance evidence.

## Phase boundary

This closes the first per-handle resource-accounted specialization-eviction phase. Raising the retention count, adding more replacement-policy aliases, or exposing arbitrary cache knobs would be policy farming rather than a new architecture milestone.

A later resource-control promotion should require stronger lifecycle semantics, such as safe reclaim of process-pinned parallel artifacts or a process-wide ownership/refcount model across multiple dynamic handles. Otherwise the project should promote to a different subsystem frontier rather than expanding cache configuration surface.