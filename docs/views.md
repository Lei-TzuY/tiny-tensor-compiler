# Verified tensor view and writable-alias semantics

The storage subsystem has four bounded zero-copy read-only alias transforms over one storage root plus a generation-sequenced writable effect:

- `Tensor.view(shape)` changes only the logical C-order shape of a contiguous value;
- `Tensor.slice(axis=..., start=..., stop=..., step=...)` creates one positive-stride single-axis slice;
- `Tensor.reverse(axis)` reverses one logical axis by flipping the sign of that axis storage stride;
- `Tensor.transpose(axes)` creates one compile-time axis-permutation view by reordering logical shape and storage strides;
- `copy_into(root, target, source)` copies one same-typed source tensor into a verified alias region of internally owned storage and returns the fresh full-root generation handle.

The four view transforms create no second physical storage allocation. Logical view handles remain distinct from their backing storage root, while verification tracks root lifetime and storage generation independently. Writable effects use that same model explicitly: every write consumes fresh handles, advances exactly one root generation, invalidates older aliases of that root, and produces the only fresh full-root handle that may represent the new generation. This is ordered mutation semantics, not a general in-place kernel mode.

## Storage layout descriptor

Physical view semantics are represented by `StorageLayout(offset, strides)` in element units.

- `offset` is a non-negative element offset from the storage root;
- every stride is a non-zero signed integer;
- layout rank must match the logical tensor rank;
- both the minimum and maximum reachable element offsets must remain inside the root storage allocation;
- empty logical tensors remain valid without manufacturing a zero stride;
- zero-element transformed views canonicalize their absolute offset to zero before bounds validation.

A root allocation has the canonical positive C-order layout for its concrete shape. A whole-storage `view` derives a new contiguous logical layout at the same offset. A positive-stride slice composes from the source layout by adding `start * source_stride[axis]` to the absolute root offset and multiplying that axis stride by `step`. `reverse(axis)` moves the absolute offset to the final logical element on that source axis and negates the corresponding source stride. A transpose leaves the root-relative offset unchanged while permuting both logical dimensions and the corresponding storage strides.

For example, slicing a `(3, 6)` contiguous tensor as `axis=1, start=1, stop=6, step=2` produces logical shape `(3, 3)` with layout:

```text
offset = 1
strides = (6, 2)
```

Reversing axis `1` of that slice keeps shape `(3, 3)` and produces:

```text
offset = 5
strides = (6, -2)
```

Transposing that reversed result with axes `(1, 0)` produces shape `(3, 3)` with:

```text
offset = 5
strides = (-2, 6)
```

Likewise a contiguous `(2, 3, 4)` tensor has strides `(12, 4, 1)`; `transpose((2, 0, 1))` produces shape `(4, 2, 3)` with strides `(1, 12, 4)` and no new allocation.

The descriptor is absolute relative to the storage root, so transitive views do not accumulate hidden pointer arithmetic in different backends.

## Tensor-IR contract

`view` uses the same exact element-count and symbolic-shape proof as `reshape`: source and target element-count polynomials must be identical, dtype is unchanged, and the target may not introduce a symbol absent from the source.

`slice` remains deliberately bounded:

- exactly one axis is sliced per operation;
- `axis`, `start`, `stop`, and `step` are compile-time integers;
- `step >= 1`;
- bounds satisfy `0 <= start <= stop <= extent`;
- the sliced axis extent must already be concrete in tensor IR;
- other axes may remain symbolic and specialize through the existing runtime-symbolic boundary;
- dtype is unchanged.

`reverse` is deliberately separate from the bounded `slice` syntax:

- exactly one axis is reversed per operation;
- `axis` must be a compile-time integer in range and booleans are rejected;
- logical shape and dtype are unchanged;
- the reversed axis may remain symbolic in tensor IR because its concrete storage offset is derived only after the existing runtime specialization boundary;
- generic negative-step `slice` syntax is not enabled by this operation.

`transpose` is also compile-time bounded:

- `axes` must contain exactly one integer for every input rank position;
- every source axis appears exactly once;
- duplicate, missing, out-of-range, and boolean axes are rejected deterministically;
- `axes=None` is the convenience spelling for reversing axis order, not element order;
- symbolic, affine, and relational dimensions are merely moved to new positions; transpose does not add a shape equation or runtime solver rule;
- dtype is unchanged.

`copy_into(root, target, source)` is an explicit storage effect. It requires:

- `root` to be either the internally computed owning value or the latest fresh full-root result of an earlier `copy_into` on that storage;
- the underlying owner to be compiler-owned computed storage rather than an input or constant;
- `target` to resolve to the same owning storage root through a fresh verified alias chain;
- `source` to be fresh and resolve to a different storage root;
- source and target `TensorType` values to match exactly, with no cast or broadcast;
- the result type to match the complete owning-root type and layout;
- every root, target, source, and later consumer to match the current generation of its storage root.

A write increments the owner generation exactly once. Every pre-write root or view handle for that storage becomes stale immediately; only the returned full-root handle represents the new generation. A later view may be derived from that fresh result and used by another ordered write. Ordinary pure computation may also consume the fresh generation between writes, so sequencing is defined by SSA operation order plus explicit generation checks rather than by a special terminal-only syntax. Returning or reading an older root/view generation is rejected.

Reference execution uses NumPy views and requires `numpy.shares_memory()` for non-empty view/slice/reverse/transpose results. Each `copy_into` executes with `numpy.copyto(target, source)` and then exposes the owning array as the fresh post-write value. Caller-visible return values still copy through the existing result contract.

## Buffer planning and lifetimes

Buffer IR represents read-only alias forms with `BufferView`. A view output keeps its own logical `TensorType` but receives no physical storage assignment.

Writable effects lower to ordered `BufferCopyInto` operations. Each fresh result is another logical alias of the same mutated owning root rather than a new storage allocation. Memory planning therefore treats view results and every post-write generation handle as aliases whose observable lifetimes extend the root storage lifetime.

`MemoryPlan` records a `BufferAlias` with:

- the virtual alias id;
- the source/root virtual value;
- the storage-root physical slot;
- the logical alias type;
- the verified absolute `StorageLayout`.

Every direct or transitive alias use extends the storage root's lifetime. A physical slot cannot be reused while any live alias can still observe its prior contents. Buffer verification independently tracks storage roots and generations: every operand must be fresh before use; `BufferCopyInto` requires a fresh full-root handle, a fresh same-root target, and a fresh different-root source; the write advances the owner generation and makes only its result fresh. This catches malformed low-level IR even if it bypasses tensor-IR construction.

Alias validation is layout-based rather than element-count-only: dtype must be preserved and the complete minimum/maximum reachable root-relative interval must remain inside the backing allocation. A reverse or transpose therefore needs no special lifetime rule; each changes only the existing alias layout.

## Loop IR and storage generations

Loop IR separates storage roots from logical handles:

- `LoopAlloc` identifies physical storage;
- `LoopView` identifies a logical type plus optional explicit storage layout;
- `LoopCopyInto` identifies one verified write effect plus the fresh post-write full-root handle;
- kernels may read storage roots or fresh view/generation handles;
- ordinary kernels may write only allocated storage roots.

A contiguous `LoopView` may omit an explicit layout and derive it from its source. Slice, reverse, and transpose lowering carry their explicit absolute layouts.

Alias safety is checked by storage root, not by handle number. A kernel output may not share a storage root with any input, including sliced, reversed, or transposed aliases.

Every write to a storage root advances its generation. A view captures the current generation. Reading or returning that view after the root has been rewritten is rejected as stale, including transitive slice/reverse/transpose chains. `LoopCopyInto` accepts the latest fresh full-root generation handle, resolves it to the backing `LoopAlloc`, verifies the target and source generations, advances the backing root generation, and records only the newly produced result handle as fresh. This permits ordered write chains without confusing logical generation handles with physical allocation identity.

Runtime-input roots are tracked separately and cannot be mutation roots. Writable generations therefore cannot turn verified borrowed/copy-in input storage into caller-visible mutation.

## CPU execution

The Loop CPU backend materializes no view buffer. It creates NumPy logical views directly over the root array using:

- root buffer ownership;
- byte offset derived from the element offset;
- signed byte strides derived from the verified element strides.

Downstream elementwise kernels then index that logical NumPy view normally. Positive slices, reversals, and transposes all use exactly the same layout-driven path. Borrowed external inputs remain compatible: a verified borrowed root may feed one or more view transforms without input or view materialization.

For each `LoopCopyInto`, the backend executes `np.copyto(target, source)` only after Loop IR has proven that target and source use different storage roots and exact matching types. The fresh logical output references the same mutated owning array and can safely feed later views, kernels, or another verified write. Same-root overlap is rejected by verification instead of depending on unspecified copy overlap behavior.

## Generated C and native execution

Generated C emits each read-only logical view as a typed pointer alias to the root plus its absolute element offset, for example:

```c
const int32_t *p3 = p0 + 5;
```

Logical reads then use the layout strides rather than assuming the view type is physically row-major. A `(3, 3)` reversed slice with strides `(6, -2)` computes offsets from `i0 * 6 + i1 * -2`; a transposed reversed view with strides `(-2, 6)` uses those same verified signed strides in its logical index expression.

`LoopCopyInto` is emitted as a deterministic serial copy over the verified logical target/source shapes. Destination addresses use the current fresh root handle plus the target's root-relative offset and signed strides; source addresses use the source layout. After each copy, generated C exposes a **mutable** typed pointer for the fresh full-root generation handle so a later verified `copy_into` may write through it. Read-only views derived from that handle remain `const` aliases. The copy itself is deliberately not OpenMP-scheduled.

Backend eligibility remains conservative:

- an input layout must match canonical positive C-order strides for the existing flat-loop/SSE2 path;
- any non-degenerate negative-stride reverse is therefore non-contiguous and falls back to the general nested generated-C path;
- OpenMP may still schedule the verified outer loop of ordinary general-C kernels between writable effects;
- the implicit OpenMP barrier at each kernel boundary preserves the program order before any following serial `copy_into`;
- native returns from strided/permuted/reversed views gather logical elements into the existing contiguous caller-owned output ABI.

Internal aliases remain zero-copy. Writable effects mutate only compiler-owned internal storage. The public native result remains an owned/copied output array, so neither internal alias lifetime nor writable internal storage is exposed directly to callers.

## Borrowed inputs

Verified borrowed inputs still split an external read epoch when planned storage is reused later for a write. Logical view and writable-generation handles remain in a separate id space and are shifted when extra borrowed storage slots are inserted, preventing storage/handle id collisions.

The signed layout descriptor is preserved by that transform, so a borrowed input may flow directly through slice, reverse, and transpose aliases into downstream CPU/native kernels without hidden normalization or view copies.

The borrowing transform also preserves every `LoopCopyInto` handle and remaps root/target/source references. A copy into an internally owned root counts as a write for lifetime splitting, while Loop verification continues to reject any attempt to mutate a runtime-input root. Ordered writable generations therefore remain compatible with borrowed external sources without exposing caller-owned storage to mutation.

## Optimization and fusion boundary

`view`, `slice`, `reverse`, and `transpose` are known pure operations for DCE. Existing exact CSE continues to merge attribute-free whole-storage `view` operations; this phase intentionally does not add attribute-aware slice, reverse, or transpose CSE.

`copy_into` is an explicit effect barrier. Optimizer passes that assume a pure expression graph still do not move, duplicate, canonicalize through, or fold across a module containing any writable effect. This is deliberately conservative even though the verifier now supports computation between ordered generations; effect-aware motion requires a separate dependence model and is not inferred from generation freshness alone.

Views and writable effects remain explicit fusion boundaries. Elementwise fusion does not absorb or cross view/slice/reverse/transpose creation or `copy_into`, and the planner treats alias creation from a producer as an observable later use.

## Verification evidence

Regression coverage includes:

- typed positive-stride slices, signed-stride reversals, and full axis permutations with NumPy reference semantics;
- absolute offset/stride derivation with no physical allocation;
- minimum/maximum signed-layout root-bound verification;
- reverse composition after an already-strided slice and before transpose;
- deterministic invalid-bound, invalid-step, invalid-reverse-axis, and invalid-permutation rejection;
- symbolic dimensions moving or reversing across axes without a new runtime shape solver;
- storage-root lifetime/generation safety inherited across transitive aliases;
- generated-C pointer aliases and arbitrary signed-stride logical indexing;
- deterministic fallback from flat/SSE2 selection for negative-stride layouts;
- borrowed-input CPU and native execution;
- ordered multi-output execution using aliases and downstream kernels;
- dynamic specialization with native-cache reuse, including zero extent;
- DCE purity for read-only aliases;
- writable copies through verified positive/signed-stride view layouts in reference, Loop CPU, generated C, and native execution;
- multiple ordered writes to one internal root, with pure computation and fresh-view derivation between generations;
- rejection of input/constant roots, non-full-root handles, same-root sources, type mismatches, and every stale post-write root/view use or return;
- preservation of writable effects through borrowed-input rewriting;
- native ordered-write execution with OpenMP kernels between writes on GCC-style and MSVC toolchains;
- optimizer effect-barrier behavior;
- canonical zero offsets for empty reshape/slice/reverse/transpose layouts;
- Ubuntu and Windows execution under Python 3.11 and 3.13.

These tests establish alias/layout correctness and ordered generation semantics. They do not establish a wall-clock speedup, measured memory-footprint reduction, overlap-safe mutation, or general in-place elementwise execution.

## Deliberately out of scope

The storage-layout and mutation abstraction is still intentionally bounded. This phase does not add:

- zero strides;
- generic negative-step `slice` syntax or arbitrary reverse slicing bounds;
- writable caller-owned input roots;
- same-root or partial-overlap copy semantics / `memmove` behavior;
- unordered, concurrent, or automatically rescheduled writable effects;
- in-place elementwise kernels or arbitrary destination-bearing operators;
- effect-aware optimizer motion across `copy_into`;
- alias-aware fusion across a view or write effect;
- caller-visible native output views;
- arbitrary runtime permutation or slice bounds;
- advanced indexing/gather semantics;
- a performance or peak-memory claim.

The ordered writable-generation phase removes the former one-write/terminal restriction by making freshness explicit at tensor IR, Buffer IR, and Loop IR. A fresh post-write root may feed ordinary computation, new views, and later writes; each subsequent mutation must consume the current generation, while every older root/view handle is deterministically stale. The next storage milestone should add a genuinely new effect capability—such as verifier-proven non-overlapping region writes, overlap-safe copy semantics, or a bounded in-place kernel contract—rather than merely increasing a write count.