# Verified tensor view and writable-alias semantics

The storage subsystem has four bounded zero-copy read-only alias transforms over one storage root plus one deliberately narrow writable effect:

- `Tensor.view(shape)` changes only the logical C-order shape of a contiguous value;
- `Tensor.slice(axis=..., start=..., stop=..., step=...)` creates one positive-stride single-axis slice;
- `Tensor.reverse(axis)` reverses one logical axis by flipping the sign of that axis storage stride;
- `Tensor.transpose(axes)` creates one compile-time axis-permutation view by reordering logical shape and storage strides;
- `copy_into(root, target, source)` is the first explicit writable-alias operation and copies one same-typed source tensor into a verified alias region of internally owned storage before returning a fresh full-root handle.

The four view transforms create no second physical storage allocation. Logical view handles remain distinct from their backing storage root, while verification tracks root lifetime and storage generation independently. The writable operation does not weaken that model: it is a verifier-controlled terminal effect, not a general in-place execution mode.

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

The first writable operation is intentionally stricter than the read-only aliases. `copy_into(root, target, source)` requires:

- `root` to be internally computed owning storage rather than an input, constant, or alias handle;
- `target` to resolve to that exact owning root through the verified alias chain;
- `source` to resolve to a different storage root;
- source and target `TensorType` values to match exactly, with no cast or broadcast;
- the result type to match the full owning root type;
- at most one `copy_into` in this phase;
- the write to be the final effect immediately before `return`;
- the fresh result to be returned, with no stale alias of the previous root generation returned alongside it.

Reference execution uses NumPy views and requires `numpy.shares_memory()` for non-empty view/slice/reverse/transpose results. `copy_into` executes with `numpy.copyto(target, source)` and then exposes the owning root as the fresh post-write value. Caller-visible return values still copy through the existing result contract.

## Buffer planning and lifetimes

Buffer IR represents read-only alias forms with `BufferView`. A view output keeps its own logical `TensorType` but receives no physical storage assignment.

The writable effect lowers to `BufferCopyInto`. Its fresh result is another logical alias of the mutated owning root rather than a new storage allocation. Memory planning therefore treats both view results and post-write results as aliases whose observable lifetimes extend the root storage lifetime.

`MemoryPlan` records a `BufferAlias` with:

- the virtual alias id;
- the source/root virtual value;
- the storage-root physical slot;
- the logical alias type;
- the verified absolute `StorageLayout`.

Every direct or transitive alias use extends the storage root's lifetime. A physical slot cannot be reused while any live alias can still observe its prior contents. For `BufferCopyInto`, target and source values must already be live, the target must resolve to the owning root, and the source must resolve to a different root.

Alias validation is layout-based rather than element-count-only: dtype must be preserved and the complete minimum/maximum reachable root-relative interval must remain inside the backing allocation. A reverse or transpose therefore needs no special lifetime rule; each changes only the existing alias layout.

## Loop IR and storage generations

Loop IR separates storage roots from logical handles:

- `LoopAlloc` identifies physical storage;
- `LoopView` identifies a logical type plus optional explicit storage layout;
- `LoopCopyInto` identifies one verified write effect plus the fresh post-write root handle;
- kernels may read storage roots or view handles;
- ordinary kernels may write only allocated storage roots.

A contiguous `LoopView` may omit an explicit layout and derive it from its source. Slice, reverse, and transpose lowering carry their explicit absolute layouts.

Alias safety is checked by storage root, not by handle number. A kernel output may not share a storage root with any input, including sliced, reversed, or transposed aliases.

Every write to a storage root advances its generation. A view captures the current generation. Reading or returning that view after the root has been rewritten is rejected as stale, including transitive slice/reverse/transpose chains. `LoopCopyInto` uses the same rule explicitly: root, target, and source must all be fresh before the write; the root generation then advances and only the newly produced result handle receives the new generation. Pre-write handles cannot be consumed afterward.

Runtime-input roots are tracked separately and cannot be mutation roots. The writable phase therefore cannot turn verified borrowed/copy-in input storage into caller-visible mutation.

## CPU execution

The Loop CPU backend materializes no view buffer. It creates NumPy logical views directly over the root array using:

- root buffer ownership;
- byte offset derived from the element offset;
- signed byte strides derived from the verified element strides.

Downstream elementwise kernels then index that logical NumPy view normally. Positive slices, reversals, and transposes all use exactly the same layout-driven path. Borrowed external inputs remain compatible: a verified borrowed root may feed one or more view transforms without input or view materialization.

For `LoopCopyInto`, the backend executes `np.copyto(target, source)` only after Loop IR has proven that target and source use different storage roots and exact matching types. The fresh logical output then references the mutated owning root. Same-root overlap is rejected by verification instead of depending on unspecified copy overlap behavior.

## Generated C and native execution

Generated C emits each logical view as a typed pointer alias to the root plus its absolute element offset, for example:

```c
const int32_t *p3 = p0 + 5;
```

Logical reads then use the layout strides rather than assuming the view type is physically row-major. A `(3, 3)` reversed slice with strides `(6, -2)` computes offsets from `i0 * 6 + i1 * -2`; a transposed reversed view with strides `(-2, 6)` uses those same verified signed strides in its logical index expression.

`LoopCopyInto` is emitted as a deterministic serial copy over the verified logical target/source shapes. Destination addresses use the target's root-relative offset and signed strides; source addresses use the source layout. After the copy, generated C exposes a fresh pointer handle to the owning root. The copy itself is deliberately not OpenMP-scheduled in this phase.

Backend eligibility remains conservative:

- an input layout must match canonical positive C-order strides for the existing flat-loop/SSE2 path;
- any non-degenerate negative-stride reverse is therefore non-contiguous and falls back to the general nested generated-C path;
- OpenMP may still schedule the verified outer loop of ordinary general-C kernels;
- native returns from strided/permuted/reversed views gather logical elements into the existing contiguous caller-owned output ABI.

Internal read-only aliases remain zero-copy. A writable alias effect mutates only compiler-owned internal storage. The public native result remains an owned/copied output array, so neither internal alias lifetime nor writable internal storage is exposed directly to callers.

## Borrowed inputs

Verified borrowed inputs still split an external read epoch when planned storage is reused later for a write. Logical view handles remain in a separate id space and are shifted when extra borrowed storage slots are inserted, preventing storage/view id collisions.

The signed layout descriptor is preserved by that transform, so a borrowed input may flow directly through slice, reverse, and transpose aliases into downstream CPU/native kernels without hidden normalization or view copies.

The borrowing transform also preserves `LoopCopyInto` handles and remaps their root/target/source references. A copy into an internally owned root counts as a write for lifetime splitting, while Loop verification continues to reject any attempt to mutate a runtime-input root.

## Optimization and fusion boundary

`view`, `slice`, `reverse`, and `transpose` are known pure operations for DCE. Existing exact CSE continues to merge attribute-free whole-storage `view` operations; this phase intentionally does not add attribute-aware slice, reverse, or transpose CSE.

`copy_into` is an explicit effect barrier. Optimizer passes that assume a pure expression graph do not move, duplicate, canonicalize through, or fold across a module containing this effect in the first writable phase. This is conservative by design; effect-aware motion is a later problem, not a reason to weaken mutation ordering now.

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
- terminal writable copy through verified view layouts in reference, Loop CPU, generated C, and native execution;
- rejection of input/constant roots, non-owning roots, same-root sources, type mismatches, multiple or non-terminal writes, and stale post-write handles;
- preservation of writable effects through borrowed-input rewriting;
- optimizer effect-barrier behavior;
- canonical zero offsets for empty reshape/slice/reverse/transpose layouts;
- Ubuntu and Windows execution under Python 3.11 and 3.13.

These tests establish alias/layout correctness and one bounded executable writable-storage effect. They do not establish a wall-clock speedup, measured memory-footprint reduction, general in-place semantics, or overlap-safe mutation.

## Deliberately out of scope

The storage-layout and mutation abstraction is still intentionally bounded. This phase does not add:

- zero strides;
- generic negative-step `slice` syntax or arbitrary reverse slicing bounds;
- more than one ordered writable effect;
- non-terminal or freely scheduled mutation;
- writable caller-owned input roots;
- same-root or partial-overlap copy semantics / `memmove` behavior;
- general in-place elementwise kernels;
- alias-aware fusion across a view or write effect;
- caller-visible native output views;
- arbitrary runtime permutation or slice bounds;
- advanced indexing/gather semantics;
- a performance or peak-memory claim.

The first writable-alias phase is closed at one terminal `copy_into`: the compiler can now mutate an internally owned root through a verified view while preserving generation freshness across tensor IR, Buffer IR, Loop IR, reference/CPU execution, generated C/native execution, input borrowing, and optimizer ordering. The next storage milestone should expand mutation only together with an explicit effect/lifetime model—for example multiple ordered generations or carefully proven non-overlapping regions—rather than merely removing the one-write/terminal guards.