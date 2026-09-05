# Verified tensor view semantics

The view subsystem now has two bounded zero-copy alias forms over one storage root:

- `Tensor.view(shape)` changes only the logical C-order shape of a contiguous value;
- `Tensor.slice(axis=..., start=..., stop=..., step=...)` creates one read-only positive-stride single-axis slice.

Neither operation creates a second physical storage allocation. Logical view handles remain distinct from their backing storage root, while verification tracks the root lifetime and storage generation independently.

## Storage layout descriptor

Physical view semantics are represented by `StorageLayout(offset, strides)` in element units.

- `offset` is a non-negative element offset from the storage root;
- every stride is a strictly positive integer;
- layout rank must match the logical tensor rank;
- the maximum reachable element must remain inside the root storage allocation;
- empty logical tensors remain valid without manufacturing a zero stride.

A root allocation has the canonical contiguous layout for its concrete shape. A whole-storage `view` derives a new contiguous logical layout at the same offset. A positive-stride slice composes from the source layout by adding `start * source_stride[axis]` to the absolute root offset and multiplying that axis stride by `step`.

For example, slicing a `(3, 6)` contiguous tensor as `axis=1, start=1, stop=6, step=2` produces logical shape `(3, 3)` with layout:

```text
offset = 1
strides = (6, 2)
```

The descriptor is absolute relative to the storage root, so transitive views do not accumulate hidden pointer arithmetic in different backends.

## Tensor-IR contract

`view` uses the same exact element-count and symbolic-shape proof as `reshape`: source and target element-count polynomials must be identical, dtype is unchanged, and the target may not introduce a symbol absent from the source.

`slice` is deliberately narrower in this phase:

- exactly one axis is sliced per operation;
- `axis`, `start`, `stop`, and `step` are compile-time integers;
- `step >= 1`;
- bounds satisfy `0 <= start <= stop <= extent`;
- the sliced axis extent must already be concrete in tensor IR;
- other axes may remain symbolic and specialize through the existing runtime-symbolic boundary;
- dtype is unchanged.

This keeps runtime solving independent from slice-bound normalization while still allowing shapes such as `(B, 6)` to slice the concrete second axis and later specialize `B`.

Reference execution uses NumPy views and requires `numpy.shares_memory()` for non-empty results. Caller-visible return values still copy through the existing result contract.

## Buffer planning and lifetimes

Buffer IR represents both alias forms with `BufferView`. A view output keeps its own logical `TensorType` but receives no physical storage assignment.

`MemoryPlan` records a `BufferAlias` with:

- the virtual view id;
- the source virtual value;
- the storage-root physical slot;
- the logical view type;
- the verified absolute `StorageLayout`.

Every direct or transitive alias use extends the storage root's lifetime. A physical slot cannot be reused while any live alias can still observe its prior contents.

Alias validation is no longer based on equal element count alone: the layout descriptor must preserve dtype and remain within the root storage bounds.

## Loop IR and storage generations

Loop IR separates storage roots from logical view handles:

- `LoopAlloc` identifies physical storage;
- `LoopView` identifies a logical type plus optional explicit storage layout;
- kernels may read storage roots or view handles;
- kernels may write only allocated storage roots.

A contiguous `LoopView` may omit an explicit layout and derive it from its source. Lowered slice views carry their explicit absolute layout.

Alias safety is checked by storage root, not by handle number. A kernel output may not share a storage root with any input, including a strided view.

Every write to a storage root advances its generation. A view captures the current generation. Reading or returning that view after the root has been rewritten is rejected as stale, including transitive aliases.

## CPU execution

The Loop CPU backend materializes no slice buffer. It creates a NumPy logical view directly over the root array using:

- root buffer ownership;
- byte offset derived from the element offset;
- byte strides derived from the verified element strides.

Downstream elementwise kernels then index that logical NumPy view normally. Borrowed external inputs remain compatible: a verified borrowed root may feed one or more views without input materialization or view materialization.

## Generated C and native execution

Generated C emits each logical view as a typed pointer alias to the root plus its absolute element offset, for example:

```c
const int32_t *p3 = p0 + 1;
```

Logical reads then use the layout strides rather than assuming the view type is physically row-major. A `(3, 3)` slice with strides `(6, 2)` therefore computes source offsets from `i0 * 6 + i1 * 2`.

This changes backend eligibility conservatively:

- an input layout must be contiguous for the existing flat-loop/SSE2 path;
- a strided slice therefore falls back to the general nested generated-C path;
- OpenMP may still schedule the verified outer loop of that general-C kernel;
- native returns from strided views gather logical elements into the existing contiguous caller-owned output ABI.

The internal alias is zero-copy. The public native result remains an owned/copied output array, so this phase does not expose internal storage lifetime to callers.

## Borrowed inputs

Verified borrowed inputs still split an external read epoch when the planned root storage is reused later for a write. Logical view handles remain in a separate id space and are shifted when extra borrowed storage slots are inserted, preventing storage/view id collisions.

The layout descriptor is preserved by that transform, so a borrowed input may flow directly into a positive-stride slice and downstream native kernels without a hidden normalization or slice copy.

## Optimization and fusion boundary

Both `view` and `slice` are known pure operations for DCE. Existing exact CSE continues to merge attribute-free whole-storage `view` operations; this phase does not add attribute-aware slice CSE.

Views remain explicit fusion boundaries. Elementwise fusion does not absorb or cross a view/slice creation, and the planner still treats creation of an alias from a producer as an observable later use.

## Verification evidence

Regression coverage now includes:

- typed positive-stride slice construction and NumPy reference semantics;
- absolute offset/stride layout derivation with no physical allocation;
- positive canonical strides for zero-extent storage layouts;
- deterministic invalid-bound, non-positive-step, and symbolic-sliced-axis rejection;
- storage-root lifetime/generation safety inherited from the contiguous-view phase;
- generated-C offset pointer aliases and strided logical indexing;
- fallback from flat/SSE2 selection for non-contiguous input layouts;
- borrowed-input CPU and native execution;
- ordered multi-output execution using both a slice and a downstream kernel;
- dynamic specialization on unsliced symbolic axes with native-cache reuse;
- DCE purity;
- Ubuntu and Windows execution under Python 3.11 and 3.13.

These tests establish alias/layout correctness and executable zero-copy internal slicing. They do not establish a wall-clock speedup or measured memory-footprint reduction.

## Deliberately out of scope

The storage-layout abstraction is intentionally bounded. This phase does not add:

- negative or zero strides;
- reverse slicing;
- transpose/permutation views;
- multi-axis slicing in one operation;
- writable/in-place alias kernels;
- partial-overlap output mutation;
- alias-aware fusion across a view;
- caller-visible native output views;
- arbitrary runtime slice bounds;
- a performance or peak-memory claim.

The next layout phase should add a genuinely new layout transformation such as verified axis permutation/transpose or a broader multi-axis view model, but only after preserving the same root-bound, generation, and backend-indexing invariants. Merely adding more slice spellings or step values would be low-value farming.