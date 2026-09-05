# Verified tensor view semantics

The view subsystem now has three bounded read-only zero-copy alias forms over one storage root:

- `Tensor.view(shape)` changes only the logical C-order shape of a contiguous value;
- `Tensor.slice(axis=..., start=..., stop=..., step=...)` creates one positive-stride single-axis slice;
- `Tensor.transpose(axes)` creates one compile-time axis-permutation view by reordering logical shape and storage strides.

None of these operations creates a second physical storage allocation. Logical view handles remain distinct from their backing storage root, while verification tracks the root lifetime and storage generation independently.

## Storage layout descriptor

Physical view semantics are represented by `StorageLayout(offset, strides)` in element units.

- `offset` is a non-negative element offset from the storage root;
- every stride is a strictly positive integer;
- layout rank must match the logical tensor rank;
- the maximum reachable element must remain inside the root storage allocation;
- empty logical tensors remain valid without manufacturing a zero stride.

A root allocation has the canonical contiguous layout for its concrete shape. A whole-storage `view` derives a new contiguous logical layout at the same offset. A positive-stride slice composes from the source layout by adding `start * source_stride[axis]` to the absolute root offset and multiplying that axis stride by `step`. A transpose leaves the root-relative offset unchanged while permuting both logical dimensions and the corresponding storage strides.

For example, slicing a `(3, 6)` contiguous tensor as `axis=1, start=1, stop=6, step=2` produces logical shape `(3, 3)` with layout:

```text
offset = 1
strides = (6, 2)
```

Transposing that result with axes `(1, 0)` produces shape `(3, 3)` with:

```text
offset = 1
strides = (2, 6)
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

`transpose` is also compile-time bounded:

- `axes` must contain exactly one integer for every input rank position;
- every source axis appears exactly once;
- duplicate, missing, out-of-range, and boolean axes are rejected deterministically;
- `axes=None` is the convenience spelling for reversing all axes;
- symbolic, affine, and relational dimensions are merely moved to new positions; transpose does not add a shape equation or runtime solver rule;
- dtype is unchanged.

Reference execution uses NumPy views and requires `numpy.shares_memory()` for non-empty view/slice/transpose results. Caller-visible return values still copy through the existing result contract.

## Buffer planning and lifetimes

Buffer IR represents all alias forms with `BufferView`. A view output keeps its own logical `TensorType` but receives no physical storage assignment.

`MemoryPlan` records a `BufferAlias` with:

- the virtual view id;
- the source virtual value;
- the storage-root physical slot;
- the logical view type;
- the verified absolute `StorageLayout`.

Every direct or transitive alias use extends the storage root's lifetime. A physical slot cannot be reused while any live alias can still observe its prior contents.

Alias validation is layout-based rather than element-count-only: dtype must be preserved and every reachable logical element must remain inside the root storage bounds. A transpose therefore needs no special lifetime rule; it changes only the existing alias layout.

## Loop IR and storage generations

Loop IR separates storage roots from logical view handles:

- `LoopAlloc` identifies physical storage;
- `LoopView` identifies a logical type plus optional explicit storage layout;
- kernels may read storage roots or view handles;
- kernels may write only allocated storage roots.

A contiguous `LoopView` may omit an explicit layout and derive it from its source. Slice and transpose lowering carry their explicit absolute layouts.

Alias safety is checked by storage root, not by handle number. A kernel output may not share a storage root with any input, including sliced or transposed aliases.

Every write to a storage root advances its generation. A view captures the current generation. Reading or returning that view after the root has been rewritten is rejected as stale, including transitive slice/transpose chains.

## CPU execution

The Loop CPU backend materializes no view buffer. It creates NumPy logical views directly over the root array using:

- root buffer ownership;
- byte offset derived from the element offset;
- byte strides derived from the verified element strides.

Downstream elementwise kernels then index that logical NumPy view normally. Since transpose is encoded purely by permuted strides, it uses exactly the same path as a positive-stride slice. Borrowed external inputs remain compatible: a verified borrowed root may feed one or more view transforms without input or view materialization.

## Generated C and native execution

Generated C emits each logical view as a typed pointer alias to the root plus its absolute element offset, for example:

```c
const int32_t *p3 = p0 + 1;
```

Logical reads then use the layout strides rather than assuming the view type is physically row-major. A `(3, 3)` slice with strides `(6, 2)` computes offsets from `i0 * 6 + i1 * 2`; a transposed `(4, 2, 3)` view with strides `(1, 12, 4)` uses those same verified strides in its logical index expression.

Backend eligibility remains conservative:

- an input layout must be contiguous for the existing flat-loop/SSE2 path;
- an ordinary non-degenerate transpose is therefore non-contiguous and falls back to the general nested generated-C path;
- OpenMP may still schedule the verified outer loop of that general-C kernel;
- native returns from strided/permuted views gather logical elements into the existing contiguous caller-owned output ABI.

The internal alias is zero-copy. The public native result remains an owned/copied output array, so this phase does not expose internal storage lifetime to callers.

## Borrowed inputs

Verified borrowed inputs still split an external read epoch when the planned root storage is reused later for a write. Logical view handles remain in a separate id space and are shifted when extra borrowed storage slots are inserted, preventing storage/view id collisions.

The layout descriptor is preserved by that transform, so a borrowed input may flow directly through slice and transpose aliases into downstream CPU/native kernels without hidden normalization or view copies.

## Optimization and fusion boundary

`view`, `slice`, and `transpose` are known pure operations for DCE. Existing exact CSE continues to merge attribute-free whole-storage `view` operations; this phase intentionally does not add attribute-aware slice or transpose CSE.

Views remain explicit fusion boundaries. Elementwise fusion does not absorb or cross view/slice/transpose creation, and the planner treats alias creation from a producer as an observable later use.

## Verification evidence

Regression coverage now includes:

- typed positive-stride slices and full axis permutations with NumPy reference semantics;
- absolute offset/stride derivation with no physical allocation;
- transpose composition after an already-strided slice;
- deterministic invalid-bound, invalid-step, and invalid-permutation rejection;
- symbolic dimensions moving across axes without a new runtime shape solver;
- storage-root lifetime/generation safety inherited across transitive aliases;
- generated-C pointer aliases and arbitrary positive-stride logical indexing;
- deterministic fallback from flat/SSE2 selection for non-contiguous transpose layouts;
- borrowed-input CPU and native execution;
- ordered multi-output execution using aliases and downstream kernels;
- dynamic specialization with native-cache reuse;
- DCE purity;
- Ubuntu and Windows execution under Python 3.11 and 3.13.

These tests establish alias/layout correctness and executable zero-copy internal permutation. They do not establish a wall-clock speedup or measured memory-footprint reduction.

## Deliberately out of scope

The storage-layout abstraction is still intentionally bounded. This phase does not add:

- negative or zero strides;
- reverse slicing;
- writable/in-place alias kernels;
- partial-overlap output mutation;
- alias-aware fusion across a view;
- caller-visible native output views;
- arbitrary runtime permutation or slice bounds;
- advanced indexing/gather semantics;
- a performance or peak-memory claim.

The positive-stride read-only layout phase is now closed: contiguous reshape views, single-axis slices, and arbitrary compile-time axis permutations all use one root-relative storage-layout model. The next storage milestone should therefore change the alias model itself—such as negative-stride/reverse semantics or verifier-backed writable alias regions—rather than adding more positive-stride permutations or slice spellings.