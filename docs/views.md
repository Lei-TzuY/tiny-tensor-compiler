# Verified contiguous view semantics

This phase adds `Tensor.view(shape)` as the first alias-aware tensor/storage capability. A view changes the logical C-order shape while sharing the complete underlying storage allocation; it is not a copy kernel and it does not create a second physical buffer.

## Semantic contract

`view` uses the same exact element-count and symbolic-shape proof as `reshape`. Source and target element-count polynomials must be identical, dtype is unchanged, and the target may not introduce a symbol absent from the source.

The difference is physical semantics:

- `reshape` remains an explicit row-major copy into distinct storage;
- `view` creates a logical alias handle over the source storage root;
- this phase supports only whole-storage C-contiguous reshaping, so the alias has zero offset and no arbitrary stride metadata.

Reference execution uses `numpy.reshape(..., order="C")` and requires `numpy.shares_memory()` with the source. Loop execution creates the same NumPy view without an allocation or `copyto`. Generated C emits a typed pointer alias such as `const float *p3 = p0;` rather than a view array or element-copy loop.

Native return values still cross the existing output ABI by copy-out. Returning a view therefore does not expose internal storage ownership to the caller; it only avoids an intermediate internal materialization.

## Storage planning and lifetimes

Buffer IR represents a view with `BufferView(output, source)`. The virtual view keeps its own logical `TensorType` but receives no physical storage assignment.

`MemoryPlan` records a `BufferAlias` from the virtual view to the source storage root. Every use or return of a direct or transitive view extends the root virtual value's lifetime. A storage slot therefore cannot be reused while any alias can still observe the old contents.

Loop IR makes the split explicit:

- `LoopAlloc` identifies a physical storage root;
- `LoopView` identifies a logical shape/type handle backed by an existing root;
- kernels may read either storage handles or view handles;
- kernels may write only allocated storage roots.

Alias safety is checked by storage root, not by handle number. A kernel output may not share a storage root with any input, even when the input is reached through one or more view handles.

## Storage generations

Loop verification independently checks lifetime freshness instead of trusting the planner alone.

Every write to a storage root advances its generation. A `LoopView` captures the current generation of its source root. Reading or returning that view after the root has been rewritten is rejected as a stale alias. This catches malformed hand-built Loop IR even if it bypasses the normal memory planner.

Transitive views inherit both the same storage root and the same captured generation.

## Borrowed inputs

Verified borrowed inputs remain compatible with views. If an input's planned storage root is later reused for a write, `borrow_inputs()` still splits the external read-only epoch into a dedicated storage slot.

Because view handles live in a separate logical id space after the storage ids, the borrowing transform shifts existing view handles when it appends split storage slots. This prevents a newly allocated borrowed slot from colliding with an existing logical view handle.

A borrowed input may therefore flow directly through `view` into downstream kernels or returns without an input materialization copy or a view materialization copy.

## Optimization and fusion boundary

`view` is a known pure operation:

- DCE may remove an unused view;
- exact CSE may merge duplicate views with the same source and result type.

This phase keeps views as explicit fusion boundaries. Elementwise fusion does not absorb or cross a view operation. The fusion planner treats creation of a view from a producer as a real later use of that producer.

## Verification evidence

Regression coverage includes:

- typed view construction and C-order reference semantics;
- no physical allocation for a view;
- source-root lifetime extension across downstream view users;
- storage-root output/input alias rejection;
- stale-view rejection after a root rewrite;
- generated-C pointer aliases with no view array/copy loop;
- direct return of a logical view through the native ABI;
- borrowed-input views with storage/view id collision prevention;
- multi-output native execution using both a view and a downstream kernel;
- symbolic dynamic specialization and native-cache reuse;
- DCE/CSE behavior and an explicit fusion boundary;
- Ubuntu and Windows execution under Python 3.11 and 3.13.

These tests establish zero-copy internal view semantics and alias correctness. They do not establish a wall-clock speedup or a measured peak-memory reduction.

## Deliberately out of scope

This is not a general strided tensor subsystem. The phase does not add:

- non-zero storage offsets;
- arbitrary element strides;
- transpose/permutation views;
- slicing or negative strides;
- writable/in-place view kernels;
- overlapping partial views;
- alias-aware fusion across a view;
- caller-visible native output views;
- a performance or memory-footprint claim.

The next layout frontier should add explicit offset/stride descriptors and prove bounds/overlap/indexing semantics end-to-end before exposing transpose or slice views. Raising the number of view shapes or adding syntax aliases would be low-value farming rather than a new architecture phase.
