# Verified contiguous alias views

This phase introduces the compiler's first explicit alias model. A safe row-major reshape may now stop being an internal copy and become a read-only logical view over an existing storage root.

The change is deliberately narrower than a general strided-tensor system. It establishes verifiable storage identity and alias lifetime semantics first, while keeping all currently executable views contiguous and zero-offset.

## Storage roots and logical views

`LoopAlloc` continues to own physical storage. `LoopView` introduces a second logical buffer id with its own `TensorType` while sharing the source buffer's storage root.

A `LoopView` is valid only when:

- source and view dtypes are identical;
- source and view element counts are identical;
- the type relation is exactly the existing contiguous reshape relation;
- the source has already been written;
- the view is read-only and is never an input or kernel write destination.

The logical id may therefore have a different rank or shape from the root allocation without pretending that the allocation itself changed type.

## Alias lifetime invariant

Loop verification resolves every logical buffer to its storage root and computes each view's declaration position and last use. Any later `LoopInput` or `LoopKernel` write to that root while the view is live is rejected.

A storage root may be reused after the view's final use. This preserves the existing physical-slot reuse model while making the alias dependency explicit and executable rather than relying on convention.

Kernel input/output safety is also storage-root aware: a kernel may not read a logical view and write the same root in one operation.

## Conservative reshape-copy elision

`alias_contiguous_reshapes()` is a post-lowering rewrite. For each reshape-copy kernel it finds the reshape result's current value epoch and checks whether the source storage root is rewritten before that result's final read or return.

If the root remains stable, the copy kernel is replaced by `LoopView` and reads/returns in that value epoch are redirected to the view. If the root is overwritten, the original reshape copy remains unchanged.

A later redefinition of the reshape destination slot starts a new value epoch and is not redirected through the old view. This is important because physical slots may be reused after earlier logical values die.

The current transform intentionally leaves the already-planned destination allocation in place even when the reshape copy disappears. This phase proves alias semantics and eliminates the data movement; it does **not** claim physical-allocation-count reduction.

## Compiler ordering

The supported high-level pipeline is:

```text
Loop lowering
-> elementwise fusion
-> optional borrowed-input lifetime splitting
-> contiguous alias-view rewrite
-> generated C / native execution
```

Fusion runs before views exist and therefore does not need to reinterpret alias boundaries. Borrowed-input splitting also runs first so it can establish a dedicated read-only input storage epoch before a view points at that storage.

Calling `borrow_inputs()` on an already view-bearing program is rejected; the high-level compiler always uses the supported order. `alias_contiguous_reshapes()` accepts both ordinary and borrowed Loop programs and preserves the borrowed-input contract.

## Execution semantics

The Loop CPU executor implements `LoopView` with NumPy's contiguous `reshape`, producing a logical array view over the same storage rather than calling `np.copyto`.

Generated C lowers a view to a typed read-only pointer alias:

```c
const int32_t *p_view = p_source;
```

No reshape copy loop is emitted for an elided reshape. Native return handling resolves return types from logical buffer metadata, so a view may be returned through the existing ABI.

The public ABI remains value-oriented: terminal outputs are still copied into caller-visible output arrays. This phase therefore introduces **internal zero-copy aliasing**, not a caller-visible aliased output API.

The same representation composes with verified borrowed inputs, ordered multi-output execution, dynamic symbolic specialization, and the existing OpenMP native path. OpenMP does not schedule a view because a view performs no kernel iteration.

## Verification evidence

Regression coverage includes:

- view dtype and element-count rejection;
- rejection of root writes while a view is live;
- permission for root reuse after a view's last use;
- value-epoch-aware reshape-copy elimination;
- conservative fallback when source storage is rewritten;
- CPU execution with reshape materialization disabled;
- generated-C pointer-alias evidence;
- borrowed-input + parallel + multi-output native execution;
- dynamic symbolic reshape specialization and cache reuse;
- legacy single-output and canonical multi-output C emitter compatibility;
- Ubuntu and Windows CI on Python 3.11 and 3.13.

The phase makes no wall-clock speedup claim. CI timing is not benchmark evidence.

## Deliberately out of scope

This phase does not add:

- write-through views or in-place kernels;
- non-zero storage offsets;
- arbitrary stride vectors;
- transpose, slicing, or non-contiguous views;
- negative strides;
- caller-visible aliased native outputs;
- physical allocation elimination for dead reshape destinations;
- a profitability model or performance claim.

## Next architectural frontier

The alias lifetime model is now explicit enough to support a real layout descriptor. The next high-value CPU-verifiable phase is to generalize a logical view from “same contiguous storage, new shape” to a checked `(storage_root, offset, strides, shape)` descriptor and make one non-contiguous read-only transform executable end to end, most naturally transpose or bounded slicing.

That next phase must prove bounds, indexing, alias lifetime safety, CPU semantics, deterministic C offset generation, native execution, and interaction with borrowed inputs before claiming arbitrary strided-view support.
