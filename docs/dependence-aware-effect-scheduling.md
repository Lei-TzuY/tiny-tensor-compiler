# Dependence-aware write-effect scheduling

The native `parallel=True` path can now execute consecutive verified mutation effects concurrently when their storage-root access sets are pairwise independent. This extends OpenMP from parallelism *inside* one kernel/effect loop to a bounded form of parallelism *across* multiple ordered write effects.

This is an executable scheduling/correctness capability. It is not a wall-clock speedup claim and does not use CI duration as performance evidence.

## Effect access model

The first scheduler is deliberately root-conservative. Each `copy_into`, partial `binary_into`, or full-root `binary_inplace` effect contributes:

- one written storage root: the current destination root;
- a read of that destination root, conservatively, because partial and in-place updates preserve or consume existing destination contents;
- a read of the source storage root.

Two effects may share one group only when these sets have no RAW, WAR, or WAW conflict. Shared read-only source roots are allowed. Different logical views of the same storage root are therefore treated as dependent in this phase even when their physical regions could later be proven disjoint.

The planner does not move operations. It scans only already-consecutive mutation effects. Any non-effect operation closes the current group, including ordinary pure kernels, views, returns, and the explicit pure materialization inserted by `casting="widen"`. A detected root hazard also closes the current group and starts a new barrier level.

## OpenMP lowering

A group containing two or more independent effects lowers to one barriered region:

```c
#pragma omp parallel sections
{
    #pragma omp section
    {
        /* first serial effect body */
    }
    #pragma omp section
    {
        /* second serial effect body */
    }
}
```

Each section uses the existing serial write emitter. The scheduler intentionally does not place `parallel for` inside a section, so this phase does not introduce nested OpenMP regions or a second write-index calculation. Broadcast `source_map` indexing, signed target strides, transpose/reverse layouts, and exact writable-generation semantics therefore stay owned by the same current write emitter.

The implicit barrier at the end of `parallel sections` publishes every completed mutation before the fresh effect-result pointer aliases are exposed in original IR order. Later operations therefore observe the same generation ordering as the serial program.

Singleton groups keep the previously verified behavior: eligible `copy_into` and partial `binary_into` effects may use their existing outer-loop `parallel for`; `binary_inplace` remains serial when it is not part of a multi-effect sections group.

## Interaction with current write semantics

The scheduler sits after all existing correctness transformations and does not relax them:

- borrowed runtime input roots remain read-only;
- same-root RHS snapshot canonicalization still happens before writable-effect IR;
- `copy_into` / `binary_into` broadcast source maps remain verifier-owned;
- `casting="widen"` still materializes an exact destination-typed RHS before the write effect;
- storage-root generation freshness remains enforced by Loop IR;
- returned values retain their existing ownership/lifetime behavior.

Because widening is an ordinary pure kernel, it also acts as a scheduling boundary in this first consecutive-only planner. No implicit cast or mixed-dtype write is moved into a sections region.

## Evidence

Regression coverage proves that:

- independent partial effects on different roots can share a read-only source and enter one sections group;
- mixed `copy_into` and `binary_inplace` effects on independent roots can share a group;
- broadcast `copy_into` and broadcast `binary_into` preserve current `source_map` semantics under native sections execution;
- generated sections contain serial effect bodies rather than nested `parallel for` directives;
- a cross-root RAW dependence forces separate barrier levels;
- borrowed runtime inputs remain unchanged while independently mutated internal roots produce the expected native outputs.

The production candidate is validated on Ubuntu and Windows with Python 3.11 and 3.13, including GCC-style and MSVC OpenMP compilation/execution.

## Deliberate boundary and next promotion

This phase does not add:

- reordering across non-effect operations;
- region/subview disjointness proofs for writes sharing one storage root;
- OpenMP tasks, `nowait`, asynchronous returns, or nested OpenMP scheduling;
- a profitability, grain-size, or thread-count cost model;
- a performance claim.

The next scheduling/storage promotion should require genuinely new dependence information rather than more effect spellings. Strong candidates are region-aware disjoint-write analysis on one storage root, or a broader operation-access model that can prove when an effect may cross intervening pure work. A second executable ISA/backend or accelerator backend remains an independent architectural frontier when its toolchain/hardware can be validated.