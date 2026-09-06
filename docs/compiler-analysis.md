# Deterministic compiler analysis reports

`tiny_tensor_compiler.analysis.analyze_module()` exposes a read-only structural report for a verified **concrete** tensor module. The report is derived by running the repository's existing lowering pipeline rather than by maintaining a second model of compiler behavior:

```text
Module
-> lower_to_cpu()
-> plan_memory()
-> lower_to_loops()
-> fuse_elementwise()
-> CompilerReport
```

The analysis API does not rewrite the module, change optimization decisions, compile native code, or execute user data.

## Basic use

```python
from tiny_tensor_compiler.analysis import analyze_module
from tiny_tensor_compiler.frontend import GraphBuilder

builder = GraphBuilder("report-demo")
x = builder.input((8,), "int32")
y = builder.input((8,), "int32")
module = builder.finish((x, y, (x + y).relu()))

report = analyze_module(module)
print(report.planned_owning_storage_bytes)
print(report.post_fusion_kernel_counts)
print(report.to_json())
```

`CompilerReport.to_json()` uses sorted keys and compact separators so the same report object serializes deterministically. The top-level `format` is `tiny-tensor-compiler-report` and the current `version` is `1`.

## Reported facts

The v1 report records:

- function name and ordered input/output counts;
- tensor-IR operation histogram;
- logical allocated-value count and the sum of their logical tensor byte sizes;
- physical compiler-owned storage-root count and byte size after the actual memory planner;
- one deterministic `StorageSlotReport` per owning physical slot;
- alias-value and Loop-IR view counts without charging aliases as new owning storage;
- explicit mutation-effect counts for `copy_into`, partial `binary_into`, and full-root `binary_inplace` effects;
- Loop-kernel counts and opcode histograms before and after the existing fusion planner;
- fused-kernel count, including both the historical primitive `relu_add` / `relu_mul` forms and structured `FusedExpression` kernels;
- the number of Loop kernels structurally eliminated by fusion.

These are compiler-structure facts, not measurements of host-process behavior.

## Storage accounting boundary

`planned_owning_storage_bytes` sums the concrete `MemoryPlan.physical_types` roots. A row-major or strided view that aliases an existing root contributes a logical value and an alias entry but does **not** add the root's bytes again. This makes the field useful for understanding the compiler's ordinary physical storage plan without confusing logical tensor volume with allocated owning storage.

The report describes the standard pre-native ownership plan produced before the optional `borrow_inputs()` transform. It therefore does not claim to be a borrow-specific allocation report. It also excludes native shared-library code, Python/NumPy object overhead, allocator metadata, OpenMP runtime state, caches, stacks, and operating-system memory.

Accordingly, `planned_owning_storage_bytes` is **not** peak RSS, heap peak, working-set size, or a controlled runtime-memory benchmark.

## Fusion and effect boundary

The report asks the real fusion planner for the post-fusion `LoopProgram`. It does not infer fusion from tensor-IR syntax. A graph that cannot legally fuse because of physical aliasing, liveness, indexing, dtype, or other existing planner rules is reported as unfused.

Likewise, `kernels_eliminated_by_fusion` and `fused_kernel_count` are structural transformation counts. They do not imply a wall-clock speedup, vectorization, parallel execution, or profitability result.

Mutation effects are counted as effects and are not treated as pure kernels. Reporting them does not relax generation checks, alias proofs, signed-stride injectivity, snapshot rules, or OpenMP scheduling safety.

## Dynamic-shape boundary

`analyze_module()` requires concrete tensor extents. A module containing unresolved `SymbolicDim`, `AffineDim`, or `LinearDim` shapes is rejected with an instruction to specialize it first. This keeps byte accounting exact and prevents an "unknown" symbolic extent from being presented as a precise memory value.

After normal runtime shape solving and specialization produce a concrete `Module`, that concrete module can be analyzed normally.

## Evidence scope

The regression suite covers:

- an alias-safe add -> ReLU fusion and the actual pre/post fusion counts;
- a zero-copy logical view whose storage is charged only to its owning root;
- a graph combining a strided view, partial binary mutation, transpose view, reduction, and multiple outputs;
- canonical JSON stability across repeated analysis;
- fail-closed rejection of unspecialized symbolic modules.

The analysis layer is intentionally observational. Performance benchmarks, runtime memory profiling, hardware counters, compiler timing, and borrow-specific storage accounting remain separate concerns and must use their own evidence rather than being inferred from this report.
