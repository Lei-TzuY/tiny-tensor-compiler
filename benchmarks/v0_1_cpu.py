from __future__ import annotations

import argparse
import platform
import statistics
import time
from collections.abc import Callable

import numpy as np

from tiny_tensor_compiler import (
    GraphBuilder,
    compile_native,
    execute_loop,
    execute_reference,
    fuse_elementwise,
    lower_to_cpu,
    lower_to_loops,
)


def _median_seconds(action: Callable[[], object], repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        action()

    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        action()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _build_add_tree(size: int):
    builder = GraphBuilder()
    a = builder.input((size,), dtype="int32")
    b = builder.input((size,), dtype="int32")
    c = builder.input((size,), dtype="int32")
    d = builder.input((size,), dtype="int32")
    module = builder.finish((a + b) + (c + d))
    loops = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))
    return module, loops


def _rate(size: int, seconds: float) -> float:
    return size / seconds if seconds > 0.0 else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare v0.1 reference, loop-interpreter, and warm native CPU execution."
    )
    parser.add_argument("--size", type=int, default=32_768)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    if args.size <= 0:
        parser.error("--size must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    rng = np.random.default_rng(0)
    inputs = [
        rng.integers(-1000, 1001, size=args.size, dtype=np.int32)
        for _ in range(4)
    ]
    module, loops = _build_add_tree(args.size)

    reference = execute_reference(module, inputs=inputs)
    interpreted = execute_loop(loops, inputs=inputs)
    np.testing.assert_array_equal(interpreted, reference)

    compile_start = time.perf_counter()
    executable = compile_native(loops)
    compile_seconds = time.perf_counter() - compile_start

    native = executable(inputs=inputs)
    np.testing.assert_array_equal(native, reference)

    reference_seconds = _median_seconds(
        lambda: execute_reference(module, inputs=inputs),
        repeats=args.repeats,
        warmup=args.warmup,
    )
    loop_seconds = _median_seconds(
        lambda: execute_loop(loops, inputs=inputs),
        repeats=args.repeats,
        warmup=args.warmup,
    )
    native_seconds = _median_seconds(
        lambda: executable(inputs=inputs),
        repeats=args.repeats,
        warmup=args.warmup,
    )

    print(f"platform: {platform.platform()}")
    print(f"python: {platform.python_version()}")
    print(f"numpy: {np.__version__}")
    print(f"elements: {args.size}")
    print(f"native_compile_once_ms: {compile_seconds * 1_000:.3f}")
    print()
    print("backend,median_ms,elements_per_second")
    for name, seconds in (
        ("reference", reference_seconds),
        ("loop_interpreter", loop_seconds),
        ("native_warm", native_seconds),
    ):
        print(f"{name},{seconds * 1_000:.3f},{_rate(args.size, seconds):.0f}")


if __name__ == "__main__":
    main()
