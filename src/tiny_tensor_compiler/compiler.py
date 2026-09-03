from __future__ import annotations

import os

from .ir import Module
from .loop_ir import fuse_elementwise, lower_to_loops
from .lowering import lower_to_cpu
from .native import NativeExecutable, compile_native


def compile_module(
    module: Module,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> NativeExecutable:
    """Lower verified tensor IR through the native pipeline and compile it eagerly."""
    loops = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))
    return compile_native(loops, compiler=compiler, cache_dir=cache_dir)
