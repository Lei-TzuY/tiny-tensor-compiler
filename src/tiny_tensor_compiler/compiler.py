from __future__ import annotations

import os

from .input_binding import borrow_inputs as bind_borrowed_inputs
from .ir import Module
from .loop_ir import fuse_elementwise, lower_to_loops
from .lowering import lower_to_cpu
from .native import NativeExecutable, compile_native


def compile_module(
    module: Module,
    compiler: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    borrow_inputs: bool = False,
) -> NativeExecutable:
    """Lower verified tensor IR through the native pipeline and compile it eagerly."""
    loops = fuse_elementwise(lower_to_loops(lower_to_cpu(module)))
    if borrow_inputs:
        loops = bind_borrowed_inputs(loops)
    return compile_native(loops, compiler=compiler, cache_dir=cache_dir)
