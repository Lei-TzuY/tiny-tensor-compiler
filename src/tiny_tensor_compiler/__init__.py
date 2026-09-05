from .backends.cpu import execute as execute_cpu
from .backends.cpu import execute_loop
from .c_abi_codegen import generate_c
from .compiler import (
    DynamicExecutable,
    compile_dynamic_module,
    compile_module,
)
from .frontend import GraphBuilder, Tensor
from .fusion_planner import fuse_elementwise
from .inference import TypeInferenceError
from .input_binding import BorrowedInput, BorrowedLoopProgram, borrow_inputs
from .ir import SymbolicDim
from .loop_ir import (
    IndexMap,
    LoopAlloc,
    LoopInput,
    LoopKernel,
    LoopProgram,
    LoopReturn,
    lower_to_loops,
)
from .lowering import (
    BufferAlloc,
    BufferAssignment,
    BufferInput,
    BufferKernel,
    BufferReturn,
    CPUProgram,
    MemoryPlan,
    lower_to_cpu,
    plan_memory,
)
from .native import (
    NativeCompilationError,
    NativeExecutable,
    clear_native_cache,
    compile_native,
    execute_native,
)
from .passes import (
    algebraic_simplify,
    canonicalize,
    common_subexpression_eliminate,
    constant_fold,
    dead_code_eliminate,
)
from .runtime import execute_reference
from .symbolic import (
    SymbolicShapeError,
    bind_dynamic_batch,
    has_symbolic_shapes,
    specialize_module,
)
from .verifier import VerificationError, verify

__all__ = [
    "BorrowedInput",
    "BorrowedLoopProgram",
    "BufferAlloc",
    "BufferAssignment",
    "BufferInput",
    "BufferKernel",
    "BufferReturn",
    "CPUProgram",
    "DynamicExecutable",
    "GraphBuilder",
    "IndexMap",
    "LoopAlloc",
    "LoopInput",
    "LoopKernel",
    "LoopProgram",
    "LoopReturn",
    "MemoryPlan",
    "NativeCompilationError",
    "NativeExecutable",
    "SymbolicDim",
    "SymbolicShapeError",
    "Tensor",
    "TypeInferenceError",
    "VerificationError",
    "algebraic_simplify",
    "bind_dynamic_batch",
    "borrow_inputs",
    "canonicalize",
    "clear_native_cache",
    "common_subexpression_eliminate",
    "compile_dynamic_module",
    "compile_module",
    "compile_native",
    "constant_fold",
    "dead_code_eliminate",
    "execute_cpu",
    "execute_loop",
    "execute_native",
    "execute_reference",
    "fuse_elementwise",
    "generate_c",
    "has_symbolic_shapes",
    "lower_to_cpu",
    "lower_to_loops",
    "plan_memory",
    "specialize_module",
    "verify",
]
