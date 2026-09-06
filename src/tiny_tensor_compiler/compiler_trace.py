from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .analysis import CompilerReport, analyze_module
from .c_abi_codegen import generate_c
from .fusion_planner import fuse_elementwise
from .input_binding import BorrowedLoopProgram
from .input_binding import borrow_inputs as bind_borrowed_inputs
from .ir import Module
from .loop_ir import lower_to_loops
from .lowering import lower_to_cpu, plan_memory
from .serialization import serialize_module
from .symbolic import has_symbolic_shapes

_TRACE_FORMAT = "tiny-tensor-compiler-trace"
_TRACE_VERSION = 1


@dataclass(frozen=True)
class CompilerTracePhase:
    """One exact deterministic compiler-phase snapshot and its UTF-8 SHA-256."""

    name: str
    text: str
    sha256: str

    @classmethod
    def capture(cls, name: str, text: str) -> CompilerTracePhase:
        if not name:
            raise ValueError("compiler trace phase name must not be empty")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return cls(name=name, text=text, sha256=digest)


@dataclass(frozen=True)
class CompilerTrace:
    """Deterministic snapshots from one concrete compiler configuration."""

    report: CompilerReport
    phases: tuple[CompilerTracePhase, ...]
    borrow_inputs: bool
    parallel: bool
    format: str = _TRACE_FORMAT
    version: int = _TRACE_VERSION

    def phase(self, name: str) -> CompilerTracePhase:
        """Return one named phase, failing loudly on an unknown or duplicate name."""
        matches = tuple(phase for phase in self.phases if phase.name == name)
        if len(matches) != 1:
            raise KeyError(f"compiler trace has no unique phase named {name!r}")
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping without weakening exact phase text."""
        return {
            "borrow_inputs": self.borrow_inputs,
            "format": self.format,
            "parallel": self.parallel,
            "phases": [asdict(phase) for phase in self.phases],
            "report": self.report.to_dict(),
            "version": self.version,
        }

    def to_json(self) -> str:
        """Return canonical JSON suitable for exact snapshots and diffs."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def trace_module(
    module: Module,
    *,
    borrow_inputs: bool = False,
    parallel: bool = False,
) -> CompilerTrace:
    """Trace the real concrete lowering/fusion/codegen pipeline without compiling C."""
    if not isinstance(module, Module):
        raise TypeError("trace_module requires a Module")
    if not isinstance(borrow_inputs, bool):
        raise TypeError("borrow_inputs must be a bool")
    if not isinstance(parallel, bool):
        raise TypeError("parallel must be a bool")
    if has_symbolic_shapes(module):
        raise ValueError(
            "trace_module requires concrete tensor shapes; specialize symbolic modules first"
        )

    tensor_ir = serialize_module(module)
    cpu = lower_to_cpu(module)
    memory = plan_memory(cpu)
    pre_fusion = lower_to_loops(cpu)
    post_fusion = fuse_elementwise(pre_fusion)
    execution = bind_borrowed_inputs(post_fusion) if borrow_inputs else post_fusion
    generated_c = generate_c(execution, parallel=parallel)

    phases = (
        CompilerTracePhase.capture("tensor_ir", tensor_ir),
        CompilerTracePhase.capture("buffer_ir", cpu.dump()),
        CompilerTracePhase.capture("memory_plan", memory.dump()),
        CompilerTracePhase.capture("pre_fusion_loop_ir", pre_fusion.dump()),
        CompilerTracePhase.capture("post_fusion_loop_ir", post_fusion.dump()),
        CompilerTracePhase.capture("execution_loop_ir", _execution_loop_snapshot(execution)),
        CompilerTracePhase.capture("generated_c", generated_c),
    )
    return CompilerTrace(
        report=analyze_module(module),
        phases=phases,
        borrow_inputs=borrow_inputs,
        parallel=parallel,
    )


def _execution_loop_snapshot(program) -> str:
    if not isinstance(program, BorrowedLoopProgram):
        return "borrowed_inputs: none\n" + program.dump()
    bindings = ",".join(
        f"input{binding.index}->p{binding.buffer}"
        for binding in program.borrowed_inputs
    )
    return f"borrowed_inputs: {bindings}\n" + program.dump()
