from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .input_validation import prepare_runtime_inputs
from .ir import Module, Value
from .symbolic import has_symbolic_shapes, specialize_for_inputs
from .verifier import verify

ExecutionResult = np.ndarray | tuple[np.ndarray, ...]


def execute_reference(module: Module, inputs: Sequence[Any] = ()) -> ExecutionResult:
    """Execute verified tensor IR directly; symbolic shapes specialize from inputs."""
    if has_symbolic_shapes(module):
        module, _ = specialize_for_inputs(module, inputs)
    verify(module)
    input_ops = tuple(op for op in module.function.ops if op.opcode == "input")
    runtime_inputs = prepare_runtime_inputs(
        tuple(op.results[0].type for op in input_ops),
        inputs,
    )
    values: dict[Value, np.ndarray] = {}

    for op in module.function.ops:
        if op.opcode == "input":
            values[op.results[0]] = np.array(runtime_inputs[op.attrs["index"]], copy=True)
        elif op.opcode == "const":
            values[op.results[0]] = np.array(op.attrs["value"], copy=True)
        elif op.opcode in {"add", "mul"}:
            dtype = op.results[0].type.dtype.to_numpy()
            lhs = values[op.operands[0]].astype(dtype, copy=False)
            rhs = values[op.operands[1]].astype(dtype, copy=False)
            fn = np.add if op.opcode == "add" else np.multiply
            values[op.results[0]] = np.asarray(fn(lhs, rhs), dtype=dtype)
        elif op.opcode == "relu":
            dtype = op.results[0].type.dtype.to_numpy()
            operand = values[op.operands[0]].astype(dtype, copy=False)
            values[op.results[0]] = np.maximum(operand, np.array(0, dtype=dtype))
        elif op.opcode == "return":
            outputs = tuple(np.array(values[operand], copy=True) for operand in op.operands)
            return outputs[0] if len(outputs) == 1 else outputs
    raise RuntimeError("verified module unexpectedly has no return")
