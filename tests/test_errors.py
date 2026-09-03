import numpy as np
import pytest

from tiny_tensor_compiler import GraphBuilder, TypeInferenceError, VerificationError, verify
from tiny_tensor_compiler.ir import DType, TensorType


def test_rejects_non_broadcastable_shapes():
    builder = GraphBuilder()
    lhs = builder.tensor([1, 2])
    rhs = builder.tensor([1, 2, 3])
    with pytest.raises(TypeInferenceError, match="cannot broadcast shapes"):
        _ = lhs + rhs


def test_rejects_unsupported_dtype():
    builder = GraphBuilder()
    with pytest.raises(TypeInferenceError, match="unsupported tensor dtype"):
        builder.tensor([True, False])


def test_verifier_rejects_malformed_result_type():
    builder = GraphBuilder()
    x = builder.tensor([1, 2, 3])
    module = builder.finish(x + 1)
    add = next(op for op in module.function.ops if op.opcode == "add")
    add.results[0].type = TensorType((999,), DType.INT64)

    with pytest.raises(VerificationError, match="does not match inferred type"):
        verify(module)


def test_verifier_rejects_corrupt_use_def_tracking():
    builder = GraphBuilder()
    x = builder.tensor(np.array([1, 2, 3], dtype=np.int32))
    module = builder.finish(x.relu())
    relu = next(op for op in module.function.ops if op.opcode == "relu")
    relu.operands[0].uses.clear()

    with pytest.raises(VerificationError, match="use-def tracking mismatch"):
        verify(module)


def test_graph_builder_rejects_cross_graph_values():
    left = GraphBuilder()
    right = GraphBuilder()
    x = left.tensor([1])
    y = right.tensor([2])

    with pytest.raises(ValueError, match="different GraphBuilder"):
        _ = x + y
