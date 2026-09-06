from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .inference import TypeInferenceError, infer_binary
from .ir import DType, Module, Operation, TensorType, Value


@dataclass(frozen=True)
class DirectMatmulPattern:
    """One exact reshape/mul/sum composition safe to contract during lowering."""

    reduction: Operation
    lhs: Value
    rhs: Value
    intermediates: tuple[Operation, ...]


def infer_direct_matmul_type(lhs: TensorType, rhs: TensorType) -> TensorType:
    """Infer the exact rank-2 matmul result used by direct physical lowering."""
    if len(lhs.shape) != 2 or len(rhs.shape) != 2:
        raise TypeInferenceError("direct matmul requires rank-2 tensor inputs")
    m, k = lhs.shape
    rhs_k, n = rhs.shape
    if k != rhs_k:
        raise TypeInferenceError(
            f"direct matmul inner dimensions must match exactly, got {k} and {rhs_k}"
        )
    try:
        dtype = DType.from_numpy(np.result_type(lhs.dtype.to_numpy(), rhs.dtype.to_numpy()))
    except TypeError as exc:
        raise TypeInferenceError(str(exc)) from exc
    return TensorType((m, n), dtype)


def find_direct_matmul_patterns(
    module: Module,
) -> tuple[dict[Operation, DirectMatmulPattern], frozenset[Operation]]:
    """Find non-overlapping compiler-owned matmul compositions with private intermediates."""
    matches: dict[Operation, DirectMatmulPattern] = {}
    skipped: set[Operation] = set()
    for op in module.function.ops:
        pattern = _match_reduction(op)
        if pattern is None:
            continue
        if any(intermediate in skipped for intermediate in pattern.intermediates):
            continue
        matches[op] = pattern
        skipped.update(pattern.intermediates)
    return matches, frozenset(skipped)


def _match_reduction(reduction: Operation) -> DirectMatmulPattern | None:
    if (
        reduction.opcode != "sum"
        or reduction.attrs != {"axis": 1}
        or len(reduction.operands) != 1
        or len(reduction.results) != 1
    ):
        return None
    product = reduction.operands[0].producer
    if (
        product is None
        or product.opcode != "mul"
        or product.attrs
        or len(product.operands) != 2
        or len(product.results) != 1
        or not _used_only_by(product.results[0], reduction, 0)
    ):
        return None

    first = product.operands[0].producer
    second = product.operands[1].producer
    if first is None or second is None or first is second:
        return None
    if first.opcode != "reshape" or second.opcode != "reshape":
        return None
    if first.attrs or second.attrs or len(first.operands) != 1 or len(second.operands) != 1:
        return None

    for lhs_reshape, rhs_reshape, lhs_position, rhs_position in (
        (first, second, 0, 1),
        (second, first, 1, 0),
    ):
        pattern = _match_orientation(
            reduction,
            product,
            lhs_reshape,
            rhs_reshape,
            lhs_position,
            rhs_position,
        )
        if pattern is not None:
            return pattern
    return None


def _match_orientation(
    reduction: Operation,
    product: Operation,
    lhs_reshape: Operation,
    rhs_reshape: Operation,
    lhs_position: int,
    rhs_position: int,
) -> DirectMatmulPattern | None:
    lhs = lhs_reshape.operands[0]
    rhs = rhs_reshape.operands[0]
    try:
        result_type = infer_direct_matmul_type(lhs.type, rhs.type)
    except TypeInferenceError:
        return None

    m, k = lhs.type.shape
    _, n = rhs.type.shape
    if lhs_reshape.results[0].type != TensorType((m, k, 1), lhs.type.dtype):
        return None
    if rhs_reshape.results[0].type != TensorType((1, k, n), rhs.type.dtype):
        return None
    if not _used_only_by(lhs_reshape.results[0], product, lhs_position):
        return None
    if not _used_only_by(rhs_reshape.results[0], product, rhs_position):
        return None

    expected_product = infer_binary(lhs_reshape.results[0].type, rhs_reshape.results[0].type)
    if product.results[0].type != expected_product or expected_product.shape != (m, k, n):
        return None
    if reduction.results[0].type != result_type:
        return None
    return DirectMatmulPattern(
        reduction=reduction,
        lhs=lhs,
        rhs=rhs,
        intermediates=(lhs_reshape, rhs_reshape, product),
    )


def _used_only_by(value: Value, user: Operation, operand_index: int) -> bool:
    return (
        len(value.uses) == 1
        and value.uses[0].user is user
        and value.uses[0].operand_index == operand_index
    )
