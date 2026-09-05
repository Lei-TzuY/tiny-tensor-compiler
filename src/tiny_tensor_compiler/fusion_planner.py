from __future__ import annotations

from dataclasses import dataclass

from . import fused_expr
from .ir import DType, TensorType
from .loop_ir import (
    IndexMap,
    LoopInput,
    LoopKernel,
    LoopOperation,
    LoopProgram,
    LoopReturn,
    fused_expression_for_kernel,
)

_BINARY_OPCODES = {"add", "mul"}
_MAX_BINARY_NODES = 4


@dataclass(frozen=True)
class _BinaryFusionPlan:
    consumed: int
    kernel: LoopKernel


def fuse_elementwise(program: LoopProgram) -> LoopProgram:
    """Fuse verified elementwise subgraphs through one bounded DAG planner."""
    operations = program.operations
    types = {alloc.buffer: alloc.type for alloc in program.allocations}
    fused: list[LoopOperation] = []
    index = 0

    while index < len(operations):
        plan = _plan_binary_subgraph(operations, index, types)
        if plan is not None:
            kernel, next_index = _absorb_relu_tail(
                operations,
                index + plan.consumed,
                plan.kernel,
            )
            fused.append(kernel)
            index = next_index
            continue

        operation = operations[index]
        if isinstance(operation, LoopKernel):
            kernel, next_index = _absorb_relu_tail(operations, index + 1, operation)
            if next_index != index + 1:
                fused.append(kernel)
                index = next_index
                continue

        fused.append(operation)
        index += 1

    return LoopProgram(tuple(fused))


def _plan_binary_subgraph(
    operations: tuple[LoopOperation, ...],
    start_index: int,
    types: dict[int, TensorType],
) -> _BinaryFusionPlan | None:
    max_length = 0
    for offset in range(_MAX_BINARY_NODES):
        index = start_index + offset
        if index >= len(operations):
            break
        operation = operations[index]
        if not isinstance(operation, LoopKernel) or operation.opcode not in _BINARY_OPCODES:
            break
        max_length += 1

    for length in range(max_length, 1, -1):
        kernel = _plan_binary_window(operations, start_index, length, types)
        if kernel is not None:
            return _BinaryFusionPlan(length, kernel)
    return None


def _plan_binary_window(
    operations: tuple[LoopOperation, ...],
    start_index: int,
    length: int,
    types: dict[int, TensorType],
) -> LoopKernel | None:
    nodes = operations[start_index : start_index + length]
    if len(nodes) != length or not all(isinstance(node, LoopKernel) for node in nodes):
        return None
    kernels = tuple(node for node in nodes if isinstance(node, LoopKernel))
    if any(kernel.opcode not in _BINARY_OPCODES for kernel in kernels):
        return None

    root = kernels[-1]
    if any(kernel.iteration_shape != root.iteration_shape for kernel in kernels):
        return None

    outputs = tuple(kernel.output for kernel in kernels)
    if len(set(outputs)) != len(outputs):
        return None
    identity = IndexMap(tuple(range(len(root.iteration_shape))))

    edges: list[tuple[int | None, int | None]] = []
    internal_uses = [0] * length
    prior_outputs: dict[int, int] = {}
    for node_index, kernel in enumerate(kernels):
        if len(kernel.inputs) != 2 or len(kernel.input_maps) != 2:
            return None
        node_edges: list[int | None] = []
        for buffer, index_map in zip(kernel.inputs, kernel.input_maps, strict=True):
            producer_index = prior_outputs.get(buffer)
            if producer_index is None:
                node_edges.append(None)
                continue
            if index_map != identity:
                return None
            node_edges.append(producer_index)
            internal_uses[producer_index] += 1
        edges.append((node_edges[0], node_edges[1]))
        prior_outputs[kernel.output] = node_index

    if internal_uses[-1] != 0 or any(count != 1 for count in internal_uses[:-1]):
        return None

    after_candidate = start_index + length
    if any(
        not _producer_value_has_no_later_use(
            operations,
            after_candidate,
            kernel.output,
        )
        for kernel in kernels[:-1]
    ):
        return None

    output_type = types[root.output]
    if output_type.dtype not in {DType.INT32, DType.INT64}:
        return None
    if any(types[kernel.output] != output_type for kernel in kernels[:-1]):
        return None

    classified = _classify_supported_topology(kernels, tuple(edges))
    if classified is None:
        return None
    expression, fused_inputs, fused_maps = classified

    if root.output in fused_inputs:
        return None
    if any(types[buffer].dtype != output_type.dtype for buffer in fused_inputs):
        return None

    return LoopKernel(
        opcode=fused_expr.encode_fused_opcode(expression),
        output=root.output,
        inputs=fused_inputs,
        iteration_shape=root.iteration_shape,
        input_maps=fused_maps,
        fused_expression=expression,
    )


def _classify_supported_topology(
    kernels: tuple[LoopKernel, ...],
    edges: tuple[tuple[int | None, int | None], ...],
) -> tuple[
    fused_expr.FusedExpression,
    tuple[int, ...],
    tuple[IndexMap, ...],
] | None:
    if len(kernels) == 2:
        return _classify_binary_chain(kernels, edges)
    if len(kernels) == 3:
        return _classify_binary_tree(kernels, edges)
    if len(kernels) == 4:
        return _classify_chain_tree(kernels, edges)
    return None


def _classify_binary_chain(
    kernels: tuple[LoopKernel, ...],
    edges: tuple[tuple[int | None, int | None], ...],
) -> tuple[
    fused_expr.FusedExpression,
    tuple[int, ...],
    tuple[IndexMap, ...],
] | None:
    inner, outer = kernels
    if edges[0] != (None, None):
        return None
    producer_slots = tuple(slot for slot, child in enumerate(edges[1]) if child == 0)
    if len(producer_slots) != 1:
        return None
    tail_slot = 1 - producer_slots[0]
    if edges[1][tail_slot] is not None:
        return None

    expression = fused_expr.binary_chain_expression(inner.opcode, outer.opcode)
    return (
        expression,
        (*inner.inputs, outer.inputs[tail_slot]),
        (*inner.input_maps, outer.input_maps[tail_slot]),
    )


def _classify_binary_tree(
    kernels: tuple[LoopKernel, ...],
    edges: tuple[tuple[int | None, int | None], ...],
) -> tuple[
    fused_expr.FusedExpression,
    tuple[int, ...],
    tuple[IndexMap, ...],
] | None:
    root = kernels[-1]
    if edges[0] != (None, None) or edges[1] != (None, None):
        return None
    left_index, right_index = edges[2]
    if left_index is None or right_index is None or left_index == right_index:
        return None

    left = kernels[left_index]
    right = kernels[right_index]
    expression = fused_expr.binary_tree_expression(
        left.opcode,
        right.opcode,
        root.opcode,
    )
    return (
        expression,
        (*left.inputs, *right.inputs),
        (*left.input_maps, *right.input_maps),
    )


def _classify_chain_tree(
    kernels: tuple[LoopKernel, ...],
    edges: tuple[tuple[int | None, int | None], ...],
) -> tuple[
    fused_expr.FusedExpression,
    tuple[int, ...],
    tuple[IndexMap, ...],
] | None:
    root = kernels[-1]
    root_children = edges[-1]
    if root_children[0] is None or root_children[1] is None:
        return None
    if root_children[0] == root_children[1]:
        return None

    chain_candidates = [
        child
        for child in root_children
        if sum(edge is not None for edge in edges[child]) == 1
    ]
    simple_candidates = [
        child
        for child in root_children
        if edges[child] == (None, None)
    ]
    if len(chain_candidates) != 1 or len(simple_candidates) != 1:
        return None

    chain_index = chain_candidates[0]
    simple_index = simple_candidates[0]
    chain = kernels[chain_index]
    simple = kernels[simple_index]

    inner_slots = tuple(
        slot for slot, child in enumerate(edges[chain_index]) if child is not None
    )
    if len(inner_slots) != 1:
        return None
    inner_slot = inner_slots[0]
    inner_index = edges[chain_index][inner_slot]
    if inner_index is None or edges[inner_index] != (None, None):
        return None
    if {inner_index, chain_index, simple_index, len(kernels) - 1} != set(
        range(len(kernels))
    ):
        return None

    inner = kernels[inner_index]
    tail_slot = 1 - inner_slot
    if edges[chain_index][tail_slot] is not None:
        return None

    expression = fused_expr.chain_tree_expression(
        inner.opcode,
        chain.opcode,
        simple.opcode,
        root.opcode,
    )
    return (
        expression,
        (*inner.inputs, chain.inputs[tail_slot], *simple.inputs),
        (*inner.input_maps, chain.input_maps[tail_slot], *simple.input_maps),
    )


def _absorb_relu_tail(
    operations: tuple[LoopOperation, ...],
    start_index: int,
    producer: LoopKernel,
) -> tuple[LoopKernel, int]:
    next_index = start_index
    current = producer
    while True:
        fused = _fuse_trailing_relu(operations, next_index, current)
        if fused is None:
            return current, next_index
        current = fused
        next_index += 1


def _fuse_trailing_relu(
    operations: tuple[LoopOperation, ...],
    consumer_index: int,
    producer: LoopKernel,
) -> LoopKernel | None:
    if consumer_index >= len(operations):
        return None
    consumer = operations[consumer_index]
    if not isinstance(consumer, LoopKernel):
        return None
    if consumer.opcode != "relu" or consumer.inputs != (producer.output,):
        return None
    if producer.iteration_shape != consumer.iteration_shape:
        return None
    identity = IndexMap(tuple(range(len(consumer.iteration_shape))))
    if consumer.input_maps != (identity,):
        return None
    if consumer.output in producer.inputs:
        return None
    if not _producer_value_has_no_later_use(
        operations,
        consumer_index + 1,
        producer.output,
    ):
        return None

    expression = fused_expression_for_kernel(producer)
    if expression is not None:
        try:
            expression = fused_expr.with_terminal_relu(expression)
        except ValueError:
            return None
        return LoopKernel(
            opcode=fused_expr.encode_fused_opcode(expression),
            output=consumer.output,
            inputs=producer.inputs,
            iteration_shape=consumer.iteration_shape,
            input_maps=producer.input_maps,
            fused_expression=expression,
        )

    if producer.opcode in {"add", "mul"}:
        opcode = f"relu_{producer.opcode}"
    elif producer.opcode in {"relu", "relu_add", "relu_mul"}:
        opcode = producer.opcode
    else:
        return None

    return LoopKernel(
        opcode=opcode,
        output=consumer.output,
        inputs=producer.inputs,
        iteration_shape=consumer.iteration_shape,
        input_maps=producer.input_maps,
        fused_expression=producer.fused_expression,
    )


def _producer_value_has_no_later_use(
    operations: tuple[LoopOperation, ...],
    start_index: int,
    buffer: int,
) -> bool:
    for operation in operations[start_index:]:
        if isinstance(operation, LoopInput):
            if operation.output == buffer:
                return True
        elif isinstance(operation, LoopKernel):
            if operation.output == buffer:
                return True
            if buffer in operation.inputs:
                return False
        elif isinstance(operation, LoopReturn) and operation.buffer == buffer:
            return False
    return True
