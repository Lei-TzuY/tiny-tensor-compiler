from __future__ import annotations

from collections import Counter

import numpy as np

from .inference import (
    TypeInferenceError,
    infer_binary,
    infer_relu,
    infer_reshape,
    infer_reverse,
    infer_slice,
    infer_transpose,
)
from .ir import DType, Module, Operation, TensorType, Value


class VerificationError(ValueError):
    pass


_ALIAS_OPCODES = frozenset({"view", "slice", "reverse", "transpose"})


def verify(module: Module) -> None:
    function = module.function
    seen: set[Value] = set()
    all_results: list[Value] = []
    expected_uses: Counter[tuple[Value, Operation, int]] = Counter()
    returns = 0
    copy_into_count = 0
    next_input_index = 0

    for op_index, op in enumerate(function.ops):
        if op.parent is not function:
            _fail(op_index, op, "operation parent does not match containing function")
        for operand_index, operand in enumerate(op.operands):
            if operand not in seen:
                _fail(op_index, op, f"operand {operand_index} is not defined before use")
            expected_uses[(operand, op, operand_index)] += 1
        for result_index, result in enumerate(op.results):
            if result.producer is not op or result.result_index != result_index:
                _fail(op_index, op, f"result {result_index} has invalid producer metadata")
            if result in seen:
                _fail(op_index, op, "result value is defined more than once")
            seen.add(result)
            all_results.append(result)

        if op.opcode == "input":
            _verify_input(op_index, op, next_input_index)
            next_input_index += 1
        elif op.opcode == "const":
            _verify_const(op_index, op)
        elif op.opcode in {"add", "mul"}:
            _verify_binary(op_index, op)
        elif op.opcode == "relu":
            _verify_relu(op_index, op)
        elif op.opcode in {"reshape", "view"}:
            _verify_shape_transform(op_index, op)
        elif op.opcode == "slice":
            _verify_slice(op_index, op)
        elif op.opcode == "reverse":
            _verify_reverse(op_index, op)
        elif op.opcode == "transpose":
            _verify_transpose(op_index, op)
        elif op.opcode == "copy_into":
            copy_into_count += 1
            _verify_copy_into(op_index, op, function.ops)
        elif op.opcode == "return":
            returns += 1
            _verify_return(op_index, op)
            if op_index != len(function.ops) - 1:
                _fail(op_index, op, "return must be the final operation")
        else:
            _fail(op_index, op, f"unknown opcode {op.opcode!r}")

    if returns != 1:
        raise VerificationError(
            f"function @{function.name}: expected exactly one return, found {returns}"
        )
    if copy_into_count > 1:
        raise VerificationError(
            f"function @{function.name}: first writable-alias phase permits at most one copy_into"
        )

    actual_uses: Counter[tuple[Value, Operation, int]] = Counter()
    for value in all_results:
        for use in value.uses:
            actual_uses[(value, use.user, use.operand_index)] += 1
    if actual_uses != expected_uses:
        raise VerificationError(f"function @{function.name}: use-def tracking mismatch")


def _verify_input(op_index: int, op: Operation, expected_index: int) -> None:
    _expect_arity(op_index, op, operands=0, results=1)
    if set(op.attrs) != {"index"}:
        _fail(op_index, op, "input requires exactly one 'index' attribute")
    index = op.attrs["index"]
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        _fail(op_index, op, "input index must be a non-negative integer")
    if index != expected_index:
        _fail(
            op_index,
            op,
            f"input index {index} is not the next dense input index {expected_index}",
        )
    if not isinstance(op.results[0].type, TensorType):
        _fail(op_index, op, "input result must have a TensorType")


def _verify_const(op_index: int, op: Operation) -> None:
    _expect_arity(op_index, op, operands=0, results=1)
    if set(op.attrs) != {"value"}:
        _fail(op_index, op, "const requires exactly one 'value' attribute")
    array = np.asarray(op.attrs["value"])
    try:
        expected_type = TensorType(tuple(array.shape), DType.from_numpy(array.dtype))
    except (TypeError, ValueError) as exc:
        _fail(op_index, op, str(exc))
    if op.results[0].type != expected_type:
        _fail(
            op_index,
            op,
            f"const result type {op.results[0].type} does not match literal type {expected_type}",
        )


def _verify_binary(op_index: int, op: Operation) -> None:
    _expect_arity(op_index, op, operands=2, results=1)
    if op.attrs:
        _fail(op_index, op, "binary operation does not accept attributes")
    try:
        expected_type = infer_binary(op.operands[0].type, op.operands[1].type)
    except TypeInferenceError as exc:
        _fail(op_index, op, str(exc))
    if op.results[0].type != expected_type:
        _fail(
            op_index,
            op,
            f"result type {op.results[0].type} does not match inferred type {expected_type}",
        )


def _verify_relu(op_index: int, op: Operation) -> None:
    _expect_arity(op_index, op, operands=1, results=1)
    if op.attrs:
        _fail(op_index, op, "relu does not accept attributes")
    try:
        expected_type = infer_relu(op.operands[0].type)
    except TypeInferenceError as exc:
        _fail(op_index, op, str(exc))
    if op.results[0].type != expected_type:
        _fail(
            op_index,
            op,
            f"result type {op.results[0].type} does not match operand type {expected_type}",
        )


def _verify_shape_transform(op_index: int, op: Operation) -> None:
    _expect_arity(op_index, op, operands=1, results=1)
    if op.attrs:
        _fail(op_index, op, f"{op.opcode} does not accept attributes")
    try:
        expected_type = infer_reshape(op.operands[0].type, op.results[0].type.shape)
    except TypeInferenceError as exc:
        _fail(op_index, op, str(exc))
    if op.results[0].type != expected_type:
        _fail(
            op_index,
            op,
            f"{op.opcode} result type {op.results[0].type} does not match inferred type {expected_type}",
        )


def _verify_slice(op_index: int, op: Operation) -> None:
    _expect_arity(op_index, op, operands=1, results=1)
    if set(op.attrs) != {"axis", "start", "stop", "step"}:
        _fail(op_index, op, "slice requires axis/start/stop/step attributes")
    try:
        expected_type = infer_slice(op.operands[0].type, **op.attrs)
    except (TypeError, TypeInferenceError) as exc:
        _fail(op_index, op, str(exc))
    if op.results[0].type != expected_type:
        _fail(
            op_index,
            op,
            f"slice result type {op.results[0].type} does not match inferred type {expected_type}",
        )


def _verify_reverse(op_index: int, op: Operation) -> None:
    _expect_arity(op_index, op, operands=1, results=1)
    if set(op.attrs) != {"axis"}:
        _fail(op_index, op, "reverse requires exactly one 'axis' attribute")
    try:
        expected_type = infer_reverse(op.operands[0].type, op.attrs["axis"])
    except TypeInferenceError as exc:
        _fail(op_index, op, str(exc))
    if op.results[0].type != expected_type:
        _fail(
            op_index,
            op,
            f"reverse result type {op.results[0].type} does not match inferred type {expected_type}",
        )


def _verify_transpose(op_index: int, op: Operation) -> None:
    _expect_arity(op_index, op, operands=1, results=1)
    if set(op.attrs) != {"axes"}:
        _fail(op_index, op, "transpose requires exactly one 'axes' attribute")
    try:
        expected_type = infer_transpose(op.operands[0].type, op.attrs["axes"])
    except (TypeError, TypeInferenceError) as exc:
        _fail(op_index, op, str(exc))
    if op.results[0].type != expected_type:
        _fail(
            op_index,
            op,
            f"transpose result type {op.results[0].type} does not match inferred type {expected_type}",
        )


def _verify_copy_into(
    op_index: int,
    op: Operation,
    operations: list[Operation],
) -> None:
    _expect_arity(op_index, op, operands=3, results=1)
    if op.attrs:
        _fail(op_index, op, "copy_into does not accept attributes")

    root, target, source = op.operands
    result = op.results[0]
    if result.type != root.type:
        _fail(op_index, op, "copy_into result type must match the owning root type")
    if target.type != source.type:
        _fail(op_index, op, "copy_into target and source types must exactly match")
    if _storage_root(root) is not root:
        _fail(op_index, op, "copy_into root must be an owning tensor value")
    producer = root.producer
    if producer is None or producer.opcode in {"input", "const"}:
        _fail(op_index, op, "copy_into root must use internal computed storage")
    if _storage_root(target) is not root:
        _fail(op_index, op, "copy_into target must alias its owning root")
    if _storage_root(source) is root:
        _fail(op_index, op, "copy_into source must use a different storage root")

    if op_index != len(operations) - 2 or operations[-1].opcode != "return":
        _fail(op_index, op, "copy_into must be the final effect before return")
    return_op = operations[-1]
    if result not in return_op.operands:
        _fail(op_index, op, "copy_into fresh result must be returned")
    for operand in return_op.operands:
        if operand is result:
            continue
        if _storage_root(operand) is root:
            _fail(op_index, op, "return contains a stale alias from before copy_into")


def _verify_return(op_index: int, op: Operation) -> None:
    if op.results:
        _fail(op_index, op, "return must not produce results")
    if not op.operands:
        _fail(op_index, op, "return requires at least one operand")
    if op.attrs:
        _fail(op_index, op, "return does not accept attributes")


def _storage_root(value: Value) -> Value:
    current = value
    seen: set[Value] = set()
    while True:
        if current in seen:
            raise VerificationError("tensor alias cycle detected")
        seen.add(current)
        producer = current.producer
        if producer is None:
            return current
        if producer.opcode in _ALIAS_OPCODES:
            current = producer.operands[0]
            continue
        if producer.opcode == "copy_into":
            current = producer.operands[0]
            continue
        return current


def _expect_arity(op_index: int, op: Operation, operands: int, results: int) -> None:
    if len(op.operands) != operands or len(op.results) != results:
        _fail(
            op_index,
            op,
            f"expected {operands} operands/{results} results, got "
            f"{len(op.operands)} operands/{len(op.results)} results",
        )


def _fail(op_index: int, op: Operation, message: str) -> None:
    raise VerificationError(f"op #{op_index} ({op.opcode}): {message}")
