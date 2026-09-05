from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from .inference import (
    TypeInferenceError,
    infer_binary,
    infer_prod,
    infer_relu,
    infer_reshape,
    infer_reverse,
    infer_slice,
    infer_sum,
    infer_transpose,
    normalize_prod_axis,
    normalize_sum_axis,
)
from .ir import DType, Function, Module, ShapeDim, TensorType, Value

_ALIAS_OPCODES = frozenset({"view", "slice", "reverse", "transpose"})


class Tensor:
    def __init__(self, builder: GraphBuilder, value: Value) -> None:
        self._builder = builder
        self.value = value

    @property
    def type(self) -> TensorType:
        return self.value.type

    def __add__(self, other: Any) -> Tensor:
        return self._builder.binary("add", self, other)

    def __radd__(self, other: Any) -> Tensor:
        return self._builder.binary("add", other, self)

    def __mul__(self, other: Any) -> Tensor:
        return self._builder.binary("mul", self, other)

    def __rmul__(self, other: Any) -> Tensor:
        return self._builder.binary("mul", other, self)

    def relu(self) -> Tensor:
        return self._builder.relu(self)

    def sum(self, axis: int | None = None) -> Tensor:
        return self._builder.sum(self, axis=axis)

    def prod(self, axis: int | None = None) -> Tensor:
        return self._builder.prod(self, axis=axis)

    def reshape(self, shape: Iterable[ShapeDim]) -> Tensor:
        return self._builder.reshape(self, shape)

    def view(self, shape: Iterable[ShapeDim]) -> Tensor:
        return self._builder.view(self, shape)

    def slice(
        self,
        *,
        axis: int,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
    ) -> Tensor:
        return self._builder.slice(self, axis=axis, start=start, stop=stop, step=step)

    def reverse(self, axis: int) -> Tensor:
        return self._builder.reverse(self, axis)

    def transpose(self, axes: Iterable[int] | None = None) -> Tensor:
        return self._builder.transpose(self, axes)

    def copy_into(self, target: Tensor, source: Tensor) -> Tensor:
        """Copy ``source`` into one alias region of this fresh internal root generation."""
        return self._builder.copy_into(self, target, source)


class GraphBuilder:
    def __init__(self, name: str = "main") -> None:
        self.function = Function(name)
        self._finished = False
        self._next_input_index = 0

    def input(
        self,
        shape: Iterable[ShapeDim],
        dtype: str | np.dtype[Any] | DType,
    ) -> Tensor:
        self._ensure_open()
        try:
            tensor_dtype = dtype if isinstance(dtype, DType) else DType.from_numpy(np.dtype(dtype))
            type_ = TensorType(tuple(shape), tensor_dtype)
        except (TypeError, ValueError) as exc:
            raise TypeInferenceError(str(exc)) from exc

        index = self._next_input_index
        self._next_input_index += 1
        op = self.function.add_op(
            "input",
            result_types=[type_],
            attrs={"index": index},
        )
        return Tensor(self, op.results[0])

    def tensor(self, data: Any, dtype: str | np.dtype[Any] | DType | None = None) -> Tensor:
        self._ensure_open()
        np_dtype = dtype.to_numpy() if isinstance(dtype, DType) else dtype
        try:
            array = np.asarray(data, dtype=np_dtype)
            tensor_dtype = DType.from_numpy(array.dtype)
        except (TypeError, ValueError) as exc:
            raise TypeInferenceError(str(exc)) from exc
        array = np.array(array, copy=True)
        type_ = TensorType(tuple(array.shape), tensor_dtype)
        op = self.function.add_op("const", result_types=[type_], attrs={"value": array})
        return Tensor(self, op.results[0])

    def binary(self, opcode: str, lhs: Any, rhs: Any) -> Tensor:
        self._ensure_open()
        lhs_tensor, rhs_tensor = self._coerce_binary_operands(lhs, rhs)
        result_type = infer_binary(lhs_tensor.type, rhs_tensor.type)
        op = self.function.add_op(
            opcode,
            operands=[lhs_tensor.value, rhs_tensor.value],
            result_types=[result_type],
        )
        return Tensor(self, op.results[0])

    def relu(self, tensor: Tensor) -> Tensor:
        self._ensure_open()
        self._check_tensor_owner(tensor)
        result_type = infer_relu(tensor.type)
        op = self.function.add_op(
            "relu", operands=[tensor.value], result_types=[result_type]
        )
        return Tensor(self, op.results[0])

    def sum(self, tensor: Tensor, axis: int | None = None) -> Tensor:
        self._ensure_open()
        self._check_tensor_owner(tensor)
        normalized_axis = None if axis is None else normalize_sum_axis(tensor.type, axis)
        result_type = infer_sum(tensor.type, normalized_axis)
        attrs = {} if normalized_axis is None else {"axis": normalized_axis}
        op = self.function.add_op(
            "sum",
            operands=[tensor.value],
            result_types=[result_type],
            attrs=attrs,
        )
        return Tensor(self, op.results[0])

    def prod(self, tensor: Tensor, axis: int | None = None) -> Tensor:
        self._ensure_open()
        self._check_tensor_owner(tensor)
        normalized_axis = None if axis is None else normalize_prod_axis(tensor.type, axis)
        result_type = infer_prod(tensor.type, normalized_axis)
        attrs = {} if normalized_axis is None else {"axis": normalized_axis}
        op = self.function.add_op(
            "prod",
            operands=[tensor.value],
            result_types=[result_type],
            attrs=attrs,
        )
        return Tensor(self, op.results[0])

    def reshape(self, tensor: Tensor, shape: Iterable[ShapeDim]) -> Tensor:
        self._ensure_open()
        self._check_tensor_owner(tensor)
        result_type = infer_reshape(tensor.type, shape)
        op = self.function.add_op(
            "reshape",
            operands=[tensor.value],
            result_types=[result_type],
        )
        return Tensor(self, op.results[0])

    def view(self, tensor: Tensor, shape: Iterable[ShapeDim]) -> Tensor:
        self._ensure_open()
        self._check_tensor_owner(tensor)
        result_type = infer_reshape(tensor.type, shape)
        op = self.function.add_op(
            "view",
            operands=[tensor.value],
            result_types=[result_type],
        )
        return Tensor(self, op.results[0])

    def slice(
        self,
        tensor: Tensor,
        *,
        axis: int,
        start: int = 0,
        stop: int | None = None,
        step: int = 1,
    ) -> Tensor:
        self._ensure_open()
        self._check_tensor_owner(tensor)
        if (
            not isinstance(axis, int)
            or isinstance(axis, bool)
            or axis < 0
            or axis >= len(tensor.type.shape)
        ):
            raise TypeInferenceError("slice axis is out of range")
        extent = tensor.type.shape[axis]
        if not isinstance(extent, int) or isinstance(extent, bool):
            raise TypeInferenceError("slice axis extent must be concrete before slicing")
        normalized_stop = extent if stop is None else stop
        result_type = infer_slice(
            tensor.type,
            axis=axis,
            start=start,
            stop=normalized_stop,
            step=step,
        )
        op = self.function.add_op(
            "slice",
            operands=[tensor.value],
            result_types=[result_type],
            attrs={
                "axis": axis,
                "start": start,
                "stop": normalized_stop,
                "step": step,
            },
        )
        return Tensor(self, op.results[0])

    def reverse(self, tensor: Tensor, axis: int) -> Tensor:
        self._ensure_open()
        self._check_tensor_owner(tensor)
        result_type = infer_reverse(tensor.type, axis)
        op = self.function.add_op(
            "reverse",
            operands=[tensor.value],
            result_types=[result_type],
            attrs={"axis": axis},
        )
        return Tensor(self, op.results[0])

    def transpose(self, tensor: Tensor, axes: Iterable[int] | None = None) -> Tensor:
        self._ensure_open()
        self._check_tensor_owner(tensor)
        permutation = (
            tuple(reversed(range(len(tensor.type.shape)))) if axes is None else tuple(axes)
        )
        result_type = infer_transpose(tensor.type, permutation)
        op = self.function.add_op(
            "transpose",
            operands=[tensor.value],
            result_types=[result_type],
            attrs={"axes": permutation},
        )
        return Tensor(self, op.results[0])

    def copy_into(self, root: Tensor, target: Tensor, source: Tensor) -> Tensor:
        self._ensure_open()
        for tensor in (root, target, source):
            self._check_tensor_owner(tensor)

        owner = _storage_root(root.value)
        if not _is_full_root_handle(root.value):
            raise ValueError("copy_into root must be an owning tensor or fresh full-root result")
        producer = owner.producer
        if producer is None or producer.opcode in {"input", "const"}:
            raise ValueError("copy_into root must use internal computed storage")
        if _storage_root(target.value) is not owner:
            raise ValueError("copy_into target must alias the supplied root storage")
        if target.type != source.type:
            raise ValueError("copy_into target and source types must exactly match")
        if _storage_root(source.value) is owner:
            # Same-root writes have explicit snapshot semantics at the public builder
            # boundary. Materialize the logical source in C order before mutating the
            # owning root, so overlapping, interleaved, reversed, transposed, and
            # unresolved-symbolic layouts all reduce to the existing different-root
            # copy_into contract. The lower verifier/backend invariant therefore stays
            # fail-closed rather than acquiring hidden memmove behavior.
            source = self.reshape(source, source.type.shape)

        op = self.function.add_op(
            "copy_into",
            operands=[root.value, target.value, source.value],
            result_types=[root.type],
        )
        return Tensor(self, op.results[0])

    def finish(self, result: Tensor | Sequence[Tensor]) -> Module:
        self._ensure_open()
        if isinstance(result, Tensor):
            results = (result,)
        else:
            try:
                results = tuple(result)
            except TypeError as exc:
                raise TypeError("graph result must be a Tensor or a sequence of Tensors") from exc
            if not results:
                raise ValueError("graph must return at least one tensor")

        for tensor in results:
            if not isinstance(tensor, Tensor):
                raise TypeError("graph result sequence must contain only Tensor values")
            self._check_tensor_owner(tensor)

        self.function.add_op("return", operands=[tensor.value for tensor in results])
        self._finished = True
        return Module(self.function)

    def _coerce_binary_operands(self, lhs: Any, rhs: Any) -> tuple[Tensor, Tensor]:
        lhs_tensor = lhs if isinstance(lhs, Tensor) else None
        rhs_tensor = rhs if isinstance(rhs, Tensor) else None
        if lhs_tensor is not None:
            self._check_tensor_owner(lhs_tensor)
        if rhs_tensor is not None:
            self._check_tensor_owner(rhs_tensor)

        # Python scalar literals are coerced to the peer tensor's dtype. This keeps
        # tensor<float32> * 2 as float32 while tensor-vs-tensor promotion remains explicit.
        if lhs_tensor is None:
            peer_dtype = (
                rhs_tensor.type.dtype if rhs_tensor is not None and np.isscalar(lhs) else None
            )
            lhs_tensor = self.tensor(lhs, peer_dtype)
        if rhs_tensor is None:
            peer_dtype = lhs_tensor.type.dtype if np.isscalar(rhs) else None
            rhs_tensor = self.tensor(rhs, peer_dtype)
        return lhs_tensor, rhs_tensor

    def _check_tensor_owner(self, tensor: Tensor) -> None:
        if tensor._builder is not self:
            raise ValueError("cannot combine tensors from different GraphBuilder instances")

    def _ensure_open(self) -> None:
        if self._finished:
            raise RuntimeError("graph has already been finished")


def _is_full_root_handle(value: Value) -> bool:
    owner = _storage_root(value)
    if value is owner:
        return True
    producer = value.producer
    return (
        producer is not None
        and producer.opcode == "copy_into"
        and producer.results[0] is value
        and value.type == owner.type
    )


def _storage_root(value: Value) -> Value:
    current = value
    seen: set[Value] = set()
    while True:
        if current in seen:
            raise ValueError("tensor alias cycle detected")
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