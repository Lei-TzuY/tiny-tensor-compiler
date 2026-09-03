from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .inference import TypeInferenceError, infer_binary, infer_relu
from .ir import DType, Function, Module, TensorType, Value


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


class GraphBuilder:
    def __init__(self, name: str = "main") -> None:
        self.function = Function(name)
        self._finished = False
        self._next_input_index = 0

    def input(
        self,
        shape: Iterable[int],
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

    def finish(self, result: Tensor) -> Module:
        self._ensure_open()
        self._check_tensor_owner(result)
        self.function.add_op("return", operands=[result.value])
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
