from pathlib import Path


def replace(path: str, old: str, new: str, *, count: int = 1) -> None:
    file = Path(path)
    text = file.read_text()
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(f"{path}: expected {count} match(es), found {actual}: {old[:80]!r}")
    file.write_text(text.replace(old, new, count))


# frontend.py
path = "src/tiny_tensor_compiler/frontend.py"
replace(
    path,
    '''    def copy_into(self, target: Tensor, source: Tensor) -> Tensor:\n        """Copy ``source`` into one alias region of this fresh internal root generation."""\n        return self._builder.copy_into(self, target, source)\n''',
    '''    def copy_into(self, target: Tensor, source: Tensor) -> Tensor:\n        """Copy ``source`` into one alias region of this fresh internal root generation."""\n        return self._builder.copy_into(self, target, source)\n\n    def binary_inplace(self, source: Tensor, *, operator: str) -> Tensor:\n        """Apply one exact-typed binary update to this fresh internal storage root."""\n        return self._builder.binary_inplace(self, source, operator=operator)\n\n    def add_inplace(self, source: Tensor) -> Tensor:\n        return self.binary_inplace(source, operator="add")\n\n    def mul_inplace(self, source: Tensor) -> Tensor:\n        return self.binary_inplace(source, operator="mul")\n''',
)
replace(
    path,
    '''        op = self.function.add_op(\n            "copy_into",\n            operands=[root.value, target.value, source.value],\n            result_types=[root.type],\n        )\n        return Tensor(self, op.results[0])\n\n    def finish''',
    '''        op = self.function.add_op(\n            "copy_into",\n            operands=[root.value, target.value, source.value],\n            result_types=[root.type],\n        )\n        return Tensor(self, op.results[0])\n\n    def binary_inplace(self, root: Tensor, source: Tensor, *, operator: str) -> Tensor:\n        self._ensure_open()\n        for tensor in (root, source):\n            self._check_tensor_owner(tensor)\n        if not isinstance(operator, str) or operator not in {"add", "mul"}:\n            raise ValueError("binary_inplace operator must be 'add' or 'mul'")\n\n        owner = _storage_root(root.value)\n        if not _is_full_root_handle(root.value):\n            raise ValueError("binary_inplace root must be an owning tensor or fresh full-root result")\n        producer = owner.producer\n        if producer is None or producer.opcode in {"input", "const"}:\n            raise ValueError("binary_inplace root must use internal computed storage")\n        if _storage_root(source.value) is owner:\n            raise ValueError("binary_inplace source must use a different storage root")\n        if root.type != source.type:\n            raise ValueError("binary_inplace root and source types must exactly match")\n\n        op = self.function.add_op(\n            "binary_inplace",\n            operands=[root.value, source.value],\n            result_types=[root.type],\n            attrs={"operator": operator},\n        )\n        return Tensor(self, op.results[0])\n\n    def finish''',
)
replace(path, 'and producer.opcode == "copy_into"', 'and producer.opcode in {"copy_into", "binary_inplace"}')
replace(
    path,
    '''        if producer.opcode == "copy_into":\n            current = producer.operands[0]\n            continue\n''',
    '''        if producer.opcode in {"copy_into", "binary_inplace"}:\n            current = producer.operands[0]\n            continue\n''',
)

# verifier.py
path = "src/tiny_tensor_compiler/verifier.py"
replace(
    path,
    '''        elif op.opcode == "copy_into":\n            _verify_copy_into(op_index, op)\n        elif op.opcode == "return":''',
    '''        elif op.opcode == "copy_into":\n            _verify_copy_into(op_index, op)\n        elif op.opcode == "binary_inplace":\n            _verify_binary_inplace(op_index, op)\n        elif op.opcode == "return":''',
)
replace(
    path,
    '''def _verify_return(op_index: int, op: Operation) -> None:\n''',
    '''def _verify_binary_inplace(op_index: int, op: Operation) -> None:\n    _expect_arity(op_index, op, operands=2, results=1)\n    if set(op.attrs) != {"operator"}:\n        _fail(op_index, op, "binary_inplace requires exactly one 'operator' attribute")\n    operator = op.attrs["operator"]\n    if not isinstance(operator, str) or operator not in {"add", "mul"}:\n        _fail(op_index, op, "binary_inplace operator must be 'add' or 'mul'")\n\n    root, source = op.operands\n    result = op.results[0]\n    owner = _storage_root(root)\n    if not _is_full_root_handle(root):\n        _fail(op_index, op, "binary_inplace root must be an owning value or fresh full-root result")\n    producer = owner.producer\n    if producer is None or producer.opcode in {"input", "const"}:\n        _fail(op_index, op, "binary_inplace root must use internal computed storage")\n    if _storage_root(source) is owner:\n        _fail(op_index, op, "binary_inplace source must use a different storage root")\n    if root.type != source.type or result.type != root.type:\n        _fail(op_index, op, "binary_inplace root, source, and result types must exactly match")\n\n\ndef _verify_return(op_index: int, op: Operation) -> None:\n''',
)
replace(path, '    if op.opcode == "copy_into":\n', '    if op.opcode in {"copy_into", "binary_inplace"}:\n', count=1)
replace(path, 'and producer.opcode == "copy_into"', 'and producer.opcode in {"copy_into", "binary_inplace"}')
replace(
    path,
    '''        if producer.opcode == "copy_into":\n            current = producer.operands[0]\n            continue\n''',
    '''        if producer.opcode in {"copy_into", "binary_inplace"}:\n            current = producer.operands[0]\n            continue\n''',
)

# runtime.py
path = "src/tiny_tensor_compiler/runtime.py"
replace(
    path,
    '''        elif op.opcode == "copy_into":\n            root = values[op.operands[0]]\n            target = values[op.operands[1]]\n            source = values[op.operands[2]]\n            np.copyto(target, source)\n            values[op.results[0]] = root\n        elif op.opcode == "return":''',
    '''        elif op.opcode == "copy_into":\n            root = values[op.operands[0]]\n            target = values[op.operands[1]]\n            source = values[op.operands[2]]\n            np.copyto(target, source)\n            values[op.results[0]] = root\n        elif op.opcode == "binary_inplace":\n            root = values[op.operands[0]]\n            source = values[op.operands[1]]\n            binary = np.add if op.attrs["operator"] == "add" else np.multiply\n            binary(root, source, out=root)\n            values[op.results[0]] = root\n        elif op.opcode == "return":''',
)

# lowering.py
path = "src/tiny_tensor_compiler/lowering.py"
replace(
    path,
    '''@dataclass(frozen=True)\nclass BufferKernel:''',
    '''@dataclass(frozen=True)\nclass BufferInplaceBinary:\n    output: int\n    root: int\n    source: int\n    operator: str\n\n\n@dataclass(frozen=True)\nclass BufferKernel:''',
)
replace(
    path,
    'BufferOperation = BufferAlloc | BufferInput | BufferView | BufferCopyInto | BufferKernel | BufferReturn',
    'BufferOperation = (\n    BufferAlloc | BufferInput | BufferView | BufferCopyInto | BufferInplaceBinary | BufferKernel | BufferReturn\n)',
)
replace(
    path,
    '''    @property\n    def instructions(self) -> tuple[BufferKernel, ...]:''',
    '''    @property\n    def inplace_binaries(self) -> tuple[BufferInplaceBinary, ...]:\n        return tuple(op for op in self.operations if isinstance(op, BufferInplaceBinary))\n\n    @property\n    def instructions(self) -> tuple[BufferKernel, ...]:''',
)
replace(
    path,
    '''            elif isinstance(op, BufferKernel):\n                if op.opcode == "const":''',
    '''            elif isinstance(op, BufferInplaceBinary):\n                lines.append(\n                    f"b{op.output} = binary_inplace[{op.operator}] root=b{op.root} source=b{op.source}"\n                )\n            elif isinstance(op, BufferKernel):\n                if op.opcode == "const":''',
)
replace(
    path,
    '''        if op.opcode == "copy_into":\n            operations.append(\n                BufferCopyInto(\n                    output=buffer,\n                    root=buffers[op.operands[0]],\n                    target=buffers[op.operands[1]],\n                    source=buffers[op.operands[2]],\n                )\n            )\n            continue\n\n        literal = None''',
    '''        if op.opcode == "copy_into":\n            operations.append(\n                BufferCopyInto(\n                    output=buffer,\n                    root=buffers[op.operands[0]],\n                    target=buffers[op.operands[1]],\n                    source=buffers[op.operands[2]],\n                )\n            )\n            continue\n        if op.opcode == "binary_inplace":\n            operations.append(\n                BufferInplaceBinary(\n                    output=buffer,\n                    root=buffers[op.operands[0]],\n                    source=buffers[op.operands[1]],\n                    operator=op.attrs["operator"],\n                )\n            )\n            continue\n\n        literal = None''',
)
replace(
    path,
    '''        elif isinstance(op, BufferCopyInto):\n            alias_sources[op.output] = op.root\n            for buffer in (op.output, op.root, op.target, op.source):\n                last_uses[buffer] = max(last_uses.get(buffer, -1), index)\n        elif isinstance(op, BufferKernel):''',
    '''        elif isinstance(op, BufferCopyInto):\n            alias_sources[op.output] = op.root\n            for buffer in (op.output, op.root, op.target, op.source):\n                last_uses[buffer] = max(last_uses.get(buffer, -1), index)\n        elif isinstance(op, BufferInplaceBinary):\n            alias_sources[op.output] = op.root\n            for buffer in (op.output, op.root, op.source):\n                last_uses[buffer] = max(last_uses.get(buffer, -1), index)\n        elif isinstance(op, BufferKernel):''',
)
replace(
    path,
    '''        elif isinstance(op, BufferCopyInto):\n            layouts[op.output] = layouts[op.root]\n''',
    '''        elif isinstance(op, (BufferCopyInto, BufferInplaceBinary)):\n            layouts[op.output] = layouts[op.root]\n''',
)
replace(
    path,
    '''        elif isinstance(op, BufferCopyInto):\n            source = op.root\n            output = op.output\n        else:\n            continue''',
    '''        elif isinstance(op, (BufferCopyInto, BufferInplaceBinary)):\n            source = op.root\n            output = op.output\n        else:\n            continue''',
)
replace(
    path,
    '''        if isinstance(op, BufferKernel):\n            if op.output not in allocated:''',
    '''        if isinstance(op, BufferInplaceBinary):\n            for buffer in (op.output, op.root, op.source):\n                if buffer not in allocated:\n                    raise ValueError("binary_inplace requires allocated logical buffer values")\n            if op.output in written:\n                raise ValueError(f"buffer b{op.output} is written more than once")\n            for buffer in (op.root, op.source):\n                if buffer not in written:\n                    raise ValueError(f"binary_inplace reads b{buffer} before it is written")\n                require_fresh(buffer)\n            owner = roots[op.root]\n            if op.root not in full_root_handles:\n                raise ValueError("binary_inplace root must be a fresh full-root buffer handle")\n            if owner in input_roots:\n                raise ValueError("binary_inplace root must use internal computed storage")\n            if roots[op.source] == owner:\n                raise ValueError("binary_inplace source must use a different storage root")\n            if op.operator not in {"add", "mul"}:\n                raise ValueError("binary_inplace operator must be add or mul")\n            if allocated[op.root] != allocated[owner]:\n                raise ValueError("binary_inplace root handle type must match owning storage")\n            if allocated[op.root] != allocated[op.source] or allocated[op.output] != allocated[op.root]:\n                raise ValueError("binary_inplace root, source, and result types must exactly match")\n            alias_sources[op.output] = op.root\n            root_generations[owner] += 1\n            roots[op.output] = owner\n            value_generations[op.output] = root_generations[owner]\n            full_root_handles.add(op.output)\n            written.add(op.output)\n            continue\n\n        if isinstance(op, BufferKernel):\n            if op.output not in allocated:''',
)

# loop_ir.py
path = "src/tiny_tensor_compiler/loop_ir.py"
replace(path, '    BufferInput,\n', '    BufferInput,\n    BufferInplaceBinary,\n')
replace(
    path,
    '''@dataclass(frozen=True)\nclass LoopKernel:''',
    '''@dataclass(frozen=True)\nclass LoopInplaceBinary:\n    output: int\n    root: int\n    source: int\n    operator: str\n    type: TensorType\n\n\n@dataclass(frozen=True)\nclass LoopKernel:''',
)
replace(
    path,
    'LoopOperation = LoopAlloc | LoopInput | LoopView | LoopCopyInto | LoopKernel | LoopReturn',
    'LoopOperation = (\n    LoopAlloc | LoopInput | LoopView | LoopCopyInto | LoopInplaceBinary | LoopKernel | LoopReturn\n)',
)
replace(
    path,
    '''                if isinstance(op, (LoopView, LoopCopyInto))\n''',
    '''                if isinstance(op, (LoopView, LoopCopyInto, LoopInplaceBinary))\n''',
)
replace(
    path,
    '''            elif isinstance(op, LoopCopyInto):\n                if op.root not in roots:\n                    raise ValueError("copy_into root handle has no storage root")\n                root = roots[op.root]\n                if types[op.root] != root_types[root] or layouts[op.root] != layouts[root]:\n                    raise ValueError("copy_into root handle must expose the full owning root")\n                op.layout.validate_bounds(op.type.shape, element_count(root_types[root].shape))\n                layouts[op.output] = op.layout\n                roots[op.output] = root\n''',
    '''            elif isinstance(op, LoopCopyInto):\n                if op.root not in roots:\n                    raise ValueError("copy_into root handle has no storage root")\n                root = roots[op.root]\n                if types[op.root] != root_types[root] or layouts[op.root] != layouts[root]:\n                    raise ValueError("copy_into root handle must expose the full owning root")\n                op.layout.validate_bounds(op.type.shape, element_count(root_types[root].shape))\n                layouts[op.output] = op.layout\n                roots[op.output] = root\n            elif isinstance(op, LoopInplaceBinary):\n                if op.root not in roots:\n                    raise ValueError("binary_inplace root handle has no storage root")\n                root = roots[op.root]\n                if types[op.root] != root_types[root] or layouts[op.root] != layouts[root]:\n                    raise ValueError("binary_inplace root handle must expose the full owning root")\n                layouts[op.output] = layouts[root]\n                roots[op.output] = root\n''',
)
replace(
    path,
    '''            elif isinstance(op, LoopCopyInto):\n                if op.root not in roots:\n                    raise KeyError(f"copy_into root handle p{op.root} has no storage root")\n                roots[op.output] = roots[op.root]\n''',
    '''            elif isinstance(op, LoopCopyInto):\n                if op.root not in roots:\n                    raise KeyError(f"copy_into root handle p{op.root} has no storage root")\n                roots[op.output] = roots[op.root]\n            elif isinstance(op, LoopInplaceBinary):\n                if op.root not in roots:\n                    raise KeyError(f"binary_inplace root handle p{op.root} has no storage root")\n                roots[op.output] = roots[op.root]\n''',
)
replace(
    path,
    '''            if isinstance(op, LoopReturn):\n                lines.append(f"return p{op.buffer}")\n                continue\n''',
    '''            if isinstance(op, LoopInplaceBinary):\n                lines.append(\n                    f"p{op.output} = binary_inplace[{op.operator}] root=p{op.root} source=p{op.source} : {op.type}"\n                )\n                continue\n            if isinstance(op, LoopReturn):\n                lines.append(f"return p{op.buffer}")\n                continue\n''',
)
replace(
    path,
    '''    @property\n    def kernels(self) -> tuple[LoopKernel, ...]:''',
    '''    @property\n    def inplace_binaries(self) -> tuple[LoopInplaceBinary, ...]:\n        return tuple(op for op in self.operations if isinstance(op, LoopInplaceBinary))\n\n    @property\n    def kernels(self) -> tuple[LoopKernel, ...]:''',
)
replace(
    path,
    '''        if isinstance(op, BufferReturn):\n            operations.append(LoopReturn(virtual_handles[op.buffer]))\n            continue\n\n        output_type = virtual_types[op.output]''',
    '''        if isinstance(op, BufferInplaceBinary):\n            handle = next_handle\n            next_handle += 1\n            virtual_handles[op.output] = handle\n            operations.append(\n                LoopInplaceBinary(\n                    output=handle,\n                    root=virtual_handles[op.root],\n                    source=virtual_handles[op.source],\n                    operator=op.operator,\n                    type=virtual_types[op.output],\n                )\n            )\n            continue\n        if isinstance(op, BufferReturn):\n            operations.append(LoopReturn(virtual_handles[op.buffer]))\n            continue\n\n        output_type = virtual_types[op.output]''',
)
replace(
    path,
    '''        if isinstance(op, LoopKernel):\n            saw_execution = True\n''',
    '''        if isinstance(op, LoopInplaceBinary):\n            saw_execution = True\n            if op.output < 0:\n                raise ValueError(f"invalid negative binary_inplace result id p{op.output}")\n            if op.output in types:\n                raise ValueError(f"binary_inplace result p{op.output} collides with an existing loop value")\n            for buffer in (op.root, op.source):\n                if buffer not in types:\n                    raise ValueError(f"binary_inplace input p{buffer} is not defined")\n                if buffer not in written:\n                    raise ValueError(f"binary_inplace input p{buffer} is not written")\n                _verify_fresh_value(buffer, roots, root_generations, value_generations)\n            root = roots[op.root]\n            if root not in allocated:\n                raise ValueError("binary_inplace root handle has no owning storage")\n            if root in input_roots:\n                raise ValueError("binary_inplace cannot mutate borrowed or copied runtime input storage")\n            if types[op.root] != allocated[root] or layouts[op.root] != layouts[root]:\n                raise ValueError("binary_inplace root must be a fresh full-root handle")\n            if roots[op.source] == root:\n                raise ValueError("binary_inplace source must use a different storage root")\n            if op.operator not in {"add", "mul"}:\n                raise ValueError("binary_inplace operator must be add or mul")\n            if types[op.root] != types[op.source] or op.type != types[op.root]:\n                raise ValueError("binary_inplace root, source, and result types must exactly match")\n\n            root_generations[root] += 1\n            types[op.output] = op.type\n            layouts[op.output] = layouts[root]\n            roots[op.output] = root\n            value_generations[op.output] = root_generations[root]\n            written.add(op.output)\n            continue\n\n        if isinstance(op, LoopKernel):\n            saw_execution = True\n''',
)

# CPU loop backend
path = "src/tiny_tensor_compiler/backends/cpu.py"
replace(path, '    LoopInput,\n', '    LoopInput,\n    LoopInplaceBinary,\n')
replace(
    path,
    '''        if isinstance(op, LoopCopyInto):\n            np.copyto(buffers[op.target], buffers[op.source])\n            buffers[op.output] = buffers[op.root]\n            continue\n''',
    '''        if isinstance(op, LoopCopyInto):\n            np.copyto(buffers[op.target], buffers[op.source])\n            buffers[op.output] = buffers[op.root]\n            continue\n\n        if isinstance(op, LoopInplaceBinary):\n            binary = np.add if op.operator == "add" else np.multiply\n            binary(buffers[op.root], buffers[op.source], out=buffers[op.root])\n            buffers[op.output] = buffers[op.root]\n            continue\n''',
)

# generated C write emitter
path = "src/tiny_tensor_compiler/write_codegen.py"
replace(path, 'from .loop_ir import LoopCopyInto', 'from .loop_ir import LoopCopyInto, LoopInplaceBinary')
replace(
    path,
    '''def _root_ref(root: int, base_offset: int, strides: tuple[int, ...]) -> str:\n''',
    '''def emit_inplace_binary(\n    op: LoopInplaceBinary,\n    types: dict[int, TensorType],\n    layouts: dict[int, StorageLayout],\n) -> list[str]:\n    """Emit one serial exact-typed full-root binary update and expose its fresh handle."""\n    root_type = types[op.root]\n    source_type = types[op.source]\n    if root_type != source_type or op.type != root_type:\n        raise RuntimeError("verified binary_inplace unexpectedly has mismatched types")\n    if op.operator not in {"add", "mul"}:\n        raise RuntimeError("verified binary_inplace unexpectedly has an unsupported operator")\n\n    root_layout = layouts[op.root]\n    source_layout = layouts[op.source]\n    operator = "+" if op.operator == "add" else "*"\n    c_type = _c_type(root_type.dtype)\n    lines = ["    {"]\n    if not root_type.shape:\n        destination = _root_ref(op.root, root_layout.offset, ())\n        lines.append(f"        {destination} = {destination} {operator} p{op.source}[0];")\n    else:\n        indent = "        "\n        axes = tuple(range(len(root_type.shape)))\n        for axis, bound in enumerate(root_type.shape):\n            lines.append(\n                f"{indent}for (int64_t i{axis} = 0; i{axis} < {bound}; ++i{axis}) {{"\n            )\n            indent += "    "\n        destination = _root_ref(op.root, root_layout.offset, root_layout.strides)\n        source_offset = _stride_offset(axes, source_layout.strides)\n        lines.append(\n            f"{indent}{destination} = {destination} {operator} p{op.source}[{source_offset}];"\n        )\n        for _ in root_type.shape:\n            indent = indent[:-4]\n            lines.append(f"{indent}}}")\n    lines.append("    }")\n    lines.append(f"    {c_type} *p{op.output} = p{op.root};")\n    lines.append("")\n    return lines\n\n\ndef _root_ref(root: int, base_offset: int, strides: tuple[int, ...]) -> str:\n''',
)

# C ABI dispatcher: keep effect serial even in parallel mode.
path = "src/tiny_tensor_compiler/c_abi_codegen.py"
replace(
    path,
    'from .loop_ir import LoopAlloc, LoopCopyInto, LoopInput, LoopProgram, LoopReturn, LoopView',
    'from .loop_ir import (\n    LoopAlloc,\n    LoopCopyInto,\n    LoopInplaceBinary,\n    LoopInput,\n    LoopProgram,\n    LoopReturn,\n    LoopView,\n)',
)
replace(path, 'from .write_codegen import emit_copy_into', 'from .write_codegen import emit_copy_into, emit_inplace_binary')
replace(
    path,
    '''        if isinstance(op, LoopCopyInto):\n            lines.extend(emit_copy_into(op, types, layouts))\n            continue\n        if isinstance(op, LoopReturn):''',
    '''        if isinstance(op, LoopCopyInto):\n            lines.extend(emit_copy_into(op, types, layouts))\n            continue\n        if isinstance(op, LoopInplaceBinary):\n            lines.extend(emit_inplace_binary(op, types, layouts))\n            continue\n        if isinstance(op, LoopReturn):''',
)

# input borrowing: remap logical handles but never make the mutation root caller-owned.
path = "src/tiny_tensor_compiler/input_binding.py"
replace(path, '    LoopInput,\n', '    LoopInput,\n    LoopInplaceBinary,\n')
replace(
    path,
    '''    @property\n    def value_types(self):''',
    '''    @property\n    def inplace_binaries(self):\n        return self.program.inplace_binaries\n\n    @property\n    def value_types(self):''',
)
replace(
    path,
    '''        if isinstance(op, LoopKernel):\n            transformed_operations.append(''',
    '''        if isinstance(op, LoopInplaceBinary):\n            transformed_operations.append(\n                LoopInplaceBinary(\n                    output=op.output + split_count,\n                    root=remap_handle(op.root),\n                    source=remap_handle(op.source),\n                    operator=op.operator,\n                    type=op.type,\n                )\n            )\n            continue\n\n        if isinstance(op, LoopKernel):\n            transformed_operations.append(''',
)
replace(
    path,
    '''        if isinstance(other, LoopCopyInto) and other.root == buffer:\n            return True\n''',
    '''        if isinstance(other, LoopCopyInto) and other.root == buffer:\n            return True\n        if isinstance(other, LoopInplaceBinary) and other.root == buffer:\n            return True\n''',
)

# Optimizers must treat destination-bearing binary updates as effects.
path = "src/tiny_tensor_compiler/passes.py"
replace(path, '_EFFECT_OPCODES = frozenset({"copy_into"})', '_EFFECT_OPCODES = frozenset({"copy_into", "binary_inplace"})')

print("in-place binary production patch applied")
