from pathlib import Path

replacements = {
    "src/tiny_tensor_compiler/backends/cpu.py": (
        "    LoopCopyInto,\n    LoopInput,\n    LoopInplaceBinary,\n",
        "    LoopCopyInto,\n    LoopInplaceBinary,\n    LoopInput,\n",
    ),
    "src/tiny_tensor_compiler/input_binding.py": (
        "    LoopCopyInto,\n    LoopInput,\n    LoopInplaceBinary,\n",
        "    LoopCopyInto,\n    LoopInplaceBinary,\n    LoopInput,\n",
    ),
    "src/tiny_tensor_compiler/loop_ir.py": (
        "    BufferCopyInto,\n    BufferInput,\n    BufferInplaceBinary,\n",
        "    BufferCopyInto,\n    BufferInplaceBinary,\n    BufferInput,\n",
    ),
}
for path, (old, new) in replacements.items():
    file = Path(path)
    text = file.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f"unexpected import block in {path}")
    file.write_text(text.replace(old, new, 1))
