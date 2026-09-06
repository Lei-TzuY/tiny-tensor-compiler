from dataclasses import replace

import pytest

from tiny_tensor_compiler import (
    GraphBuilder,
    IndexMap,
    LoopBinaryInto,
    LoopProgram,
    lower_to_cpu,
    lower_to_loops,
)


def _broadcast_loops():
    builder = GraphBuilder()
    base = builder.input((2, 6), dtype="int32")
    source = builder.input((3,), dtype="int32")
    root = base.relu()
    target = root.slice(axis=1, start=0, stop=6, step=2)
    module = builder.finish(root.add_into(target, source))
    return lower_to_loops(lower_to_cpu(module))


def _replace_effect(loops, replacement: LoopBinaryInto) -> LoopProgram:
    return LoopProgram(
        tuple(
            replacement if isinstance(op, LoopBinaryInto) else op
            for op in loops.operations
        )
    )


def test_broadcast_binary_into_requires_explicit_source_map():
    loops = _broadcast_loops()
    effect = loops.binary_intos[0]
    assert effect.source_map == IndexMap((1,))

    with pytest.raises(ValueError, match="requires an explicit source index map"):
        _replace_effect(loops, replace(effect, source_map=None))


def test_broadcast_binary_into_rejects_noncanonical_source_map():
    loops = _broadcast_loops()
    effect = loops.binary_intos[0]

    with pytest.raises(ValueError, match="source index map does not match broadcasting semantics"):
        _replace_effect(loops, replace(effect, source_map=IndexMap((0,))))
