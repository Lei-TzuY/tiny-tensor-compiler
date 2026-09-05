from __future__ import annotations

import numpy as np
import pytest

from tiny_tensor_compiler import GraphBuilder, compile_module, lower_to_cpu, lower_to_loops
from tiny_tensor_compiler.layout import StorageLayout


def test_identity_view_preserves_negative_stride_layout_and_executes_native(tmp_path):
    builder = GraphBuilder("identity_view_reverse")
    source = builder.input((2, 3), dtype="int32")
    result = source.reverse(0).view((2, 3))
    module = builder.finish(result)

    loops = lower_to_loops(lower_to_cpu(module))
    assert len(loops.views) == 2
    reversed_view, identity_view = loops.views
    assert loops.value_layouts[reversed_view.output] == StorageLayout(
        offset=3,
        strides=(-3, 1),
    )
    assert loops.value_layouts[identity_view.output] == loops.value_layouts[reversed_view.output]

    values = np.arange(6, dtype=np.int32).reshape(2, 3)
    actual = compile_module(module, cache_dir=tmp_path)(inputs=(values,))
    np.testing.assert_array_equal(actual, values[::-1])


def test_shape_changing_view_still_rejects_noncontiguous_source():
    builder = GraphBuilder("noncontiguous_view_reshape")
    source = builder.input((2, 3), dtype="int32")
    result = source.reverse(0).view((6,))
    module = builder.finish(result)

    with pytest.raises(ValueError, match="cannot reshape a non-contiguous storage view"):
        lower_to_loops(lower_to_cpu(module))


def test_copy_reshape_materializes_noncontiguous_source(tmp_path):
    builder = GraphBuilder("noncontiguous_copy_reshape")
    source = builder.input((2, 3), dtype="int32")
    result = source.reverse(0).reshape((6,))
    module = builder.finish(result)

    values = np.arange(6, dtype=np.int32).reshape(2, 3)
    actual = compile_module(module, cache_dir=tmp_path)(inputs=(values,))
    np.testing.assert_array_equal(actual, values[::-1].reshape(6))
