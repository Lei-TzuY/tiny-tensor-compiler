import numpy as np
import pytest

from tiny_tensor_compiler import CompileBudget, GraphBuilder, SymbolicDim, compile_dynamic_module
from tiny_tensor_compiler.admission import DynamicSpecializationBudgetExceeded


def test_native_dynamic_specialization_cap_preserves_cached_binding():
    builder = GraphBuilder("native-dynamic-specialization-cap")
    batch = SymbolicDim("B")
    source = builder.input((batch,), "int32")
    result = (source + 1).relu()
    module = builder.finish(result)

    executable = compile_dynamic_module(
        module,
        budget=CompileBudget(max_dynamic_specializations=1),
    )

    first_input = np.array([-2, 0], dtype=np.int32)
    np.testing.assert_array_equal(
        executable(inputs=[first_input]),
        np.array([0, 1], dtype=np.int32),
    )
    assert executable.cached_bindings == ((('B', 2),),)

    with pytest.raises(DynamicSpecializationBudgetExceeded) as exc_info:
        executable(inputs=[np.array([1, 2, 3], dtype=np.int32)])
    assert exc_info.value.attempted_binding == (("B", 3),)
    assert executable.cached_bindings == ((('B', 2),),)

    second_input = np.array([5, -4], dtype=np.int32)
    np.testing.assert_array_equal(
        executable(inputs=[second_input]),
        np.array([6, 0], dtype=np.int32),
    )
    assert executable.cached_bindings == ((('B', 2),),)
