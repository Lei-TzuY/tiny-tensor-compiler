from __future__ import annotations

import numpy as np

from tiny_tensor_compiler.differential import _compare_results


def test_differential_comparator_supports_exact_int64_index_outputs():
    expected = np.array([0, 2, 2, 0], dtype=np.int64)

    assert _compare_results(expected.copy(), expected) is None

    wrong = expected.copy()
    wrong[2] = 1
    assert _compare_results(wrong, expected) == "mismatch:bytes:0"
