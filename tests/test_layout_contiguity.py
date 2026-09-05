from tiny_tensor_compiler.layout import StorageLayout


def test_singleton_axis_stride_does_not_break_c_contiguity():
    assert StorageLayout(offset=0, strides=(-1, 1)).is_contiguous((1, 1))
    assert StorageLayout(offset=0, strides=(-7, 1)).is_contiguous((1, 4))
    assert StorageLayout(offset=0, strides=(4, -9)).is_contiguous((3, 1))


def test_empty_layout_has_no_element_order_to_break_contiguity():
    assert StorageLayout(offset=0, strides=(-9, 3)).is_contiguous((0, 4))
    assert StorageLayout(offset=0, strides=(7, -2)).is_contiguous((3, 0))


def test_non_singleton_wrong_or_negative_stride_remains_non_contiguous():
    assert not StorageLayout(offset=0, strides=(-1, 1)).is_contiguous((2, 1))
    assert not StorageLayout(offset=0, strides=(5, 1)).is_contiguous((2, 3))
    assert not StorageLayout(offset=0, strides=(1, 3)).is_contiguous((2, 3))


def test_singleton_negative_stride_can_be_zero_copy_reshaped():
    layout = StorageLayout(offset=0, strides=(-1, 1))

    assert layout.reshaped((1, 1), (1,)) == StorageLayout(offset=0, strides=(1,))
