from tiny_tensor_compiler import StorageLayout


def test_empty_storage_transforms_canonicalize_offset_to_zero():
    base = StorageLayout.contiguous((0, 6))

    reshaped = base.reshaped((0, 6), (0, 3, 2))
    sliced, sliced_shape = base.sliced(
        (0, 6),
        axis=1,
        start=1,
        stop=6,
        step=2,
    )
    reversed_layout = base.reversed((0, 6), axis=1)
    transposed, transposed_shape = base.permuted((0, 6), (1, 0))

    assert reshaped.offset == 0
    assert sliced_shape == (0, 3)
    assert sliced.offset == 0
    assert reversed_layout.offset == 0
    assert transposed_shape == (6, 0)
    assert transposed.offset == 0

    for layout, shape in (
        (reshaped, (0, 3, 2)),
        (sliced, sliced_shape),
        (reversed_layout, (0, 6)),
        (transposed, transposed_shape),
    ):
        layout.validate_bounds(shape, storage_elements=0)
