import numpy as np

from tools.structural_grid import structural_grid


def test_structure_grid_tracks_detector_padding_and_rejects_flat_views():
    image = np.ones((108, 192), np.uint8)*71
    values, valid = structural_grid(image, (96, 192))
    assert values.shape == (8, 12, 24) and valid.shape == (1, 12, 24)
    assert values.dtype == np.float16 and valid.dtype == np.bool_
    assert not valid.any()
    image[20:80, 20:140] = np.random.default_rng(3).integers(0, 256, (60, 120), dtype=np.uint8)
    values, valid = structural_grid(image, (96, 192))
    assert valid.any()
    assert not valid[:, :, -2:].any()
    assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1
