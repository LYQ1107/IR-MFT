import importlib.util
from pathlib import Path

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location('cfp_control_views_cli', Path(__file__).parents[1] / 'tools/cfp_build_control_views.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_background_control_uses_same_past_state_and_does_not_mutate_rgb():
    rng = np.random.default_rng(31)
    images = [rng.integers(0, 256, (64, 96, 3), dtype=np.uint8) for _ in range(5)]
    params = dict(alpha=.01, percentiles=[2., 98.5], gamma=.42)
    def stream(frames):
        background = frames[0].astype(np.float32)
        result = []
        for frame in frames:
            before = frame.copy()
            result.append(module.transform(frame, 'rgb_background_difference', params, background))
            cv2.accumulateWeighted(frame, background, params['alpha'])
            assert np.array_equal(before, frame)
        return result
    expected = stream(images)
    changed = stream(images[:3] + [np.zeros_like(images[0]) for _ in range(2)])
    assert np.all(expected[0] == 0)
    assert all(np.array_equal(x, y) for x, y in zip(expected[:3], changed[:3]))


def test_current_frame_contrast_is_shape_preserving_without_background():
    image = np.random.default_rng(32).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    original = image.copy()
    clahe = cv2.createCLAHE(2., (8, 8))
    result = module.transform(image, 'rgb_contrast', {}, clahe=clahe)
    assert result.shape == image.shape and result.dtype == np.uint8
    assert np.array_equal(image, original)
