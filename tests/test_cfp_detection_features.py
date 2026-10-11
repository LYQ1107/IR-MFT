import importlib.util
from pathlib import Path

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location('cfp_detector_features_cli', Path(__file__).parents[1] / 'tools/cfp_cache_detection_features.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_detection_index_survives_empty_frames_and_unextractable_crops(tmp_path):
    directory = tmp_path / 'train/a'
    directory.mkdir(parents=True)
    images = {}
    for frame in range(1, 5):
        image = np.zeros((16, 32, 3), np.uint8)
        image[:, :, 2 if frame == 2 else 0] = 255
        path = directory / f'{frame}.png'
        cv2.imwrite(str(path), image)
        images[frame] = dict(vi_file_name=f'a/{frame}.png', ir_file_name='not_read.png')
    cache = dict(frame_ids=np.arange(1, 5), offsets=np.array([0, 0, 2, 2, 3]),
                 boxes=np.array([[0, 0, 32, 16], [100, 100, 120, 120], [0, 0, 32, 16]], np.float32))
    dataset = module.DetectionCrops(tmp_path, 'train', images, cache, paired=False)
    assert len(dataset) == 3
    first, _, valid, _, position = dataset[0]
    assert valid and position == 0 and first[0].max() > 2.
    invalid, _, valid, _, position = dataset[1]
    assert not valid and position == 1 and np.all(invalid.numpy() == 0)
    last, _, valid, _, position = dataset[2]
    assert valid and position == 2 and last[2].max() > 2.


def test_auxiliary_control_reads_separate_root_without_changing_rgb(tmp_path):
    primary, auxiliary = tmp_path / 'primary', tmp_path / 'control'
    for root in (primary, auxiliary):
        (root / 'train/a').mkdir(parents=True)
    rgb = np.zeros((16, 32, 3), np.uint8)
    rgb[:, :, 2] = 255
    control = np.zeros_like(rgb)
    control[:, :, 0] = 255
    cv2.imwrite(str(primary / 'train/a/rgb.png'), rgb)
    cv2.imwrite(str(auxiliary / 'train/a/view.png'), control)
    images = {1: dict(vi_file_name='a/rgb.png', ir_file_name='a/view.png')}
    cache = dict(frame_ids=np.array([1]), offsets=np.array([0, 1]), boxes=np.array([[0, 0, 32, 16]]))
    dataset = module.DetectionCrops(primary, 'train', images, cache, True, auxiliary)
    r, q, valid_r, valid_q, _ = dataset[0]
    assert valid_r and valid_q and r[0].max() > 2 and q[2].max() > 2
    assert r[2].max() < 0 and q[0].max() < 0
