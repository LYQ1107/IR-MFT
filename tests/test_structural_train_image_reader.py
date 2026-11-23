from pathlib import Path

import cv2
import numpy as np
import pytest

from tools import structural_train_image_reader as module


def image(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), value)


def test_encoded_reuse_preserves_pixels_and_rejects_changed_source(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'available_memory_bytes', lambda: 10**9)
    rgb = tmp_path / 'train/TrainA/visible/000001.jpg'
    ir = tmp_path / 'train/TrainA/infrared_diffv2ir/000001.png'
    rng = np.random.default_rng(0)
    image(rgb, rng.integers(0, 256, (37, 53, 3), dtype=np.uint8))
    image(ir, rng.integers(0, 256, (37, 53), dtype=np.uint8))
    reader = module.TrainImageReader(tmp_path, tmp_path, ['TrainA'], 10**6, 0)
    for modality, path, flag in [('rgb', rgb, 1), ('ir', ir, -1)]:
        expected = cv2.imread(str(path), flag)
        assert np.array_equal(reader.read('TrainA', 1, modality), expected)
        assert np.array_equal(reader.read('TrainA', 1, modality), expected)
    assert reader.hits == reader.misses == 2
    image(ir, np.zeros((39, 53), np.uint8))
    with pytest.raises(RuntimeError, match='changed'):
        reader.read('TrainA', 1, 'ir')
    with pytest.raises(ValueError, match='registered Train'):
        reader.read('HeldOut', 1, 'rgb')


def test_byte_limit_and_host_memory_reserve(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'available_memory_bytes', lambda: 10**9)
    rng = np.random.default_rng(0)
    paths = [tmp_path / f'train/TrainA/infrared_diffv2ir/{f:06d}.png' for f in [1, 2]]
    for path in paths:
        image(path, rng.integers(0, 256, (31, 47), dtype=np.uint8))
    reader = module.TrainImageReader(tmp_path, tmp_path, ['TrainA'], max(p.stat().st_size for p in paths), 0)
    reader.read('TrainA', 1, 'ir')
    reader.read('TrainA', 2, 'ir')
    assert reader.evictions == 1 and reader.bytes <= reader.max_bytes
    monkeypatch.setattr(module, 'available_memory_bytes', lambda: 0)
    no_room = module.TrainImageReader(tmp_path, tmp_path, ['TrainA'], 10**6, 40 * 2**30)
    no_room.read('TrainA', 1, 'ir')
    assert no_room.bytes == 0 and not no_room.cache
