from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.content_verified_train_reader import ContentVerifiedTrainImageReader
from tools.structural_train_image_reader import TrainImageReader


def image(tmp_path, frame=1):
    path = tmp_path / 'ir/train/A/infrared_diffv2ir' / f'{frame:06d}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    value = np.arange(24, dtype=np.uint8).reshape(4, 6)+frame
    assert cv2.imwrite(str(path), value)
    return path, value


def reader(tmp_path, **kwargs):
    return ContentVerifiedTrainImageReader(tmp_path / 'rgb', tmp_path / 'ir', ['A'], reserve_bytes=0, **kwargs)


def test_metadata_replacement_requires_exact_bytes_and_preserves_pixels(tmp_path):
    path, value = image(tmp_path)
    old = TrainImageReader(tmp_path / 'rgb', tmp_path / 'ir', ['A'], reserve_bytes=0)
    current = reader(tmp_path)
    np.testing.assert_array_equal(old.read('A', 1, 'ir'), value)
    np.testing.assert_array_equal(current.read('A', 1, 'ir'), value)
    replacement = path.with_suffix('.replacement')
    replacement.write_bytes(path.read_bytes())
    replacement.replace(path)
    with pytest.raises(RuntimeError, match='changed during RAM reuse'):
        old.read('A', 1, 'ir')
    np.testing.assert_array_equal(current.read('A', 1, 'ir'), value)
    assert current.statistics()['metadata_only_revalidations'] == 1


def test_changed_content_is_rejected_on_cache_hit(tmp_path):
    path, value = image(tmp_path)
    current = reader(tmp_path)
    current.read('A', 1, 'ir')
    assert cv2.imwrite(str(path), value+1)
    with pytest.raises(RuntimeError, match='bytes changed during RAM reuse'):
        current.read('A', 1, 'ir')


def test_digest_remains_after_cache_eviction_or_cache_disabled(tmp_path):
    path, value = image(tmp_path)
    current = reader(tmp_path, max_bytes=0)
    current.read('A', 1, 'ir')
    assert not current.cache and len(current.content_hashes) == 1
    np.testing.assert_array_equal(current.read('A', 1, 'ir'), value)
    assert cv2.imwrite(str(path), value+1)
    with pytest.raises(RuntimeError, match='after cache eviction'):
        current.read('A', 1, 'ir')


def test_only_train_scalar_files_and_original_ram_budget(tmp_path):
    path, value = image(tmp_path)
    current = reader(tmp_path, max_bytes=len(path.read_bytes()))
    current.read('A', 1, 'ir')
    image(tmp_path, 2)
    current.read('A', 2, 'ir')
    assert current.bytes <= current.max_bytes
    with pytest.raises(ValueError):
        current.read('B', 1, 'ir')
    with pytest.raises(ValueError):
        current.read('A', 1, 'test')
