"""Paired identity sampling preserves namespaces, labels, time and crop geometry."""
import json

import numpy as np
import pytest
import torch

from cfp.identity_data import build_identity_index, crop_to_tensor, PairedIdentityDataset


def test_video_identity_namespace_and_temporal_coverage(tmp_path):
    (tmp_path / 'annotations').mkdir()
    videos = [{'id': 1, 'file_name': 'a'}, {'id': 2, 'file_name': 'b'}]
    images, anns = [], []
    for video in videos:
        for frame in range(1, 11):
            i = (video['id'] - 1) * 10 + frame
            seq = video['file_name']
            images.append(dict(id=i, video_id=video['id'], frame_id=frame,
                               vi_file_name=f'{seq}/visible/{frame:06d}.jpg',
                               ir_file_name=f'{seq}/infrared/{frame:06d}.jpg'))
            anns.append(dict(image_id=i, track_id=1, bbox=[0, 0, 5, 5], conf=1))
    (tmp_path / 'annotations/train.json').write_text(json.dumps(dict(videos=videos, images=images, annotations=anns)))
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps(dict(train=['a'], dev=['b'])))
    index = build_identity_index(tmp_path, protocol, samples_per_identity=4)
    assert index['train_identity_count'] == 1
    a, b = [r for r in index['records'] if r['sequence'] == 'a'], [r for r in index['records'] if r['sequence'] == 'b']
    assert [r['frame_id'] for r in a] == [1, 4, 7, 10]
    assert all(r['train_label'] == 0 for r in a)
    assert all(r['train_label'] == -1 for r in b)
    # Never choose one ambiguous box merely because it is first in the file.
    data = dict(videos=videos, images=images, annotations=anns + [dict(anns[0], bbox=[1, 1, 5, 5])])
    (tmp_path / 'annotations/train.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Duplicate'):
        build_identity_index(tmp_path, protocol, samples_per_identity=4)
    protocol.write_text(json.dumps(dict(train=['a'], dev=['b'], identity_annotation_policy='exclude_all_duplicate_keys')))
    clean = build_identity_index(tmp_path, protocol, samples_per_identity=4)
    assert clean['excluded_ambiguous_labels'] == [dict(sequence='a', track_id=1, frame_id=1, rows=2)]
    assert all(r['frame_id'] != 1 for r in clean['records'] if r['sequence'] == 'a')


def test_clipped_crop_shape_finite_and_color_order():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[:, :, 2] = 255  # OpenCV BGR red
    crop = crop_to_tensor(image, [-2, -3, 20, 10], (16, 32))
    assert crop.shape == (3, 16, 32)
    assert torch.isfinite(crop).all()
    assert crop[0].max() > 2
    assert crop[2].min() < -1
    with pytest.raises(ValueError):
        crop_to_tensor(image, [0, 0, 0, 10])
    with pytest.raises(ValueError):
        crop_to_tensor(image, [100, 100, 5, 5])


def test_pair_decode_alignment_is_checked(tmp_path):
    import cv2
    directory = tmp_path / 'train/a'
    directory.mkdir(parents=True)
    cv2.imwrite(str(directory / 'rgb.jpg'), np.zeros((10, 20, 3), np.uint8))
    cv2.imwrite(str(directory / 'ir.jpg'), np.zeros((5, 20, 3), np.uint8))
    dataset = PairedIdentityDataset(tmp_path, [dict(image_id=1, rgb_file='a/rgb.jpg', ir_file='a/ir.jpg', bbox_xywh=[0, 0, 5, 5])])
    with pytest.raises(ValueError, match='Invalid paired image'):
        dataset[0]
