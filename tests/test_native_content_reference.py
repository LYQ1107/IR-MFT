import hashlib
import json

import cv2
import numpy as np
import pytest

from tools.native_pyramid_content_guard import make_image_reader


def fixtures(tmp_path):
    seqs = ['A', 'B', 'C', 'D', 'E', 'F']
    lengths = dict.fromkeys(seqs, 1)
    records = []
    for seq in seqs:
        for modality in ['rgb', 'ir']:
            path = tmp_path / modality / 'train' / seq / ('visible/000001.jpg' if modality == 'rgb' else 'infrared_diffv2ir/000001.png')
            path.parent.mkdir(parents=True, exist_ok=True)
            image = np.arange(24, dtype=np.uint8).reshape(4, 6)
            if modality == 'rgb':
                image = np.repeat(image[..., None], 3, axis=2)
            assert cv2.imwrite(str(path), image)
            records.append(dict(sequence=seq, frame=1, modality=modality,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size))
    for variant in ['111', '011', '101', '110']:
        folder = tmp_path / 'structural_full_train_fit' / ('ir_'+variant+'_s0')
        folder.mkdir(parents=True)
        manifest = folder / 'source.json'
        manifest.write_text(json.dumps(records))
        (folder / 'full_train_complete.json').write_text(json.dumps(dict(status='COMPLETE_THREE_FULL_TRAIN_TRAVERSALS',
            seed=0, logical_Train_epochs_completed=3, source_image_content_manifest=dict(path=str(manifest),
            sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()))))
    return dict(data_root=str(tmp_path / 'rgb'), generated_ir_root=str(tmp_path / 'ir')), dict(train=seqs), lengths


def test_completed_data_reference_is_checked_before_native_inputs_are_returned(tmp_path):
    plan, protocol, lengths = fixtures(tmp_path)
    reader, info = make_image_reader(tmp_path, plan, protocol, lengths, 'ir')
    assert info['expected_input_entries'] == 12 and not reader.content_hashes
    reader.read('A', 1, 'ir')
    assert len(reader.content_hashes) == 1  # Expected hashes are not invented delivery.
    reader_rgb, info_rgb = make_image_reader(tmp_path, plan, protocol, lengths, 'rgb')
    assert info_rgb['expected_input_entries'] == 6
    path = tmp_path / 'ir/train/B/infrared_diffv2ir/000001.png'
    assert cv2.imwrite(str(path), np.zeros((4, 6), np.uint8))
    with pytest.raises(RuntimeError, match='differs from completed reference'):
        reader.read('B', 1, 'ir')


def test_disagreeing_completed_conditions_cannot_define_a_new_data_version(tmp_path):
    plan, protocol, lengths = fixtures(tmp_path)
    path = tmp_path / 'structural_full_train_fit/ir_110_s0/source.json'
    records = json.loads(path.read_text())
    records[0]['sha256'] = '0'*64
    path.write_text(json.dumps(records))
    complete_path = path.parent / 'full_train_complete.json'
    complete = json.loads(complete_path.read_text())
    complete['source_image_content_manifest']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    complete_path.write_text(json.dumps(complete))
    with pytest.raises(AssertionError, match='identical image bytes'):
        make_image_reader(tmp_path, plan, protocol, lengths, 'ir')
