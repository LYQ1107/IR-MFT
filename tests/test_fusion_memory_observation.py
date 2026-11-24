"""Temporary artificial encoded images and tiny teacher; no dataset accuracy."""
import hashlib
import json
from pathlib import Path

import cv2
import pytest

from test_fusion_memory_online import engine, inputs, visual
from tools.fusion_memory_design import make_case
from tools.fusion_memory_observation import FixedSequenceImages, LENGTHS, observe_sequence


def artificial_files(root, frames=3):
    plan = dict(data_root=str(root/'RGB'), generated_ir_root=str(root/'IR'))
    digest = hashlib.sha256()
    for frame in range(1, frames+1):
        rgb, ir = inputs(frame)
        p = Path(plan['data_root'])/'train/BT-003/visible'/f'{frame:06d}.jpg'
        q = Path(plan['generated_ir_root'])/'train/BT-003/infrared_diffv2ir'/f'{frame:06d}.png'
        for path, pixels in [(p, rgb), (q, ir)]:
            path.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(path), pixels)
        row = ['BT-003', frame, hashlib.sha256(p.read_bytes()).hexdigest(), hashlib.sha256(q.read_bytes()).hexdigest()]
        digest.update(json.dumps(row, separators=(',', ':')).encode()+b'\n')
    return plan, digest.hexdigest()


def rgb_cases():
    source = visual(baseline=True)
    return [make_case(source, 'byte', 0.)]+[make_case(source, 'rgb_uniform', w) for w in [.05, .1]]


def test_complete_declared_train_prefix_has_exact_source_prefix_and_final_stream_hashes(tmp_path):
    plan, digest = artificial_files(tmp_path)
    cases = rgb_cases()
    value, _, _ = engine(cases)
    images = FixedSequenceImages(plan, 'train', 'BT-003', 3)
    result = observe_sequence(value, images, cases, tmp_path/'out', digest, {'synthetic': True})
    assert result['status'] == 'COMPLETE_F4_TRAIN_PREFIX_ENGINEERING_OBSERVATIONS'
    assert result['frames'] == 3 and result['complete_source_sequence_frames'] == 243
    assert not result['formal_Dev_Test_sequence_coverage'] and not result['formal_authorization_established_here']
    assert result['actual_IR_byte_frames_hashed'] == 3 and result['actual_IR_frames_decoded'] == 0
    assert len(result['prefix_source_rows']) == result['prefix_frames'] == 3
    assert value.closed
    for name, stream in result['streams'].items():
        assert stream['track_sha256'] == hashlib.sha256(Path(stream['track_path']).read_bytes()).hexdigest()
        assert result['prefix_literal_MOT_sha256'][name] == stream['track_sha256']
        assert result['prefix_packet_sha256'][name] == stream['detection_packet_sha256']
    assert (tmp_path/'out/sequence_manifests/BT-003.json').exists()


@pytest.mark.parametrize('split,seq', [('dev', 'BT-005'), ('test', 'BT-002')])
def test_formal_prefix_rejected_before_any_image_read(split, seq):
    with pytest.raises(ValueError, match='complete sequence'):
        FixedSequenceImages({}, split, seq, frames=1)
    assert sum(LENGTHS['dev'].values()) == 4834 and sum(LENGTHS['test'].values()) == 15985


def test_current_images_cannot_skip_or_repeat_frame(tmp_path):
    plan, _ = artificial_files(tmp_path)
    reader = FixedSequenceImages(plan, 'train', 'BT-003', 3)
    with pytest.raises(ValueError, match='in order'):
        reader.read_frame(2, False)
    reader.read_frame(1, False)
    for frame in [1, 3]:
        with pytest.raises(ValueError, match='in order'):
            reader.read_frame(frame, False)
    reader.read_frame(2, False)


def test_actual_scalar_image_is_decoded_only_when_visual_route_needs_it(tmp_path):
    plan, digest = artificial_files(tmp_path, 1)
    cases = [make_case(visual(), 'fused_uniform', .05)]
    value, _, _ = engine(cases)
    reader = FixedSequenceImages(plan, 'train', 'BT-003', 1)
    result = observe_sequence(value, reader, cases, tmp_path/'out', digest, {'synthetic': True})
    assert result['actual_IR_frames_decoded'] == 1


def test_changed_ir_bytes_are_rejected_even_when_rgb_network_does_not_consume_ir(tmp_path):
    plan, digest = artificial_files(tmp_path)
    q = Path(plan['generated_ir_root'])/'train/BT-003/infrared_diffv2ir/000001.png'
    q.write_bytes(b'CHANGED')
    cases = rgb_cases()
    value, _, _ = engine(cases)
    with pytest.raises(ValueError, match='Full source image-byte delivery differs'):
        observe_sequence(value, FixedSequenceImages(plan, 'train', 'BT-003', 3), cases, tmp_path/'out', digest, {})
    assert not (tmp_path/'out/sequence_manifests/BT-003.json').exists()
    errors = list((tmp_path/'out/sequence_manifests').glob('*.failure_*.json'))
    assert len(errors) == 1 and json.loads(errors[0].read_text())['emitted_frames'] == 3
    assert all((tmp_path/'out'/c['id']/'track_results/BT-003.txt').exists() for c in cases)


def test_partial_failure_retains_debug_mot_but_no_complete_receipt(tmp_path):
    plan, digest = artificial_files(tmp_path)
    cases = rgb_cases()
    value, teacher, _ = engine(cases)
    teacher.head.fail_at = 2
    with pytest.raises(RuntimeError, match='Intentional'):
        observe_sequence(value, FixedSequenceImages(plan, 'train', 'BT-003', 3), cases, tmp_path/'out', digest, {})
    assert not (tmp_path/'out/sequence_manifests/BT-003.json').exists()
    error = json.loads(next((tmp_path/'out/sequence_manifests').glob('*.failure_*.json')).read_text())
    assert error['emitted_frames'] == 1 and error['attempted_frame'] == 2
    assert value.closed and value.failed
    assert (tmp_path/'out'/cases[0]['id']/'track_results/BT-003.txt').read_text()


def test_complete_observations_are_never_overwritten(tmp_path):
    plan, digest = artificial_files(tmp_path, 1)
    cases = rgb_cases()
    a, _, _ = engine(cases)
    observe_sequence(a, FixedSequenceImages(plan, 'train', 'BT-003', 1), cases, tmp_path/'out', digest, {})
    b, _, _ = engine(cases)
    try:
        with pytest.raises(ValueError, match='Preserve existing'):
            observe_sequence(b, FixedSequenceImages(plan, 'train', 'BT-003', 1), cases, tmp_path/'out', digest, {})
    finally:
        b.close()
