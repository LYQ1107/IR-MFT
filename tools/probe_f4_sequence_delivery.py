#!/usr/bin/env python3
"""Actual fixed Train-prefix RGB observation delivery; no fusion qualification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import torch

from tools.fusion_memory_model_loading import load_source_rgb_cpu
from tools.fusion_memory_observation import FixedSequenceImages, observe_sequence
from tools.fusion_memory_online import OnlineFusionMemory, visual_routes
from tools.native_pyramid_training_loop import state_sha256
from tools.neck72_runtime import require_host_memory, require_storage

SPEC = ROOT/'configs/perceptual_f4_sequence_delivery_20261125.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pytest-receipt', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    started = time.monotonic()
    spec = read(SPEC)
    plan_path = ROOT/'configs/perceptual_mft_20261122.json'
    assert sha(plan_path) == spec['main_plan_sha256']
    assert all(sha(ROOT/name) == value for name, value in spec['frozen_dependencies'].items())
    plan = read(plan_path)
    work = Path(plan['work_root'])
    output, directory = work/'fusion_memory_sequence_delivery_CPU_proof.json', work/'fusion_memory_sequence_delivery_CPU'
    assert not output.exists() and not directory.exists()
    require_host_memory(4*2**30, maximum_workers=1)
    require_storage(work)
    tests = read(args.pytest_receipt)
    assert tests['exit_code'] == 0 and sha(tests['log']) == tests['log_sha256']
    assert all(sha(ROOT/name) == value for name, value in tests['source_hashes'].items())
    found = re.search(r'\b(\d+) passed\b', Path(tests['log']).read_text())
    assert found and int(found.group(1)) == 8
    old = spec['actual_CPU_scope']
    assert sha(old['prior_online_RGB_proof']) == old['prior_online_RGB_proof_sha256']
    prior = read(old['prior_online_RGB_proof'])
    loading = read(work/'fusion_memory_model_loading_CPU_proof.json')
    source = read(ROOT/'configs/perceptual_f4_model_loading_20261125.json')
    assert loading['source_RGB_sha256'] == source['source_RGB_sha256']
    allowed = {str(Path(plan['data_root'])/'train/BT-003/visible'/f'{f:06d}.jpg') for f in range(1, 9)}
    allowed.update(str(Path(plan['generated_ir_root'])/'train/BT-003/infrared_diffv2ir'/f'{f:06d}.png') for f in range(1, 9))
    attempts = dict(forbidden_data_reads=0, forbidden_IR_decodes=0, non_RGB_tensor_loads=0)
    def audit(event, values):
        if event != 'open' or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(values[0]))
        if (path.name == 'gt.txt' or path.suffix in ['.npy', '.npz'] or 'test' in path.parts or
                (path.suffix.lower() in ['.jpg', '.jpeg', '.png'] and str(path) not in allowed)):
            attempts['forbidden_data_reads'] += 1
            raise AssertionError('Only fixed eight Train RGB/IR source byte files may be read')
    sys.addaudithook(audit)
    original_load, actual_loads = torch.load, []
    def source_only(stream, *a, **kw):
        path = Path(getattr(stream, 'name', stream)).resolve()
        if path != Path(source['source_RGB_checkpoint']).resolve():
            attempts['non_RGB_tensor_loads'] += 1
            raise AssertionError('No learned fusion checkpoint may load')
        actual_loads.append(str(path))
        return original_load(stream, *a, **kw)
    with patch.object(torch, 'load', source_only):
        teacher = load_source_rgb_cpu(source['source_RGB_checkpoint'], source['source_RGB_sha256'])
    initial = state_sha256(teacher.state_dict())
    assert initial == loading['actual_CPU_RGB_state_sha256']
    calibration_path = work/'correlation_cache/calibration.json'
    expected_calibration = read(ROOT/'configs/perceptual_native_pyramid_study_20261124.json')['calibration_sha256']
    assert sha(calibration_path) == expected_calibration
    cases = [r['case'] for r in prior['cases']]
    assert len(cases) == 3 and [c['appearance_weight'] for c in cases] == [0., .05, .1]
    routes, _ = visual_routes(cases)
    engine = OnlineFusionMemory(teacher, cases, {key: None for key in routes}, (480, 864), read(calibration_path))
    images = FixedSequenceImages(plan, 'train', 'BT-003', 8)
    assert not engine.needs_actual_IR_image
    decoder, decoded_RGB = cv2.imdecode, [0]
    def rgb_only_decode(encoded, flag):
        if flag != cv2.IMREAD_COLOR:
            attempts['forbidden_IR_decodes'] += 1
            raise AssertionError('Ordinary RGB must not decode/use IR pixels')
        decoded_RGB[0] += 1
        return decoder(encoded, flag)
    def resources():
        require_storage(work)
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 <= 4
    with patch.object(cv2, 'imdecode', rgb_only_decode):
        result = observe_sequence(engine, images, cases, directory, old['expected_image_delivery_sha256'],
            dict(engineering=True, specification_sha256=sha(SPEC), original_RGB_sha256=source['source_RGB_sha256']),
            resource_check=resources)
    for row in prior['cases']:
        stream = result['streams'][row['case']['id']]
        assert all(stream[k] == row['stream'][k] for k in ['track_sha256', 'detection_packet_sha256', 'track_rows', 'detections'])
    assert result['actual_IR_byte_frames_hashed'] == decoded_RGB[0] == 8 and result['actual_IR_frames_decoded'] == 0
    assert not result['formal_Dev_Test_sequence_coverage'] and result['complete_source_sequence_frames'] == 243
    assert len(actual_loads) == 1 and not any(attempts.values())
    assert initial == state_sha256(teacher.state_dict()) and all(p.grad is None for p in teacher.parameters())
    assert not torch.cuda.is_initialized()
    names = ['tools/fusion_memory_observation.py', 'tools/probe_f4_sequence_delivery.py',
        'tests/test_fusion_memory_observation.py', str(SPEC.relative_to(ROOT))]
    sources = {name: sha(ROOT/name) for name in names}
    sources.update(spec['frozen_dependencies'])
    value = dict(status='COMPLETE_FIXED_TRAIN_PREFIX_OBSERVATION_COMPONENT_CPU',
        specification_sha256=sha(SPEC), source_hashes=sources, synthetic_tests_passed=8,
        pytest_receipt=str(args.pytest_receipt), pytest_receipt_sha256=sha(args.pytest_receipt),
        pytest_log_sha256=tests['log_sha256'], source_sequence='BT-003', actual_frames=8, full_Train_sequence_frames=243,
        sequence_receipt=str(directory/'sequence_manifests/BT-003.json'), sequence_receipt_sha256=sha(directory/'sequence_manifests/BT-003.json'),
        source_byte_delivery_sha256=result['original_image_bytes_delivery_sha256'], prior_independent_reference_all_case_packets_MOT_exact=True,
        original_RGB_weights_BN_gradients_unchanged=True, original_RGB_state_sha256=initial,
        actual_RGB_images_decoded=8, actual_IR_byte_files_hashed=8, actual_IR_images_decoded_or_consumed=0,
        guarded_attempt_counters=attempts, formal_Dev_Test_coverage=False, current_IR_learned_model_loads=0,
        actual_learned_fusion_transferred=False, GT_read=False, actual_Dev_Test_data_read=False,
        new_GPU_work=False, new_NN_fits=0, new_accuracy=False, persistent_new_weights_saved=False,
        new_image_FPN_ROI_disk_cache=False, formal_F4_launch_ready=False, fixed_seed=0, additional_seed_work=False,
        seconds=time.monotonic()-started, peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        scope='CPUFP32 RGB only; IR bytes hashed for immutable source provenance but never decoded or consumed. '
              'Eight Train frames are engineering prefix, not full243-frameTrain/Dev/Test. No fused checkpoint or new efficacy.',
        goal_met=False, supported_innovations=0)
    with output.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(output=str(output), seconds=value['seconds'], peak_RSS_GiB=value['peak_RSS_GiB'],
        actual_frames=8, new_accuracy=False)), flush=True)


if __name__ == '__main__':
    main()
