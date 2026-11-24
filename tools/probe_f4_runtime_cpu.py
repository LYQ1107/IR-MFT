#!/usr/bin/env python3
"""Actual independent RGB CPU runtime-component check on fixed eight Train frames.

No learned fusion, formal GPU cost, Dev/Test observation, GT, fit or new accuracy.
"""
import argparse
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

from tools.fusion_memory_evaluation import read, sha
from tools.fusion_memory_model_loading import load_source_rgb_cpu
from tools.fusion_memory_observation import FixedSequenceImages, write_json
from tools.fusion_memory_online import OnlineFusionMemory, visual_routes
from tools.fusion_memory_runtime import aggregate_cost, measure_sequence
from tools.native_pyramid_training_loop import state_sha256
from tools.neck72_runtime import require_host_memory, require_storage

SPEC = ROOT/'configs/perceptual_f4_runtime_engineering_20261125.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pytest-receipt', type=Path, required=True)
    parser.add_argument('--output-name', default='fusion_memory_runtime_CPU_proof.json')
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    started = time.monotonic()
    spec, tests = read(SPEC), read(args.pytest_receipt)
    assert tests['exit_code'] == 0 and tests['all_sources_unchanged']
    assert sha(tests['log']) == tests['log_sha256']
    assert all(sha(ROOT/p) == s for p, s in tests['source_hashes'].items())
    assert re.search(r'\b23 passed\b', Path(tests['log']).read_text())
    assert all(sha(ROOT/p) == s for p, s in spec['frozen_dependencies'].items())
    plan_path = ROOT/'configs/perceptual_mft_20261122.json'
    assert sha(plan_path) == spec['main_plan_sha256']
    plan = read(plan_path)
    work = Path(plan['work_root'])
    assert re.fullmatch(r'fusion_memory_runtime_CPU_proof(?:_R[0-9]+)?\.json', args.output_name)
    output = work/args.output_name
    assert not output.exists()
    require_host_memory(4*2**30, maximum_workers=1)
    require_storage(work)
    reference_path = Path(spec['reference_Train_prefix']['path'])
    assert sha(reference_path) == spec['reference_Train_prefix']['sha256']
    reference = read(reference_path)
    assert reference['split'] == 'train' and reference['sequence'] == 'BT-003' and reference['frames'] == 8
    assert reference['complete_source_sequence_frames'] == 243 and not reference['formal_Dev_Test_sequence_coverage']
    cases = [c for c in reference['cases'] if c['appearance_weight'] in [0., .05]]
    assert len(cases) == 2 and all(c['visual']['baseline'] and c['profile'] in ['byte', 'rgb_uniform'] for c in cases)
    calibration_path = work/'correlation_cache/calibration.json'
    assert sha(calibration_path) == spec['calibration_sha256']
    calibration = read(calibration_path)
    allowed = {str(Path(plan['data_root'])/'train/BT-003/visible'/f'{f:06d}.jpg') for f in range(1, 9)}
    allowed.update(str(Path(plan['generated_ir_root'])/'train/BT-003/infrared_diffv2ir'/f'{f:06d}.png') for f in range(1, 9))
    attempts = dict(forbidden_data_reads=0, forbidden_IR_decodes=0, forbidden_checkpoint_loads=0)
    def audit(event, values):
        if event != 'open' or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(values[0]))
        if (path.name == 'gt.txt' or path.suffix in ['.npy', '.npz'] or 'test' in path.parts or
                (path.suffix.lower() in ['.jpg', '.jpeg', '.png'] and str(path) not in allowed)):
            attempts['forbidden_data_reads'] += 1
            raise AssertionError('Only eight original Train RGB/IR byte files may enter the CPU check')
    sys.addaudithook(audit)
    original_loader, loads = torch.load, []
    def source_only(stream, *args, **kwargs):
        path = Path(getattr(stream, 'name', stream)).resolve()
        if path != Path(spec['source_RGB_checkpoint']).resolve():
            attempts['forbidden_checkpoint_loads'] += 1
            raise AssertionError('Only the original RGB teacher may load')
        loads.append(str(path))
        return original_loader(stream, *args, **kwargs)
    with patch.object(torch, 'load', source_only):
        teacher = load_source_rgb_cpu(spec['source_RGB_checkpoint'], spec['source_RGB_sha256'])
    initial = state_sha256(teacher.state_dict())
    original_decode, decoded = cv2.imdecode, [0]
    def rgb_decode(encoded, flags):
        if flags != cv2.IMREAD_COLOR:
            attempts['forbidden_IR_decodes'] += 1
            raise AssertionError('Ordinary RGB must not decode auxiliary IR')
        decoded[0] += 1
        return original_decode(encoded, flags)
    def resources():
        require_storage(work)
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= 4*2**30
    rows = []
    with patch.object(cv2, 'imdecode', rgb_decode):
        for case in cases:
            routes, _ = visual_routes([case])
            modules = {name: None for name in routes}
            measurements = measure_sequence(
                lambda: OnlineFusionMemory(teacher, [case], modules, (480, 864), calibration),
                lambda: FixedSequenceImages(plan, 'train', 'BT-003', 8), case, 'BT-003', reference,
                prefix_frames=8, warmup_frames=2, resource_check=resources)
            cost = aggregate_cost([measurements])
            assert cost['measured_frames'] == 6
            if case['appearance_weight']:
                assert all(measurements[phase]['memory_statistics'][case['id']]['changed_edges'] > 0
                           for phase in ['verification', 'timing'])
            rows.append(dict(case=case, measurements=measurements, CPU_engineering_cost=cost))
    assert decoded[0] == 32 and len(loads) == 1 and not any(attempts.values())
    assert state_sha256(teacher.state_dict()) == initial and all(p.grad is None for p in teacher.parameters())
    assert not teacher.backbone._forward_hooks and not torch.cuda.is_initialized()
    assert all(sha(ROOT/p) == s for p, s in spec['frozen_dependencies'].items())
    sources = dict(spec['frozen_dependencies'])
    sources[str(SPEC.relative_to(ROOT))] = sha(SPEC)
    write_json(output, dict(status='COMPLETE_ACTUAL_RGB_TRAIN_PREFIX_INDIVIDUAL_RUNTIME_COMPONENT_CPU',
        specification_sha256=sha(SPEC), source_hashes=sources, pytest_receipt=str(args.pytest_receipt),
        pytest_receipt_sha256=sha(args.pytest_receipt), pytest_passed=23,
        source_reference=str(reference_path), source_reference_sha256=sha(reference_path), rows=rows,
        source_sequence='BT-003', unique_source_frames=8, complete_source_Train_sequence_frames=243,
        actual_RGB_decodes=32, actual_IR_byte_hash_visits=32, actual_IR_decodes=0,
        individual_configurations=2, measured_frames_each=6,
        CPU_scope='FP32 480x864, two independent eight-frame passes per case, two warmup/six timed; not formal 800x1440 GPU cost or full sequence FPS',
        source_packets_and_MOT_prefix_exact_before_and_during_timing=True, original_RGB_weights_BN_gradients_unchanged=True,
        original_RGB_state_sha256=initial, guarded_attempt_counters=attempts,
        actual_learned_fusion_transferred=False, actual_Dev_Test_data_read=False, GT_read=False,
        new_GPU_work=False, new_NN_fits=False, new_accuracy=False, current_IR_learned_model_loads=0,
        new_image_FPN_ROI_disk_cache=False, formal_F4_launch_ready=False, fixed_seed=0, additional_seed_work=False,
        seconds=time.monotonic()-started, peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
        goal_met=False, supported_innovations=0), exclusive=True)
    print(json.dumps(dict(proof=str(output), sha256=sha(output), RGB_decodes=32, formal_GPU_cost=False,
                         new_accuracy=False)), flush=True)


if __name__ == '__main__':
    main()
