#!/usr/bin/env python3
"""Real eight-frame RGB CPU check of the new engineering comparator only.

Two independently loaded CPU teachers, current ROI/memory replay and old source
prefix. No learned IR parent, GPU qualification, full Dev/Test or accuracy.
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

from tools.fusion_memory_engineering import EngineeringMismatch, compare_prefix
from tools.fusion_memory_evaluation import read, sha
from tools.fusion_memory_model_loading import load_source_rgb_cpu
from tools.fusion_memory_observation import FixedSequenceImages, write_json
from tools.fusion_memory_online import OnlineFusionMemory, visual_routes
from tools.native_pyramid_online_inference import OnlineNativeDetector
from tools.native_pyramid_training_loop import state_sha256
from tools.neck72_runtime import require_host_memory, require_storage

SPEC = ROOT/'configs/perceptual_f4_gpu_engineering_components_20261125.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pytest-receipt', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    started = time.monotonic()
    spec, tests = read(SPEC), read(args.pytest_receipt)
    assert tests['exit_code'] == 0 and tests['all_sources_unchanged'] and sha(tests['log']) == tests['log_sha256']
    assert re.search(r'\b19 passed\b', Path(tests['log']).read_text())
    assert all(sha(ROOT/p) == s for p, s in tests['source_hashes'].items())
    assert all(sha(ROOT/p) == s for p, s in spec['frozen_dependencies'].items())
    assert sha(ROOT/'configs/perceptual_mft_20261122.json') == spec['main_plan_sha256']
    plan = read(ROOT/'configs/perceptual_mft_20261122.json')
    work = Path(plan['work_root'])
    output = work/'fusion_memory_engineering_comparison_CPU_proof.json'
    assert not output.exists()
    require_host_memory(4*2**30, maximum_workers=1)
    require_storage(work)
    ref_path = Path(spec['reference_Train_prefix']['path'])
    assert sha(ref_path) == spec['reference_Train_prefix']['sha256']
    original = read(ref_path)
    assert original['sequence'] == 'BT-003' and original['frames'] == 8 and original['split'] == 'train'
    cases = original['cases']
    assert len(cases) == 3 and all(c['visual']['baseline'] for c in cases)
    allowed = {str(Path(plan['data_root'])/'train/BT-003/visible'/f'{f:06d}.jpg') for f in range(1, 9)}
    allowed.update(str(Path(plan['generated_ir_root'])/'train/BT-003/infrared_diffv2ir'/f'{f:06d}.png') for f in range(1, 9))
    attempts = dict(forbidden_data_reads=0, forbidden_IR_decodes=0, forbidden_checkpoint_loads=0)
    def audit(event, values):
        if event != 'open' or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(values[0]))
        if (path.name == 'gt.txt' or path.suffix in ['.npy', '.npz', '.csv'] or 'test' in path.parts or
                (path.suffix.lower() in ['.jpg', '.jpeg', '.png'] and str(path) not in allowed)):
            attempts['forbidden_data_reads'] += 1
            raise AssertionError('Only fixed eight Train RGB/IR byte files may enter the CPU comparison')
    sys.addaudithook(audit)
    loads, loader = [], torch.load
    def source_only(stream, *args, **kwargs):
        path = Path(getattr(stream, 'name', stream)).resolve()
        if path != Path(spec['source_RGB_checkpoint']).resolve():
            attempts['forbidden_checkpoint_loads'] += 1
            raise AssertionError('Only the original RGB source may load')
        loads.append(str(path))
        return loader(stream, *args, **kwargs)
    with patch.object(torch, 'load', source_only):
        teacher = load_source_rgb_cpu(spec['source_RGB_checkpoint'], spec['source_RGB_sha256'])
        reference_teacher = load_source_rgb_cpu(spec['source_RGB_checkpoint'], spec['source_RGB_sha256'])
    before = state_sha256(teacher.state_dict())
    assert before == state_sha256(reference_teacher.state_dict()) == spec['original_RGB_CPU_state_sha256']
    calibration_path = work/'correlation_cache/calibration.json'
    assert sha(calibration_path) == spec['calibration_sha256']
    calibration = read(calibration_path)
    routes, _ = visual_routes(cases)
    composed = OnlineFusionMemory(teacher, cases, {r: None for r in routes}, (480, 864), calibration)
    reference = OnlineNativeDetector(reference_teacher, [r['model'] for r in routes.values()],
        {r: None for r in routes}, (480, 864), calibration)
    images = FixedSequenceImages(plan, 'train', 'BT-003', 8)
    decode, decoded = cv2.imdecode, [0]
    def rgb_decode(encoded, flags):
        if flags != cv2.IMREAD_COLOR:
            attempts['forbidden_IR_decodes'] += 1
            raise AssertionError('RGB-only comparator must not decode/use actual IR')
        decoded[0] += 1
        return decode(encoded, flags)
    def resources():
        require_storage(work)
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= 4*2**30
    zero = {c['id']: original['prefix_literal_MOT_sha256'][c['id']] for c in cases if c['profile'] == 'byte'}
    zero_packets = {n: original['prefix_packet_sha256'][n] for n in zero}
    try:
        with patch.object(cv2, 'imdecode', rgb_decode):
            result = compare_prefix(composed, reference, images, cases, zero, frames=8,
                expected_source_rows=original['prefix_source_rows'], original_zero_packets=zero_packets,
                resource_check=resources)
    except Exception as error:
        write_json(work/(f'fusion_memory_engineering_comparison_CPU.failure_{time.time_ns()}.json'),
            dict(error=repr(error), details=error.details if isinstance(error, EngineeringMismatch) else None,
                specification_sha256=sha(SPEC), original_RGB_CPU_state_sha256=before,
                GPU_qualification=False, new_accuracy=False), exclusive=True)
        raise
    assert decoded[0] == images.hashed_IR_frames == 8 and images.decoded_IR_frames == 0
    assert len(loads) == 2 and not any(attempts.values())
    assert state_sha256(teacher.state_dict()) == state_sha256(reference_teacher.state_dict()) == before
    assert all(p.grad is None for model in [teacher, reference_teacher] for p in model.parameters())
    assert all(sha(ROOT/p) == s for p, s in spec['frozen_dependencies'].items())
    assert not teacher.backbone._forward_hooks and not torch.cuda.is_initialized()
    write_json(output, dict(status='COMPLETE_ACTUAL_RGB_CPU_ENGINEERING_COMPARATOR_NOT_GPU_QUALIFICATION',
        specification_sha256=sha(SPEC), source_hashes=dict(spec['frozen_dependencies'], **{str(SPEC.relative_to(ROOT)): sha(SPEC)}),
        pytest_receipt=str(args.pytest_receipt), pytest_receipt_sha256=sha(args.pytest_receipt), tests_passed=19,
        result=result, original_RGB_state_sha256=before, independent_RGB_checkpoint_loads=2,
        actual_RGB_decodes=8, independent_NN_forwards=16, actual_IR_byte_files_hashed=8, actual_IR_decodes=0,
        source_reference_sha256=sha(ref_path), source_reference=str(ref_path), guarded_attempt_counters=attempts,
        CPU_scope='Two original CPUFP32 teachers at480x864; eight Train frames, source/zero-MOT/KF/ID/current-ROI memory replay checks. Not actualIR model or800x1440GPU qualification.',
        previous_CPU_packet_anomaly_cause_still_unresolved=True, no_tolerance_relaxation=True,
        formal_GPU_qualification=False, full_Dev_Test_coverage=False, actual_learned_IR_fusion_transferred=False,
        GT_read=False, Dev_Test_data_read=False, new_GPU_work=False, new_NN_fits=False, new_accuracy=False,
        fixed_seed=0, additional_seed_work=False, new_image_FPN_ROI_disk_cache=False,
        seconds=time.monotonic()-started, peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30,
        goal_met=False, supported_innovations=0), exclusive=True)
    print(dict(proof=str(output), seconds=time.monotonic()-started, formal_GPU_qualification=False), flush=True)


if __name__ == '__main__':
    main()
