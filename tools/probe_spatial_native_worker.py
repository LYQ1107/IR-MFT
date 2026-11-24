#!/usr/bin/env python3
"""CPU proof of the actual nine-arm training worker and B1 interface.

This is not the future complete pipeline proof and cannot authorize a GPU fit.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import torch

from tools.native_pyramid_content_guard import PinnedTrainImageReader
from tools.native_pyramid_train_data import OrderedTrainPixels, epoch_clips
from tools.native_pyramid_training_loop import state_sha256
from tools.neck72_runtime import require_host_memory, require_storage
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.spatial_native_fusion import SpatialNativeFusion
from tools.spatial_native_study import STUDY, arm_path, computation_key, context, frozen_json, make_arm
from tools.train_spatial_native import canary


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    plan, study, protocol, calibration, work, checkpoint, base = context()
    output = work/'native_spatial_worker_CPU_proof.json'
    assert not output.exists()
    require_host_memory(8*2**30, maximum_workers=1)
    require_storage(work)
    parent_path = ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261124_NATIVE_SPATIAL_CPU_PROOF.json'
    parent = json.loads(parent_path.read_text())
    sources = dict(parent['source_hashes'])
    names = ['tools/spatial_native_study.py', 'tools/spatial_native_rules.py',
        'tools/spatial_native_observer.py', 'tools/train_spatial_native.py',
        'tools/probe_spatial_native_worker.py', 'tests/test_spatial_native_rules.py',
        'tests/test_spatial_native_observer.py', str(STUDY.relative_to(ROOT))]
    sources.update({n: sha256(ROOT/n) for n in names})
    assert all(sha256(ROOT/n) == h for n, h in sources.items())
    started = time.monotonic()
    rules_log = work/'native_spatial_rules_pytest.log'
    rules_match = re.search(r'\b(\d+) passed\b', rules_log.read_text())
    assert rules_match and int(rules_match.group(1)) == 14
    log = work/'native_spatial_worker_CPU_pytest.log'
    with log.open('x') as stream:
        subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_spatial_native_observer.py'],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    match = re.search(r'\b(\d+) passed\b', log.read_text())
    assert match and int(match.group(1)) == 1
    fits = []
    for fit in study['normal_fits']:
        arm = make_arm(study, fit['id'], fit['variant'], fit['cue'])
        path = arm_path(base, fit['id'])
        frozen_json(path, arm)
        fits.append((fit, arm, path))
    clips = list(epoch_clips(study['Train_delivery']['lengths'], 1))[:3]
    expected = {}
    for clip in clips:
        for frame in clip.frames:
            for modality, root, folder, suffix in [('rgb', plan['data_root'], 'visible', 'jpg'),
                    ('ir', plan['generated_ir_root'], 'infrared_diffv2ir', 'png')]:
                path = Path(root)/'train'/clip.sequence/folder/f'{frame:06d}.{suffix}'
                encoded = path.read_bytes()
                expected[clip.sequence, frame, modality] = dict(sha256=hashlib.sha256(encoded).hexdigest(), bytes=len(encoded))
    reader = PinnedTrainImageReader(plan['data_root'], plan['generated_ir_root'], protocol['train'],
        expected_content=expected, max_bytes=256*2**20, reserve_bytes=40*2**30)
    from yolox.exp import get_exp
    teacher = get_exp(str(ROOT/'exps/yolox_mft26ir_visible.py'), None).get_model().float().eval().requires_grad_(False)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    teacher.load_state_dict(saved['model'], strict=True)
    del saved
    frozen_teacher = state_sha256(teacher.state_dict())
    records, initials = [], set()
    for fit, arm, path in fits:
        torch.manual_seed(0)
        module = SpatialNativeFusion(**arm['module_kwargs']).float()
        initial = state_sha256(module.state_dict())
        initials.add(initial)
        provider = OrderedTrainPixels(plan, protocol, study['Train_delivery']['lengths'],
            calibration, reader, study['training_input_hw'], 'cpu')
        value = canary(module, teacher, provider, arm, inference_frames=2)
        assert provider.prepared == 3 and value['delivered_Train_frames'] == 12
        assert all(value['optimizer_rows'][i]['optimizer_updated'] for i in range(3))
        assert frozen_teacher == state_sha256(teacher.state_dict())
        records.append(dict(fit_id=fit['id'], variant=fit['variant'], cue=fit['cue'],
            initial_native_sha256=initial, arm_sha256=sha256(path),
            computation_key=computation_key(arm), record=value))
        print(json.dumps(dict(fit=fit['id'], updates=3, real_Train_frames=12,
            independent_B1_frames=2, source_zero_missing_exact=True)), flush=True)
        del module, provider
    assert len(initials) == 1 and len(records) == 9
    assert all(sha256(ROOT/n) == h for n, h in sources.items())
    assert not torch.cuda.is_initialized()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/2**30
    assert peak <= 8
    require_storage(work)
    report = dict(status='COMPLETE_NINE_ARM_SPATIAL_WORKER_CPU_CANARY_NOT_FULL_PIPELINE',
        study_sha256=sha256(STUDY), source_hashes=sources, records=records,
        tests_passed=int(match.group(1))+int(rules_match.group(1)), new_observer_tests_passed=int(match.group(1)),
        reused_unchanged_rules_tests_passed=int(rules_match.group(1)),
        test_log_sha256=sha256(log), unchanged_rules_test_log_sha256=sha256(rules_log),
        source_image_content=reader.content_manifest(), reader_statistics=reader.statistics(),
        normal_arms={fit['id']: sha256(path) for fit, arm, path in fits},
        source_RGB_sha256=study['source_RGB_sha256'], protocol_sha256=study['protocol_sha256'],
        actual_updates=27, unique_actual_Train_frames=12, delivered_frames_across9fits=108,
        real_independent_B1_frame_conditions=18, initial_all9_same=True,
        original_RGB_encoder_head_BN_unchanged=True, future_worker_encoded_RAM_cap_GiB=2,
        fixed_seed=0, additional_seed_work=False, GPU_work=False, Dev_rows=0, Test_rows=0,
        saved_weights=False, image_or_FPN_disk_cache=False, peak_process_RSS_GiB=peak,
        seconds=time.monotonic()-started, full_Train_fit_completed=False,
        complete_Train_Dev_Test_cost_pipeline=False, full_pipeline_registration=False,
        scope='Actual registered worker optimizer/warmup on3chronological clips and2full-resolution independentB1frames for each9condition; CPUFP32. GPU requires12B1frames and its own proof. Partial Train engineering, not an epoch, new accuracy or full workflow validation.',
        goal_met=False, supported_innovations=0)
    dump(output, report)
    print(json.dumps(dict(status=report['status'], seconds=report['seconds'], peak_RSS_GiB=peak)), flush=True)


if __name__ == '__main__':
    main()
