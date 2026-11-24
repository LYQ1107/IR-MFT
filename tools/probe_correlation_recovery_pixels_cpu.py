#!/usr/bin/env python3
"""Two real Train frames: new CGR core versus original FP32 CPU math.

Not a GPU parity proof, training run, full Dev/Test or performance benchmark.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch

from tools.build_correlation_cache import auxiliary_input
from tools.correlation_gate_observer import CorrelationGateObserver, SCOPE
from tools.correlation_recovery_inference import CorrelationCaseDetector, load_adapters
from tools.extract_recoverable_correlation import detections as legacy_detections
from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.exp import get_exp
from yolox.models.local_correlation_repair import CorrelationRepair
from yolox.tracker.foveated_regions import exact_half_preproc
from yolox.tracker.priority_observations import priority_union
from yolox.utils import postprocess


def read(path):
    return json.loads(Path(path).read_text())


def tensor_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    output = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261124_CGR_REAL_PIXELS_CPU_PROOF.json'
    assert not output.exists()
    started = time.perf_counter()
    spec_path = ROOT / 'configs/perceptual_correlation_recovery_backfill_20261124.json'
    spec = read(spec_path)
    plan = read(ROOT / 'configs/perceptual_mft_20261122.json')
    repair_path = ROOT / 'configs/perceptual_correlation_recoverability_20261123.json'
    parent = read(ROOT / 'configs/perceptual_correlation_repair_20261123.json')
    calibration = read(Path(plan['work_root']) / 'correlation_cache/calibration.json')
    checkpoint = Path(parent['source_RGB_checkpoint'])
    assert sha256(checkpoint) == parent['source_RGB_sha256']
    model = get_exp(str(ROOT / 'exps/yolox_mft26ir_visible.py'), None).get_model()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    model = model.cpu().float().eval().requires_grad_(False)
    before = tensor_digest(model)
    cases = spec['cases']
    modules, _ = load_adapters(cases, sha256(repair_path), parent['source_RGB_sha256'], 'cpu')
    images, ir_images, input_sources = [], [], []
    for frame in [1, 2]:
        rgb_path = Path(plan['data_root']) / 'train/BT-003/visible' / f'{frame:06d}.jpg'
        ir_path = Path(plan['generated_ir_root']) / 'train/BT-003/infrared_diffv2ir' / f'{frame:06d}.png'
        images.append(cv2.imread(str(rgb_path)))
        ir_images.append(cv2.imread(str(ir_path), -1))
        input_sources.append(dict(sequence='BT-003', frame=frame, split='train',
            RGB_sha256=sha256(rgb_path), actual_November_IR_sha256=sha256(ir_path)))
    shape = parent['training_input_hw']
    views = {name: [] for name in ['rgb', 'gray', 'ir', 'ir_roll']}
    ratios = []
    for rgb, ir in zip(images, ir_images):
        value, ratio = exact_half_preproc(rgb, shape, (.485, .456, .406), (.229, .224, .225))
        views['rgb'].append(value)
        ratios.append(ratio)
        for name, scalar in [('gray', cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)), ('ir', ir),
                ('ir_roll', np.roll(ir, (ir.shape[0]//4, ir.shape[1]//4), (0, 1)))]:
            norm = calibration['statistics']['ir' if name.startswith('ir') else 'gray']
            value, _ = auxiliary_input(scalar, shape, norm['mean'], norm['std'])
            views[name].append(value)
    # Direct transcription of the preserved legacy inference operations, using
    # the same real Train pixels with an explicitly FP32 CPU encoder/head.
    features = {name: model.backbone(torch.from_numpy(np.stack(values)).float()) for name, values in views.items()}
    teacher = model.head(features['rgb'])
    scores, offset = [], 0
    for feature in features['rgb']:
        count = feature.shape[-2]*feature.shape[-1]
        score = teacher[:, offset:offset+count, 4]*teacher[:, offset:offset+count, 5]
        scores.append(score.reshape(2, 1, *feature.shape[-2:]))
        offset += count
    fresh = [legacy_detections(value, ratio) for value, ratio in zip(postprocess(teacher.clone(), 1, .1, .7), ratios)]
    expected = {}
    for row in cases:
        if row.get('baseline'):
            expected[row['id']] = fresh
            continue
        intervention = row['intervention']
        view = 'ir_roll' if intervention == 'IR_image_spatial_roll' else row['source']['cue']
        flags = torch.full((2,), intervention != 'unavailable', dtype=torch.bool)
        repaired = modules[row['id']](features['rgb'], None if intervention == 'unavailable' else features[view], scores, flags)
        values = [legacy_detections(value, ratio) for value, ratio in zip(postprocess(model.head(repaired), 1, .1, .7), ratios)]
        if row['source']['output_policy'] == 'priority':
            values = [old.copy() if not ok else priority_union(old, [(current, 1.)])[0]
                      for old, current, ok in zip(fresh, values, flags)]
        expected[row['id']] = values
    observer = CorrelationGateObserver()
    try:
        for name, module in modules.items():
            if module is not None:
                observer.attach(name, module)
        engine = CorrelationCaseDetector(model, cases, modules, shape, calibration)
        actual, actual_fresh = engine.batch(images, ir_images, fresh)
        for left, right in zip(actual_fresh, fresh):
            np.testing.assert_array_equal(left, right)
        for name, values in actual.items():
            for value, original in zip(values, expected[name]):
                np.testing.assert_array_equal(value, original)
        gates = {name: observer.summary(name) for name in observer.rows}
    finally:
        observer.close()
    zero_cases, zero_modules = [], {}
    for cue in ['rgb', 'gray', 'ir']:
        for radius in [0, 2]:
            name = f'zero_{cue}_radius{radius}'
            zero_cases.append(dict(id=name, intervention='normal', source=dict(cue=cue, output_policy='raw')))
            zero_modules[name] = CorrelationRepair(radius=radius).float().eval().requires_grad_(False)
    zero = CorrelationCaseDetector(model, zero_cases, zero_modules, shape, calibration)
    zeros, direct = zero.batch(images, ir_images, fresh)
    for values in zeros.values():
        for value, original in zip(values, direct):
            np.testing.assert_array_equal(value, original)
    assert tensor_digest(model) == before
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    report = dict(status='COMPLETE_REAL2TRAIN_CGR_CPU_FORWARD_AND_ZERO_MISSING_PROOF',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_inputs=input_sources,
        CPU_precision='Original half preprocessing values, FP32 CPU encoder/head and CGR modules. Not GPU FP16 parity.',
        input_hw=shape, independent_batch_size=2, frozen_cases=len(cases),
        all_frozen_cases_exact_to_original_CGR_math=True, source_full_head_and_NMS_exact=True,
        learned_missing_exact_to_fresh_RGB=True, zero_branches_checked=len(zero_cases),
        all_zero_branches_full_head_and_NMS_exact=True, frozen_RGB_weights_and_BN_unchanged=True,
        gate_statistics=gates, gate_statistics_scope=SCOPE, source_RGB_sha256=parent['source_RGB_sha256'],
        specification_sha256=sha256(spec_path), source_hashes={name: sha256(ROOT / name) for name in [
            'tools/probe_correlation_recovery_pixels_cpu.py', 'tools/correlation_recovery_inference.py',
            'tools/correlation_gate_observer.py', 'tools/extract_recoverable_correlation.py',
            'yolox/models/local_correlation_repair.py']},
        GPU_work=False, GT_read=False, Dev_read=False, Test_read=False,
        seconds=time.perf_counter()-started, actual_GPU_engineering_pending=True,
        formal_4_5_controller_not_yet_implemented=True, goal_met=False, supported_innovations=0)
    dump(output, report)
    print(json.dumps({k: report[k] for k in ['status', 'frozen_cases', 'zero_branches_checked',
        'all_frozen_cases_exact_to_original_CGR_math', 'seconds', 'actual_GPU_engineering_pending']}), flush=True)


if __name__ == '__main__':
    main()
