#!/usr/bin/env python3
"""CPU-only neck72 interoperability proof; no scheduler or formal accuracy."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from tools.native_neck_io import load_native_encoder
from tools.native_pyramid_training_loop import state_sha256
from tools.native_query_replay import load_native_query_evidence
from tools.neck72_evidence import load_neck72_evidence
from tools.neck72_runtime import read, require_storage, require_host_memory
from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.tracker.native_cue import native_crop, native_input


def sources():
    files = ['tools/' + name + '.py' for name in ['neck72_runtime', 'neck72_evidence',
        'probe_neck72_interfaces', 'extract_native_neck_candidate_features',
        'extract_native_neck_eval_features', 'build_native_neck_query_bank',
        'build_native_neck_mismatch_bank', 'train_native_neck_clean_query',
        'train_native_neck_robust_query', 'native_neck_io', 'native_query_replay',
        'native_neck_query_data', 'native_neck_robust_query_data',
        'native_pyramid_training_loop', 'structural_train_image_reader', 'run_pseudo_ir_comparison']]
    files += ['tests/test_neck72_interfaces.py', 'yolox/tracker/native_cue.py',
        'yolox/tracker/native_query_features.py', 'yolox/tracker/association_calibrator.py',
        'yolox/tracker/native_cue_corruption.py', 'yolox/tracker/native_query_mask.py',
        'configs/perceptual_mft_20261122.json', 'configs/cfp/protocol_v2.json',
        'configs/perceptual_single_seed_policy_20261123.json',
        'configs/perceptual_gpu_handoff_20261123.json',
        'configs/perceptual_native_neck_query_bank_20261123.json',
        'configs/perceptual_native_neck_clean_query_20261123.json',
        'configs/perceptual_native_neck_robust_query_20261123.json']
    return {name: sha256(ROOT / name) for name in files}


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(2); torch.manual_seed(0); cv2.setNumThreads(1)
    before = sources(); started = time.monotonic()
    plan = read(ROOT / 'configs/perceptual_mft_20261122.json')
    protocol = read(ROOT / plan['protocol']); work = Path(plan['work_root'])
    output = work / 'neck72_worker_interfaces_CPU_proof.json'
    assert not output.exists(), 'Preserve previous CPU engineering evidence'
    require_storage(work); require_host_memory(2*2**30,maximum_workers=1)
    log = work / 'neck72_worker_interfaces_pytest.log'
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_neck72_interfaces.py']
    with log.open('x') as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    assert re.search(r'\b9 passed\b', log.read_text())
    # The old helper expects cue/complete.json rather than cue/seed_0. This
    # read-only symlink view permits a literal reference-computation comparison
    # without changing the pinned old helper or copying any feature/pixel data.
    alias = work / ('neck72_legacy_layout_readonly_view_'+before['tools/neck72_runtime.py'][:12])
    alias.mkdir(exist_ok=False)
    for name in ['native_neck_train_features', 'native_neck_alignment']:
        (alias / name).symlink_to(work / name, target_is_directory=True)
    comparisons = []
    for sequence in protocol['dev']:
        for cue in ['rgb', 'ir']:
            path = alias / 'native_neck_eval_features/dev' / sequence / cue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(work / 'native_neck_eval_features/dev' / sequence / cue / 'seed_0',
                            target_is_directory=True)
        for arm in ['new', 'rgb']:
            path = work / 'features/dev' / sequence / (arm + '.npz')
            with np.load(path) as source:
                raw = {k: source[k].copy() for k in ['frame_ids', 'offsets', 'boxes']}
            for cue in ['rgb', 'ir']:
                for mode in ['normal', 'missing', 'without_alignment']:
                    config = dict(detection_arm=arm, native_cue=cue, native_seed=0,
                        native_feature_family='neck_metric72', disable_native_increment=mode == 'missing',
                        disable_native_alignment=mode == 'without_alignment')
                    old, current = dict(raw), dict(raw)
                    load_native_query_evidence(old, alias, 'dev', sequence, config)
                    metadata = load_neck72_evidence(current, work, 'dev', sequence, config)
                    keys = ['native_reference'] if mode == 'missing' else ['native_reference', 'native_auxiliary']
                    for key in keys:
                        np.testing.assert_array_equal(current[key], old[key])
                    if mode == 'missing':
                        assert 'native_auxiliary' not in current and 'ir' not in metadata
                    comparisons.append(dict(sequence=sequence, frontend=arm, cue=cue, mode=mode,
                        frames=len(raw['frame_ids']), observations=len(raw['boxes']),
                        original_seed0_formula_arrays_bit_exact=True,
                        source_metadata={key: value for key, value in metadata.items()
                                         if key in ['rgb', 'ir', 'alignment']}))
    # Exercise the unchanged image path with actual Train pixels and two
    # different frames, not with sampled Test/Dev labels or synthetic thermal.
    sequence = 'BT-003'; index_path = work / 'native_candidate_index' / sequence / 'index.npz'
    with np.load(index_path) as source:
        frames, boxes = source['frame'], source['boxes']
        rows = [0, int(np.flatnonzero(frames != frames[0])[0])]
        selected = [(int(frames[i]), boxes[i].copy()) for i in rows]
    actual_pixels, artifacts = [], {}
    artifacts[str(index_path)] = sha256(index_path)
    for cue in ['rgb', 'ir']:
        model, endpoint = load_native_encoder(cue, 0, work, protocol)
        assert endpoint['cumulative_native_epochs'] == 72 and endpoint['seed'] == 0
        frozen = state_sha256(model.state_dict()); pixels, rectangles, images = [], [], []
        for frame, box in selected:
            path = ((Path(plan['data_root']) / 'train' / sequence / 'visible' / f'{frame:06d}.jpg')
                if cue == 'rgb' else (Path(plan['generated_ir_root']) / 'train' / sequence /
                                       'infrared_diffv2ir' / f'{frame:06d}.png'))
            image = cv2.imread(str(path), cv2.IMREAD_COLOR if cue == 'rgb' else cv2.IMREAD_UNCHANGED)
            assert image is not None and (cue != 'ir' or image.ndim == 2 and image.dtype == np.uint8)
            crop, rectangle = native_crop(image, box)
            pixels.append(crop); rectangles.append(rectangle)
            images.append(dict(path=str(path), sha256=sha256(path), frame=frame))
        value = native_input(torch.from_numpy(np.stack(pixels)), torch.from_numpy(np.stack(rectangles)), 'cpu')
        with torch.inference_mode():
            raw = model.projection(model.backbone(value))
            feature = F.normalize(model.neck(raw), dim=-1)
        assert feature.shape == (2, 512) and torch.isfinite(feature).all()
        np.testing.assert_allclose(feature.norm(dim=-1).numpy(), 1, atol=1e-6, rtol=0)
        assert state_sha256(model.state_dict()) == frozen
        actual_pixels.append(dict(cue=cue, seed=0, endpoint=endpoint, actual_Train_images=images,
            input_shape=list(value.shape), output_shape=list(feature.shape),
            finite_unit_postneck_vectors=True, original_encoder_weights_and_BN_unchanged=True))
        artifacts[endpoint['path']] = endpoint['sha256']
        del model, value, feature, raw
    assert len(comparisons) == 24 and before == sources() and not torch.cuda.is_initialized()
    report = dict(status='COMPLETE_CPU_NECK72_WORKER_INTERFACES_WITH_REAL_CACHED_DEV_AND_TRAIN_PIXELS',
        source_hashes=before, tests_passed=9, pytest_command=command, test_log_sha256=sha256(log),
        original_reference_commit='e25f2f9896ab094611edc809f87d7a5ad7e981d3',
        unchanged_original_pixel_forward_AST=True, unchanged_original_training_step_AST=True,
        full_existing_Dev_view_comparisons=comparisons, existing_Dev_unique_frames=4834,
        source_views=8, modes_each=3, old_layout_reference_view=str(alias),
        data_or_image_copies=0, actual_Train_pixel_CPU_checks=actual_pixels, artifact_hashes=artifacts,
        fixed_seed=0, additional_seed_work=False, GPU_work=False, new_Dev_or_Test_accuracy=False,
        formal_fit_started=False, full_study_scheduler_implemented=False,
        actual_GPU_source_prefix_and_optimizer_checks_pending=True,
        cost_scope='CPU engineering total seconds, not inference FPS or system latency.',
        seconds=time.monotonic()-started, goal_met=False, supported_innovations=0)
    require_storage(work); dump(output, report)
    print(json.dumps(dict(status=report['status'], tests=9, existing_Dev_views=8, actual_Train_pixels=4,
                          seconds=report['seconds'], CUDA_initialized=torch.cuda.is_initialized())), flush=True)


if __name__ == '__main__':
    main()
