"""Frozen context, explicit arm configurations and resource/provenance gates."""
import copy
import json
import os
from pathlib import Path

from tools.run_pseudo_ir_comparison import dump, sha256
from tools.structural_legacy_backfill import pid_finished, require_predecessor

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'configs/perceptual_native_pyramid_study_20261124.json'
FIT_CONFIGS = ROOT / 'configs/perceptual_native_pyramid_fit_cases_20261124'
PIPELINE_REVISION = 2
CANONICAL_PROOF = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261124_NATIVE_CANONICAL_DEV_ENGINEERING_V2.json'


def read(path):
    return json.loads(Path(path).read_text())


def frozen_json(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen native artifact differs: ' + str(path))
    else:
        dump(path, value)
    return path


def context():
    study = read(STUDY)
    plan_path = ROOT / 'configs/perceptual_mft_20261122.json'
    assert sha256(plan_path) == study['main_plan_sha256']
    plan = read(plan_path)
    protocol_path = ROOT / plan['protocol']
    assert sha256(protocol_path) == study['protocol_sha256']
    protocol = read(protocol_path)
    policy_path = ROOT / 'configs/perceptual_single_seed_policy_20261123.json'
    assert sha256(policy_path) == study['single_seed_policy_sha256']
    assert read(policy_path)['fixed_seed'] == study['fixed_seed'] == 0
    work = Path(plan['work_root'])
    calibration_path = work / 'correlation_cache/calibration.json'
    assert sha256(calibration_path) == study['calibration_sha256']
    calibration = read(calibration_path)
    assert calibration['Dev_rows'] == calibration['Test_rows'] == 0
    assert sha256(work / study['Train_length_source']) == study['Train_length_source_sha256']
    assert read(work / study['Train_length_source'])['lengths'] == study['Train_delivery']['lengths']
    parent_path = ROOT / 'configs/perceptual_correlation_repair_20261123.json'
    parent = read(parent_path)
    assert sha256(parent['source_RGB_checkpoint']) == study['source_RGB_sha256'] == parent['source_RGB_sha256']
    return plan, study, protocol, calibration, work, Path(parent['source_RGB_checkpoint'])


def make_arm(study, fit_id, mode, cue, removal=None):
    if mode not in study['modes'] or cue not in study['matched_cues'] or (removal and cue != 'ir'):
        raise ValueError('Only registered matched arms/removals are allowed')
    config = copy.deepcopy(study)
    kwargs = dict(mode=mode, bound=study['architecture']['feature_residual_bound'], spectral=True,
                  use_quality=True, protect_rgb=True, use_memory=True, use_motion=True)
    if removal:
        allowed = study['removals']['all_modes'] + (study['removals']['temporal_only'] if mode == 'temporal' else [])
        if removal not in allowed:
            raise ValueError('Unregistered removal')
        if removal == 'no_spectral_contrast': kwargs['spectral'] = False
        elif removal == 'no_fusion_quality_gate': kwargs['use_quality'] = False
        elif removal == 'fixed_original_RGB_geometry': config['geometry_fixed'] = True
        elif removal == 'no_RGB_protection':
            kwargs['protect_rgb'] = False
            config['fit']['lambda_reliable_RGB'] = 0.
        elif removal == 'no_temporal_memory': kwargs['use_memory'] = False
        elif removal == 'no_motion_alignment': kwargs['use_motion'] = False
    config.update(fit_id=fit_id, mode=mode, cue=cue, removal=removal,
                  parent_study_sha256=sha256(STUDY), module_kwargs=kwargs)
    return config


def registry(work):
    path = Path(work) / 'native_pyramid_pipeline_registered.json'
    data = read(path)
    assert data['status'] == 'COMPLETE_REGISTERED_NATIVE_DEV_TEST_PIPELINE_PENDING_GPU_ENGINEERING'
    assert data['pipeline_revision'] == PIPELINE_REVISION
    assert data['study_sha256'] == sha256(STUDY)
    assert all(sha256(ROOT / f) == h for f, h in data['source_hashes'].items())
    assert all(sha256(f) == h for f, h in data['artifact_hashes'].items())
    return data


def predecessor_ready(work):
    """Complete exports and actual process exits, not a stale queue declaration."""
    work = Path(work)
    if not require_predecessor(work):
        return False
    path = work / 'structural_legacy_backfill_state.json'
    if not path.exists(): return False
    legacy = read(path)
    if 'FAIL' in legacy.get('status', ''):
        raise RuntimeError('Preserve failed legacy structural backfill before using GPUs')
    if legacy.get('status') != 'COMPLETE_ALL9_FORMAL_BACKFILL_CASES': return False
    report = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261124_STRUCTURAL_LEGACY_BACKFILL_TEST.json'
    if not report.exists() or read(report).get('status') != 'COMPLETE_INITIAL_STRUCTURAL_FORMAL_ALL7_TEST_BACKFILL':
        return False
    if len(legacy.get('jobs', {})) != 9 or not all(j.get('status') == 'COMPLETE' for j in legacy['jobs'].values()):
        return False
    producers = list(legacy.get('producers', {}).values())
    if len(producers) != 7 or not all(j.get('status') == 'COMPLETE' for j in producers): return False
    pids = [legacy.get('controller_pid'), *(j.get('pid') for j in producers)]
    return bool(all(pids) and all(pid_finished(pid) for pid in pids))


def gpu_settings():
    """Called by a worker only after the verified predecessor releases both cards."""
    import cv2
    import torch
    physical = os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in ['4', '5'] or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Use only physical GPU4/5 and the registered CuBLAS setting')
    expected = read(ROOT / 'configs/perceptual_gpu_handoff_20261123.json')['future_gpu_uuids'][physical]
    assert torch.cuda.device_count() == 1
    actual = str(torch.cuda.get_device_properties(0).uuid)
    assert actual.lower().replace('gpu-', '') == expected.lower().replace('gpu-', ''), (physical, actual)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.4, 0)
    return physical, actual


def load_teacher(checkpoint, training):
    import torch
    from yolox.exp import get_exp
    with torch.random.fork_rng(devices=[]):
        model = get_exp(str(ROOT / 'exps/yolox_mft26ir_visible.py'), None).get_model()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    model = model.cuda().eval().requires_grad_(False)
    model.backbone.half()
    if training: model.head.float()
    else: model.head.half()
    return model
