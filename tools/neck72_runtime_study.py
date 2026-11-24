"""Independent source registration and release gates for fresh-image timing."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from tools.neck72_experiment import context, STUDY
from tools.neck72_runtime import read, require_host_memory, require_storage
from tools.run_pseudo_ir_comparison import sha256

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'configs/perceptual_neck72_runtime_20261124.json'
REGISTRY = 'neck72_fresh_runtime_registered.json'
REPORT = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261124_NECK72_SELECTED_TEST.json'
TERMINAL = 'COMPLETE_NECK72_ALL_SELECTED_TEST_PENDING_FULL_SYSTEM_COST_AUDIT'


def sources():
    _, _, work = context()
    parent = read(work / 'neck72_study_registered.json')
    files = set(parent['source_hashes'])
    files.update(['tools/neck72_online_inputs.py', 'tools/neck72_runtime_study.py',
        'tools/benchmark_neck72_runtime.py', 'tools/launch_neck72_runtime.py',
        'tools/probe_neck72_online_inputs.py', 'tests/test_neck72_online_inputs.py',
        str(SPEC.relative_to(ROOT)), 'tools/run_pseudo_ir_comparison.py',
        'tools/native_cue_io.py', 'yolox/data/data_augment.py',
        'yolox/data/data_augment_rgbt.py', 'yolox/data/datasets/mot_rgbt.py',
        'yolox/exp/__init__.py', 'yolox/exp/build.py', 'yolox/exp/yolox_base.py',
        'yolox/utils/boxes.py'])
    # Pin the detector architecture/preprocessing dependencies; do not edit the
    # already-live accuracy registrations to add a new independent workflow.
    files.update(str(p.relative_to(ROOT)) for p in (ROOT / 'yolox/models').glob('*.py'))
    files.update('exps/'+name+'.py' for name in ['yolox_mft26ir_visible',
        'yolox_mft26ir_pseudo_ir_comparison', 'yolox_mft26ir_rgbt_v1',
        'yolox_mft26ir_rgbt_v2', 'yolox_mft26ir_rgbt_v3'])
    return sorted(files)


def registered(verify_artifacts=False):
    plan, protocol, work = context()
    config, registration = read(SPEC), read(work / REGISTRY)
    assert registration['specification_sha256'] == sha256(SPEC)
    assert config['accuracy_study_sha256'] == sha256(STUDY)
    assert config['accuracy_registration_sha256'] == sha256(work / 'neck72_study_registered.json')
    assert registration['source_hashes'] == {p: sha256(ROOT / p) for p in sources()}
    if verify_artifacts:
        assert all(sha256(p) == h for p, h in registration['artifact_hashes'].items())
    return plan, protocol, work, config, registration


def exited(pid, command=None, identity=None):
    """A reused unrelated PID is not the original process still running."""
    path = Path('/proc') / str(pid)
    try:
        stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return True
        if identity and int(stat[19]) != identity['start_ticks']:
            return True
        if command and (path / 'cmdline').read_bytes().split(b'\0')[:-1] != [os.fsencode(v) for v in command]:
            return True
        return False
    except FileNotFoundError:
        return True


def cases(work):
    frozen = read(Path(work) / 'neck72_Test_selection_frozen.json')
    assert frozen['status'] == 'FROZEN_NECK72_TEST_AFTER_ALL_COMPLETE_SELECTED_DEV'
    assert frozen['study_sha256'] == sha256(STUDY)
    assert frozen['complete_selected_Dev_before_Test'] and not frozen['Test_accuracy_read_for_selection']
    assert len(frozen['cases']) >= 47
    assert len({c['id'] for c in frozen['cases']}) == len(frozen['cases'])
    return frozen


def predecessor_ready(work):
    work = Path(work)
    path = work / 'neck72_study_state.json'
    if not path.exists():
        return False
    state = read(path)
    if 'FAIL' in state['status']:
        raise RuntimeError('Preserve the failed neck72 accuracy predecessor before runtime work')
    if state['status'] != TERMINAL or not REPORT.exists():
        return False
    report = read(REPORT)
    if report['status'] != 'COMPLETE_NECK72_SELECTED_ALL7TEST_REQUIRES_FULL_SYSTEM_COST_AUDIT':
        return False
    assert state['study_sha256'] == report['study_sha256'] == sha256(STUDY)
    identity_path = work / 'neck72_resource_guard_R2_controller_identity.json'
    identity = read(identity_path)
    assert identity['pid'] == state['controller_pid']
    if not exited(state['controller_pid'], identity=identity):
        return False
    if not state['phases']:
        return False
    for jobs in state['phases'].values():
        if not jobs:
            return False
        for job in jobs.values():
            if job.get('status') != 'COMPLETE' or job.get('exit_code') != 0:
                return False
            if not exited(job['pid'], command=job['command']):
                return False
    frozen = cases(work)
    for case in frozen['cases']:
        for split, frames in [('dev', 4834), ('test', 15985)]:
            p = work / ('neck72_'+split) / case['id'] / 'inference_manifest.json'
            value = read(p)
            assert value['status'] == 'INFERENCE_COMPLETE_BEFORE_GT_EVALUATION'
            assert value['config'] == case['config'] and value['frames'] == frames
    return True


def gpu_settings(work):
    physical = os.environ.get('CUDA_VISIBLE_DEVICES')
    if physical not in {'4', '5'}:
        raise ValueError('Fresh neck72 runtime permits physicalGPU4/5 only')
    if not predecessor_ready(work):
        raise RuntimeError('Wait for complete neck72 accuracy export and actual process exit')
    require_host_memory(8*2**30, maximum_workers=1); require_storage(work)
    policy = read(ROOT / 'configs/perceptual_gpu_handoff_20261123.json')
    uuid = subprocess.check_output(['nvidia-smi', '-i', physical, '--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True).strip()
    assert uuid == policy['future_gpu_uuids'][physical]
    import cv2
    import torch
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.45, 0)
    return physical, uuid


def compare_array(actual, expected, tolerance):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return dict(passed=False, shape=list(actual.shape), expected_shape=list(expected.shape),
                    maximum_absolute_difference=None, tolerance=tolerance)
    finite = np.isfinite(actual).all() and np.isfinite(expected).all()
    maximum = float(np.max(abs(actual.astype(np.float64)-expected.astype(np.float64)), initial=0)) if finite else None
    return dict(passed=bool(finite and maximum <= tolerance), shape=list(actual.shape),
        expected_shape=list(expected.shape), maximum_absolute_difference=maximum,
        bit_exact=bool(np.array_equal(actual, expected)), tolerance=tolerance)


def frame_comparison(frame, data, reference, config, limits):
    a, b = map(int, data['offsets'][frame-1:frame+1])
    c, d = map(int, reference['offsets'][frame-1:frame+1])
    fields = dict(boxes=limits['boxes_max_abs_pixels'], scores=limits['scores_max_abs'])
    result = {}
    if config['backend'] != 'byte':
        fields.update(embedding=limits['consumed_FP16_features_max_abs'],
            rgb_mask_embedding=limits['consumed_FP16_features_max_abs'],
            native_reference=limits['native_FP32_vectors_max_abs'], rgb_hog=0, color=0)
        if not config.get('disable_native_increment', False):
            fields['native_auxiliary'] = limits['native_FP32_vectors_max_abs']
    for name, tolerance in fields.items():
        result[name] = compare_array(data[name][a:b], reference[name][c:d], tolerance)
    if config['backend'] != 'byte':
        # Only RGB quality is consumed by the ordinary core; native IR adds its
        # original scalar quality. Unused NaN columns are never compared/read.
        columns = [0, 1] if config['native_cue'] == 'ir' and not config.get('disable_native_increment', False) else [0]
        result['quality'] = compare_array(data['quality'][a:b][:, columns], reference['quality'][c:d][:, columns], 0)
        point, old = data['foreground_transport'], reference['foreground_transport']
        result['point_RGB_quality'] = compare_array(point['quality'][a:b, 0], old['quality'][c:d, 0], 0)
        for lag in [1, 2, 4, 8]:
            for kind, names in [('vote', [f'vote_{lag}_{s}_{t}' for s in ['bbox', 'rgb'] for t in ['bbox', 'rgb']]),
                ('count', [f'{k}_{lag}_{s}' for k in ['good', 'total'] for s in ['bbox', 'rgb']])]:
                i, j = map(int, point[f'{kind}_offsets_{lag}'][frame-1:frame+1])
                k, l = map(int, old[f'{kind}_offsets_{lag}'][frame-1:frame+1])
                for name in names:
                    result[name] = compare_array(point[name][i:j], old[name][k:l], 0)
    return dict(passed=all(v['passed'] for v in result.values()), frame=frame, fields=result)


def comparison_summary(records):
    """Retain every field's worst error and every failing frame, without arrays."""
    fields = {}
    for row in records:
        for name, result in row['fields'].items():
            target = fields.setdefault(name, dict(frames_checked=0, failed_frames=[],
                shape_mismatches=[], maximum_absolute_difference=0., bit_exact_all=True,
                tolerance=result['tolerance']))
            target['frames_checked'] += 1
            target['bit_exact_all'] &= result.get('bit_exact', False)
            if not result['passed']:
                target['failed_frames'].append(row['frame'])
            if result['shape'] != result['expected_shape']:
                target['shape_mismatches'].append(dict(frame=row['frame'], actual=result['shape'], expected=result['expected_shape']))
            difference = result['maximum_absolute_difference']
            if difference is not None:
                target['maximum_absolute_difference'] = max(target['maximum_absolute_difference'], difference)
            else:
                target.setdefault('non_comparable_frames', []).append(row['frame'])
    return dict(frames_checked=len(records), all_passed=all(row['passed'] for row in records), fields=fields)
