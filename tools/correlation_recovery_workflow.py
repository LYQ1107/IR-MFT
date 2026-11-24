"""Frozen CGR context, hardware release and provenance for the formal backfill."""
import json
import os
from pathlib import Path

from tools.run_pseudo_ir_comparison import sha256
from tools.structural_legacy_backfill import pid_finished

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'configs/perceptual_correlation_recovery_backfill_20261124.json'
WORKFLOW = ROOT / 'configs/perceptual_correlation_recovery_workflow_20261124.json'
DOCS = ROOT / 'docs/fusion_research'


def read(path):
    return json.loads(Path(path).read_text())


def context():
    spec = read(SPEC)
    plan_path = ROOT / 'configs/perceptual_mft_20261122.json'
    assert sha256(plan_path) == spec['source_plan_sha256']
    plan = read(plan_path)
    protocol_path = ROOT / plan['protocol']
    assert sha256(protocol_path) == 'fe3db8e991facfcd74981f76aa5c2271c18a9554386501eb2c2dd30a12bf6c60'
    policy_path = ROOT / 'configs/perceptual_single_seed_policy_20261123.json'
    assert sha256(policy_path) == spec['single_seed_policy_sha256']
    assert read(policy_path)['fixed_seed'] == 0 and not read(policy_path)['additional_seed_inference']
    repair_path = ROOT / 'configs/perceptual_correlation_recoverability_20261123.json'
    assert sha256(repair_path) == spec['repair_plan_sha256']
    parent_path = ROOT / 'configs/perceptual_correlation_repair_20261123.json'
    assert sha256(parent_path) == read(repair_path)['parent_sha256']
    parent = read(parent_path)
    assert parent['architecture']['latent_channels'] == 32 and parent['architecture']['local_radius'] == 2
    assert parent['architecture']['temperature'] == .1 and parent['architecture']['maximum_residual_RMS_fraction'] == .5
    work = Path(plan['work_root'])
    calibration_path = work / 'correlation_cache/calibration.json'
    calibration = read(calibration_path)
    assert calibration['config_sha256'] == sha256(parent_path)
    assert calibration['Dev_rows'] == calibration['Test_rows'] == 0
    return plan, spec, read(protocol_path), parent, calibration, work


def registered():
    workflow = read(WORKFLOW)
    assert workflow['source_specification_sha256'] == sha256(SPEC)
    assert all(sha256(ROOT / p) == h for p, h in workflow['source_hashes'].items())
    assert all(sha256(p) == h for p, h in workflow['artifact_sources'].items())
    assert workflow['cases'] == read(SPEC)['cases'] and len(workflow['cases']) == 12
    return workflow


def native_predecessor_ready(state, report, finished=pid_finished):
    """A live predecessor or an incomplete export cannot release these GPUs."""
    if 'FAIL' in state.get('status', ''):
        raise RuntimeError('Preserve failed native predecessor before CGR GPU work')
    if state.get('status') != 'COMPLETE_NATIVE_STUDY_REQUIRES_SCIENTIFIC_ASSESSMENT':
        return False
    if report.get('status') != 'COMPLETE_NATIVE_PYRAMID_FORMAL_STUDY_REQUIRES_RESEARCH_AUDIT':
        return False
    if not state.get('controller_pid') or not finished(state['controller_pid']):
        return False
    phases = state.get('phases', {})
    if not phases or any(not rows or any(v.get('status') != 'COMPLETE' for v in rows.values())
                         for rows in phases.values()):
        return False
    return all(row.get('pid') and finished(row['pid']) for rows in phases.values() for row in rows.values())


def predecessor_ready(work):
    from tools.native_pyramid_experiment import predecessor_ready as prior_chain_ready
    if not prior_chain_ready(work):
        return False
    path = Path(work) / 'native_pyramid_study_state.json'
    report_path = DOCS / 'PERCEPTUAL_MFT_20261124_NATIVE_PYRAMID_FORMAL.json'
    if not path.exists() or not report_path.exists():
        return False
    state, report = read(path), read(report_path)
    # Exact report name is pinned by the actual native exporter, not inferred
    # from a queue entry or an accuracy value.
    if report.get('study_sha256') != state.get('study_sha256'):
        raise RuntimeError('Native completion report and actual state disagree')
    return native_predecessor_ready(state, report)


def gpu_settings():
    import subprocess
    import torch
    physical = os.environ.get('CUDA_VISIBLE_DEVICES')
    assert physical in {'4', '5'} and torch.cuda.device_count() == 1
    policy = read(ROOT / 'configs/perceptual_gpu_handoff_20261123.json')
    uuid = subprocess.check_output(['nvidia-smi', '-i', physical, '--query-gpu=uuid',
        '--format=csv,noheader'], text=True).strip()
    assert uuid == policy['future_gpu_uuids'][physical]
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.45, 0)
    return physical, uuid


def packet(path):
    import numpy as np
    with np.load(path) as saved:
        return {k: saved[k] for k in ['frame_ids', 'offsets', 'boxes', 'scores']}


def frame_packet(data, frame):
    import numpy as np
    lo, hi = data['offsets'][frame-1:frame+1]
    return np.c_[data['boxes'][lo:hi], data['scores'][lo:hi]].astype(np.float32)


def require_probes(work):
    for physical in ['4', '5']:
        proof = read(Path(work) / 'correlation_recovery_formal/engineering' / ('gpu'+physical) / 'complete.json')
        assert proof['status'] == 'COMPLETE_CGR_ACTUAL_GPU_SOURCE_ZERO_MISSING_AND_ORIGINAL_PREFIX'
        assert proof['workflow_sha256'] == sha256(WORKFLOW) and proof['physical_gpu'] == physical
        assert proof['frames_each'] == 8 and proof['original_Dev_cases_checked'] == 8
        assert proof['frozen_RGB_weights_and_BN_unchanged'] and proof['source_missing_and_zero_full_head_NMS_exact']
        assert proof['zero_branches_each'] == 6 and not proof['GT_read']
        assert set(proof['sequences']) == {'BT-005', 'SN-013'}
        for item in proof['sequences'].values():
            assert sha256(item['path']) == item['sha256']
            data = read(item['path'])
            assert data['status'] == 'COMPLETE_CGR_GPU_PROBE' and data['frames'] == 8
            assert data['original_normal_paired_Dev_prediction_streams_exact'] == 8


def audit_source_frame(current, original, frame, total):
    """Reuse the already registered CGA final-batch numeric source bounds."""
    import numpy as np
    from tools.recover_correlation_formal_test import aligned_source_audit
    if np.array_equal(current, original):
        return None
    assert frame > total-4, 'Unexpected RGB source drift outside the final batch'
    def one(array):
        return dict(frame_ids=np.array([1]), offsets=np.array([0, len(array)]),
                    boxes=array[:, :4], scores=array[:, 4])
    result = aligned_source_audit(one(current), one(original))
    assert len(result['changed_tail_frames']) == 1
    change = result['changed_tail_frames'][0]
    change['frame'] = frame
    return change
