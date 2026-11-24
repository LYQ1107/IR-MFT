import copy
import hashlib
import json

import numpy as np
import pytest

from tools import correlation_recovery_workflow as workflow
from tools import launch_correlation_recovery_workflow as launch


def test_native_release_requires_export_and_all_original_process_exits():
    state = dict(status='COMPLETE_NATIVE_STUDY_REQUIRES_SCIENTIFIC_ASSESSMENT', controller_pid=1,
        phases=dict(probe=dict(a=dict(status='COMPLETE', pid=2)),
                    Test=dict(b=dict(status='COMPLETE', pid=3))))
    report = dict(status='COMPLETE_NATIVE_PYRAMID_FORMAL_STUDY_REQUIRES_RESEARCH_AUDIT')
    assert workflow.native_predecessor_ready(state, report, finished=lambda pid: True)
    assert not workflow.native_predecessor_ready(state, report, finished=lambda pid: pid != 1)
    assert not workflow.native_predecessor_ready(state, report, finished=lambda pid: pid != 3)
    assert not workflow.native_predecessor_ready(state, {}, finished=lambda pid: True)
    pending = copy.deepcopy(state)
    pending['phases']['Test']['b']['status'] = 'RUNNING'
    assert not workflow.native_predecessor_ready(pending, report, finished=lambda pid: True)
    failed = dict(state, status='FAILED_PRESERVE_NATIVE_STUDY')
    with pytest.raises(RuntimeError):
        workflow.native_predecessor_ready(failed, report, finished=lambda pid: True)


def test_source_numeric_policy_keeps_tail_and_threshold_requirements():
    original = np.array([[10, 10, 30, 30, .5]], np.float32)
    assert workflow.audit_source_frame(original, original.copy(), 1, 100) is None
    small = original.copy()
    small[0, 0] += .5
    small[0, 4] += .001
    change = workflow.audit_source_frame(small, original, 100, 100)
    assert change['frame'] == 100 and change['matched_strict_and_nonstrict_06_07_memberships_exact']
    with pytest.raises(AssertionError):
        workflow.audit_source_frame(small, original, 96, 100)
    threshold = original.copy()
    threshold[0, 4] = .599
    crossed = threshold.copy()
    crossed[0, 4] = .601
    with pytest.raises(AssertionError):
        workflow.audit_source_frame(crossed, threshold, 100, 100)
    weak = np.r_[original, np.array([[100, 100, 120, 120, .2]], np.float32)]
    assert len(workflow.audit_source_frame(weak, original, 100, 100)['unmatched_current']) == 1
    strong = weak.copy()
    strong[1, 4] = .8
    with pytest.raises(AssertionError):
        workflow.audit_source_frame(strong, original, 100, 100)


def test_false_child_evidence_is_failed_not_complete(tmp_path, monkeypatch):
    class FinishedProcess:
        pid = 111
        def poll(self):
            return 0
    monkeypatch.setattr(launch, 'registered', lambda: None)
    monkeypatch.setattr(launch.subprocess, 'Popen', lambda *args, **kwargs: FinishedProcess())
    monkeypatch.setattr(launch.time, 'sleep', lambda duration: None)
    state = dict(data_root=str(tmp_path), phases={}, preserved_active_children={})
    queue = launch.Queue(tmp_path, state)
    with pytest.raises(RuntimeError, match='Required CGR child evidence failed'):
        queue.jobs('CHECK', [dict(id='child', command=['unused'])], False, lambda row: False)
    assert state['phases']['CHECK']['child']['exit_code'] == 0
    assert state['phases']['CHECK']['child']['status'] == 'FAILED_EVIDENCE'


def test_both_real_gpu_proof_files_and_prefix_details_are_required(tmp_path, monkeypatch):
    frozen = tmp_path / 'workflow.json'
    frozen.write_text('{}')
    monkeypatch.setattr(workflow, 'WORKFLOW', frozen)
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    for gpu in ['4', '5']:
        folder = tmp_path / 'correlation_recovery_formal/engineering' / ('gpu'+gpu)
        folder.mkdir(parents=True)
        parts = {}
        for seq in ['BT-005', 'SN-013']:
            path = folder / (seq+'.json')
            path.write_text(json.dumps(dict(status='COMPLETE_CGR_GPU_PROBE', frames=8,
                original_normal_paired_Dev_prediction_streams_exact=8)))
            parts[seq] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        proof = dict(status='COMPLETE_CGR_ACTUAL_GPU_SOURCE_ZERO_MISSING_AND_ORIGINAL_PREFIX',
            workflow_sha256=digest, physical_gpu=gpu, frames_each=8, original_Dev_cases_checked=8,
            frozen_RGB_weights_and_BN_unchanged=True, source_missing_and_zero_full_head_NMS_exact=True,
            zero_branches_each=6, GT_read=False, sequences=parts)
        (folder / 'complete.json').write_text(json.dumps(proof))
    workflow.require_probes(tmp_path)
    path = tmp_path / 'correlation_recovery_formal/engineering/gpu5/complete.json'
    value = json.loads(path.read_text())
    value['zero_branches_each'] = 0
    path.write_text(json.dumps(value))
    with pytest.raises(AssertionError):
        workflow.require_probes(tmp_path)
