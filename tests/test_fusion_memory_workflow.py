"""Synthetic workflow/release fixtures, not actual qualification or GPU work."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from test_fusion_memory_evaluation import complete_fixture, official
from tools.fusion_memory_dev_freeze import final_dev_freeze
from tools.fusion_memory_evaluation import read, sha
from tools.fusion_memory_observation import write_json
from tools.fusion_memory_workflow import (FusionMemoryWorkflow, complete_phase_cases, release_header, release_state_ready)


def release():
    identity = dict(pid=123, start_ticks=1, command_sha256='c'*64)
    job = dict(pid=456, start_ticks=2, status='COMPLETE', exit_code=0, command=['artificial_worker'])
    state = dict(status='COMPLETE_SPATIAL_STUDY_REQUIRES_SCIENTIFIC_ASSESSMENT',
        controller_pid=123, study_sha256='a'*64, registry_sha256='b'*64, phases={'ARTIFICIAL': {'worker': job}})
    header = dict(status='COMPLETE_NINE_ARM_SPATIAL_STUDY_REQUIRES_SCIENTIFIC_ASSESSMENT',
                  study_sha256='a'*64, registry_sha256='b'*64)
    return state, identity, header


def test_release_header_stops_before_unparseable_test_score_tail(tmp_path):
    path = tmp_path/'metadata_and_forbidden_tail.json'
    path.write_text('{\n"status": "COMPLETE",\n"study_sha256": "a",\n"registry_sha256": "b",\nFORBIDDEN_TEST_SCORES_NOT_JSON\n')
    assert release_header(path) == dict(status='COMPLETE', study_sha256='a', registry_sha256='b')
    with pytest.raises(json.JSONDecodeError):
        read(path)


def test_unexpected_field_before_safe_header_is_rejected(tmp_path):
    path = tmp_path/'bad.json'
    path.write_text('{\n"primary_Test47": {},\n"status": "COMPLETE"\n}')
    with pytest.raises(ValueError, match='Unexpected field'):
        release_header(path)


def test_release_needs_both_controller_and_all_original_children_to_exit():
    state, identity, header = release()
    assert release_state_ready(state, identity, header, lambda *a, **kw: True)
    assert not release_state_ready(state, identity, header, lambda pid, **kw: pid != 123)
    assert not release_state_ready(state, identity, header, lambda pid, **kw: pid != 456)


@pytest.mark.parametrize('change', ['pending', 'no_phases', 'empty_phase', 'nonzero', 'no_identity', 'running_child'])
def test_partial_predecessor_cannot_release(change):
    state, identity, header = release()
    job = state['phases']['ARTIFICIAL']['worker']
    if change == 'pending':
        state['status'] = 'WAITING_FOR_PRIOR_CHAIN'
    elif change == 'no_phases':
        state['phases'] = {}
    elif change == 'empty_phase':
        state['phases']['ARTIFICIAL'] = {}
    elif change == 'nonzero':
        job['exit_code'] = 1
    elif change == 'no_identity':
        job.pop('start_ticks')
    else:
        job['status'] = 'RUNNING'
    assert not release_state_ready(state, identity, header, lambda *a, **kw: True)


@pytest.mark.parametrize('change', ['version', 'controller', 'failed'])
def test_mismatched_or_failed_predecessor_is_not_silently_restarted(change):
    state, identity, header = release()
    if change == 'version':
        header['registry_sha256'] = 'd'*64
    elif change == 'controller':
        state['controller_pid'] = 999
    else:
        state['status'] = 'FAILED_PRESERVE_SPATIAL_STUDY'
    with pytest.raises(ValueError):
        release_state_ready(state, identity, header, lambda *a, **kw: True)


def test_missing_full_entrypoints_rejected_before_any_registration_is_trusted(tmp_path):
    value = FusionMemoryWorkflow.__new__(FusionMemoryWorkflow)
    value.config = {'complete_pipeline_entrypoints': ['DELIBERATELY_MISSING_F4_COMPONENT.py']}
    value.base = tmp_path
    write_json(tmp_path/'pipeline_registered.json', {'status': 'pretend_complete'})
    with pytest.raises(ValueError, match='Complete F4 production'):
        value.registered()


def test_live_predecessor_blocks_device_configuration():
    value = FusionMemoryWorkflow.__new__(FusionMemoryWorkflow)
    value.validate_specification = lambda path: ({}, {})
    value.predecessor_ready = lambda: False
    with patch('tools.native_pyramid_experiment.gpu_settings', side_effect=AssertionError('CUDA attempted')) as configure:
        with pytest.raises(ValueError, match='all original processes'):
            value.gpu_settings('probe', 'artificial_spec')
    configure.assert_not_called()


def test_unproven_gpu_engineering_blocks_production_before_cuda(tmp_path):
    value = FusionMemoryWorkflow.__new__(FusionMemoryWorkflow)
    value.base = tmp_path
    value.validate_specification = lambda path: ({}, {})
    value.predecessor_ready = lambda: True
    with patch('tools.native_pyramid_experiment.gpu_settings', side_effect=AssertionError('CUDA attempted')) as configure:
        with pytest.raises(ValueError, match='Actual GPU F4 engineering'):
            value.gpu_settings('produce', 'artificial_spec')
    configure.assert_not_called()


def test_phase_sets_recomputed_from_complete_dev_files_without_test_accuracy(tmp_path, official):
    plan, normal_files, additional_files, old = complete_fixture(tmp_path, official)
    normal = tmp_path/'normal_Dev_selection.json'
    old.rename(normal)
    frozen = tmp_path/'frozen_Test_case_plan.json'
    final_dev_freeze(plan, normal, normal_files, additional_files, official[0], frozen)
    cases, basis = complete_phase_cases(plan, tmp_path, 'dev_normal', official[0])
    assert cases == plan['normal_cases'] and basis == {}
    additional, basis = complete_phase_cases(plan, tmp_path, 'dev_additional', official[0])
    assert len(additional) == 30 and basis['normal_Dev_choice_sha256'] == sha(normal)
    cases, basis = complete_phase_cases(plan, tmp_path, 'test_frozen', official[0])
    assert len(cases) == 78 and cases == read(frozen)['cases']
    assert basis['complete_Dev_Test_choice_sha256'] == sha(frozen)
    changed = read(frozen)
    changed['cases'].pop()
    write_json(frozen, changed)
    with pytest.raises(ValueError, match='differs from complete Dev files'):
        complete_phase_cases(plan, tmp_path, 'test_frozen', official[0])


def test_original_protocol_gt_bytes_checked_in_evaluation_context(tmp_path):
    value = FusionMemoryWorkflow.__new__(FusionMemoryWorkflow)
    value.plan = {'evaluation': {'GT_root': str(tmp_path)}}
    value.protocol = {'derived_gt_sha256': {}}
    for seq in ['BT-005', 'SN-013']:
        path = tmp_path/'train'/seq/'gt/gt.txt'
        path.parent.mkdir(parents=True)
        path.write_text('ARTIFICIAL_NOT_RESEARCH\n')
        value.protocol['derived_gt_sha256']['train/'+seq] = sha(path)
    result = value.gt_hashes('dev')
    assert len(result) == 2
    (tmp_path/'train/BT-005/gt/gt.txt').write_text('CHANGED')
    with pytest.raises(ValueError, match='ProtocolV2 GT bytes changed'):
        value.gt_hashes('dev')
