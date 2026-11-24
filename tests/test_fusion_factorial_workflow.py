import copy
import json
from pathlib import Path

import pytest
import torch

from tools.fusion_factorial_workflow import (
    REGISTRY, STUDY, make_joint_arm, release_state_ready, require_gpu, schedule_sha256,
    validate_history,
)
from tools.native_pyramid_train_data import schedule_proof
from tools.run_pseudo_ir_comparison import sha256


def study():
    return json.loads(STUDY.read_text())


def release_fixture():
    job=dict(status='COMPLETE',exit_code=0,source_equivalence_passed=True,start_ticks=10,
             command=['python','worker'],pid=8)
    state=dict(status='COMPLETE_NECK72_FULL_COMPONENT_PREFIX_RUNTIME_SOURCE_MATCHED',
        controller_pid=7,phases=dict(dev={'A':dict(job)},test={'A':dict(job,pid=9)}))
    identity=dict(pid=7,start_ticks=5)
    report=dict(status='COMPLETE_NECK72_FRESH_IMAGE_TEST_COST_MEASUREMENTS',all_frozen_cases_passed=True,
        rows=[dict(id='A',source_equivalence_passed=True,valid_full_component_cost_for_preserved_accuracy=True)])
    return state,identity,report


def test_completed_files_cannot_release_a_still_live_controller_or_child():
    state,identity,report=release_fixture()
    assert release_state_ready(state,identity,report,['A'],lambda pid,**kw: True)
    for alive in [7,8,9]:
        assert not release_state_ready(state,identity,report,['A'],lambda pid,**kw: pid!=alive)


@pytest.mark.parametrize('damage',['phase','case','exit','source','ticks','report_source','report_cost','duplicate'])
def test_incomplete_or_failed_runtime_evidence_keeps_both_gpus_unreleased(damage):
    state,identity,report=release_fixture()
    if damage=='phase': state['phases'].pop('test')
    elif damage=='case': state['phases']['test'].clear()
    elif damage=='exit': state['phases']['test']['A']['exit_code']=None
    elif damage=='source': state['phases']['test']['A']['source_equivalence_passed']=False
    elif damage=='ticks': state['phases']['test']['A'].pop('start_ticks')
    elif damage=='report_source': report['rows'][0]['source_equivalence_passed']=False
    elif damage=='report_cost': report['rows'][0]['valid_full_component_cost_for_preserved_accuracy']=False
    elif damage=='duplicate': report['rows'].append(copy.deepcopy(report['rows'][0]))
    assert not release_state_ready(state,identity,report,['A'],lambda pid,**kw:True)


@pytest.mark.parametrize('status',['FAILED_PRESERVE_NECK72_FRESH_RUNTIME','NECK72_RUNTIME_DEV_MISMATCH_PRESERVE_AND_DEBUG'])
def test_source_mismatch_is_preserved_instead_of_bypassed(status):
    state,identity,report=release_fixture();state['status']=status
    with pytest.raises(RuntimeError,match='Preserve'):
        release_state_ready(state,identity,report,['A'],lambda pid,**kw:True)


def test_joint_arm_changes_only_the_registered_encoder_plane_and_fusion_gate_switches():
    parent=study();before=copy.deepcopy(parent)
    for mode in parent['modes']:
        arm=make_joint_arm(parent,mode)
        assert not arm['module_kwargs']['spectral'] and not arm['module_kwargs']['use_quality']
        assert arm['module_kwargs']['protect_rgb'] and arm['module_kwargs']['use_memory'] and arm['module_kwargs']['use_motion']
        assert arm['Train_delivery']==parent['Train_delivery'] and arm['fit']==parent['fit']
        assert arm['source_RGB_sha256']==parent['source_RGB_sha256']
    assert before==parent


def valid_history():
    parent=study()
    saved=dict(completed_epochs=1,history=[dict(frozen_RGB_exact_unchanged=True,
        delivery=dict(schedule_proof(parent['Train_delivery']['lengths'],1),
            actual_address_sha256=schedule_proof(parent['Train_delivery']['lengths'],1)['address_sha256'],
            delivered_clips=6814,complete_all_Train_frames=True))],
        metadata=dict(seed=0,cue='ir',study_sha256=sha256(STUDY),source_RGB_sha256=parent['source_RGB_sha256']))
    return parent,saved


def test_checkpoint_requires_actual_all_video_train_delivery_and_frozen_teacher():
    parent,saved=valid_history()
    assert validate_history(saved,1,parent)==saved['metadata']
    with pytest.raises(ValueError): validate_history(saved,2,parent)


@pytest.mark.parametrize('damage',['addresses','reported_addresses','subset','padding','frozen','seed','cue','teacher'])
def test_equal_loss_or_epoch_count_cannot_replace_matching_actual_train_addresses(damage):
    parent,saved=valid_history();delivery=saved['history'][0]['delivery']
    if damage=='addresses':delivery['actual_address_sha256']='wrong'
    elif damage=='reported_addresses':delivery['address_sha256']='wrong'
    elif damage=='subset':delivery['frames']=1000
    elif damage=='padding':delivery['repeated_frames']=1
    elif damage=='frozen':saved['history'][0]['frozen_RGB_exact_unchanged']=False
    elif damage=='seed':saved['metadata']['seed']=1
    elif damage=='cue':saved['metadata']['cue']='gray'
    elif damage=='teacher':saved['metadata']['source_RGB_sha256']='different'
    with pytest.raises(ValueError):validate_history(saved,1,parent)


def test_common_budget_hash_is_fixed_by_complete_train_schedule():
    parent=study()
    assert schedule_sha256(parent,6)==schedule_sha256(copy.deepcopy(parent),6)
    assert schedule_sha256(parent,6)!=schedule_sha256(parent,9)
    with pytest.raises(ValueError):schedule_sha256(parent,3)


def test_incomplete_pipeline_registration_fails_before_any_gpu_settings(tmp_path,monkeypatch):
    from tools import native_pyramid_experiment
    called=[];monkeypatch.setattr(native_pyramid_experiment,'gpu_settings',lambda:called.append(True))
    (tmp_path/REGISTRY).write_text(json.dumps(dict(
        status='REGISTERED_COMPLETE_FACTORIAL_PIPELINE_PENDING_DEV_GATE_AND_GPU_ENGINEERING',
        complete_GPU_Dev_Test_pipeline_implemented=False)))
    with pytest.raises(AssertionError):require_gpu(tmp_path)
    assert not called and not torch.cuda.is_initialized()


def test_unregistered_canary_lengths_fail_before_reading_or_training():
    from tools.train_fusion_factorial import canary
    with pytest.raises(ValueError):canary(None,None,None,None,None,None,{},'cpu',3)
