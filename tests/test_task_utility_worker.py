"""Artificial arm/history/release data; no actual GPU or model qualification."""
import copy
import json
import sys

import pytest
import torch

from tools.native_pyramid_train_data import schedule_proof
from tools.task_utility_fit import LOSS_FIELDS,COUNT_FIELDS
from tools.task_utility_fusion import TaskUtilityFusion
from tools.task_utility_observer import TaskUtilityObserver
import tools.task_utility_study as study
import tools.train_task_utility as worker


def configuration():return json.loads(study.STUDY.read_text())


def arm():
    return study.make_arm(configuration(),'TU_foreground_utility_labels_ir','foreground_utility_labels','ir')


def test_all_twelve_arms_share_capacity_but_bind_distinct_supervision_and_cues():
    config=configuration();arms=[study.make_arm(config,f['id'],f['variant'],f['cue']) for f in config['normal_fits']]
    assert len(arms)==12 and len({study.computation_key(a) for a in arms})==12
    assert len({json.dumps(a['module_kwargs'],sort_keys=True) for a in arms})==1
    assert {tuple(a['task_supervision'][k] for k in ['foreground','utility']) for a in arms}=={(False,False),(True,False),(False,True),(True,True)}
    arms[0]['fit']['learning_rate']=.1
    assert configuration()['fit']==config['fit'] and arms[1]['fit']==config['fit']


def test_computation_key_binds_loss_weights_temperature_and_exact_image_reference():
    reference=arm();key=study.computation_key(reference)
    changes=[('task_supervision','lambda_foreground',.2),('task_supervision','lambda_utility',.2),
        ('task_supervision','utility_temperature',.2),('fit','lambda_reliable_RGB',.1)]
    for section,name,value in changes:
        changed=copy.deepcopy(reference);changed[section][name]=value;assert study.computation_key(changed)!=key
    changed=copy.deepcopy(reference);changed['input_content_reference_sha256']='different'
    assert study.computation_key(changed)!=key
    with pytest.raises(ValueError):study.make_arm(configuration(),'seed1','foreground_labels','ir')
    with pytest.raises(ValueError):study.arm_path('/tmp','../../other')


def finished(ineligible=False):
    identity=dict(pid=1,start_ticks=10,command_sha256='controller')
    phases={}
    if not ineligible:
        for i,name in enumerate(['GPU_ENGINEERING','DEV_NORMAL_PRODUCE','DEV_NORMAL_EVALUATE',
                'DEV_ADDITIONAL_PRODUCE','DEV_ADDITIONAL_EVALUATE','TEST_PRODUCE','TEST_EVALUATE','RUNTIME']):
            phases[name]={'fake':dict(pid=100+i,start_ticks=1000+i,command_sha256='child',
                command=['explicitly_artificial'],status='COMPLETE',exit_code=0)}
    state=dict(status='COMPLETE_F4_NOT_ELIGIBLE_NO_F4_EXPERIMENT' if ineligible else 'COMPLETE_F4_STUDY_REQUIRES_GLOBAL_SCIENTIFIC_ASSESSMENT',
        controller_pid=1,controller_identity=identity,study_sha256='study',pipeline_registration_sha256='registration',phases=phases)
    report=dict(status='F4_NOT_ELIGIBLE_FROM_ORIGINAL_COMPLETE_DEV_NOT_AN_F4_EXPERIMENT' if ineligible else 'COMPLETE_F4_FORMAL_EXPORT_REQUIRES_GLOBAL_SCIENTIFIC_ASSESSMENT',
        study_sha256='study',pipeline_registration_sha256='registration')
    export=dict(exit_code=0,identity=dict(pid=2,start_ticks=20,command_sha256='exporter'))
    return state,identity,report,export


def test_release_waits_for_actual_controller_exporter_and_every_child_exit():
    state,identity,report,export=finished()
    for alive in [{1},{2},{105}]:
        assert not study.release_state_ready(state,identity,report,export,process_exited=lambda p,**kw:p not in alive)
    assert study.release_state_ready(state,identity,report,export,process_exited=lambda p,**kw:True)
    assert not study.release_state_ready(dict(status='WAITING'),{}, {},{},process_exited=lambda p,**kw:True)


def test_release_ineligible_F4_is_compute_release_without_inventing_experiment():
    args=finished(ineligible=True)
    assert study.release_state_ready(*args,process_exited=lambda p,**kw:True)
    args[0]['phases']={'unregistered':{}}
    with pytest.raises(ValueError,match='Ineligible'):study.release_state_ready(*args,process_exited=lambda p,**kw:True)


def test_release_rejects_failed_or_forged_terminal_evidence():
    state,identity,report,export=finished()
    with pytest.raises(RuntimeError):study.release_state_ready(dict(status='FAILED'),identity,report,export)
    with pytest.raises(ValueError):study.release_state_ready(state,dict(identity,start_ticks=99),report,export)
    with pytest.raises(ValueError):study.release_state_ready(state,identity,dict(report,study_sha256='other'),export)
    state['phases'].pop('RUNTIME')
    with pytest.raises(ValueError):study.release_state_ready(state,identity,report,export,process_exited=lambda p,**kw:True)


def snapshot():
    a=arm();delivery=schedule_proof(a['Train_delivery']['lengths'],1)
    delivery.update(actual_address_sha256=delivery['address_sha256'],delivered_clips=6814,complete_all_Train_frames=True)
    metadata=dict(family='task_utility',seed=0,study_sha256=study.sha256(study.STUDY),
        source_RGB_sha256=a['source_RGB_sha256'],cue=a['cue'],fit_id=a['fit_id'],variant=a['variant'],
        task_supervision=copy.deepcopy(a['task_supervision']),computation_key=study.computation_key(a),
        initial_native_sha256=a['architecture']['initial_state_sha256'],Dev_fit_rows=0,Test_fit_rows=0)
    totals={k:0. for k in COUNT_FIELDS};totals.update(available_frames=27251,valid_task_cells=2725100)
    row=dict(epoch=1,delivery=delivery,frozen_RGB_exact_unchanged=True,delivered_clips_total=6814,
        actual_optimizer_updates=6814,mean_clip_losses={k:0. for k in LOSS_FIELDS},task_epoch_totals=totals)
    return dict(loop_family='task_utility_v1',loss_fields=list(LOSS_FIELDS),completed_epochs=1,history=[row],
        metadata=metadata,source_RGB_only_by_reference=True,artificial_not_a_real_checkpoint=True),a


def test_history_binds_task_supervision_family_and_rejects_old_loop():
    saved,a=snapshot();assert study.validate_history(saved,1,a)==saved['metadata']
    for key,value in [('loop_family','old'),('loss_fields',['loss'])]:
        bad=copy.deepcopy(saved);bad[key]=value
        with pytest.raises(ValueError):study.validate_history(bad,1,a)
    for key,value in [('seed',False),('family','native'),('initial_native_sha256','other'),('Test_fit_rows',1)]:
        bad=copy.deepcopy(saved);bad['metadata'][key]=value
        with pytest.raises(ValueError):study.validate_history(bad,1,a)
    bad=copy.deepcopy(saved);bad['metadata']['task_supervision']['utility']=False
    with pytest.raises(ValueError):study.validate_history(bad,1,a)


def test_history_requires_complete_address_delivery_and_all_new_losses_counts():
    saved,a=snapshot()
    for key,value in [('frames',27250),('actual_address_sha256','wrong'),('delivered_clips',6813)]:
        bad=copy.deepcopy(saved);bad['history'][0]['delivery'][key]=value
        with pytest.raises(ValueError):study.validate_history(bad,1,a)
    bad=copy.deepcopy(saved);bad['history'][0]['mean_clip_losses'].pop('utility')
    with pytest.raises(ValueError):study.validate_history(bad,1,a)
    bad=copy.deepcopy(saved);bad['history'][0]['task_epoch_totals']['available_frames']=27250
    with pytest.raises(ValueError):study.validate_history(bad,1,a)


def test_observer_retains_no_missing_pseudo_values_and_does_not_change_output():
    torch.set_num_threads(2);torch.manual_seed(0)
    module=TaskUtilityFusion(channels=(32,64,128)).eval().requires_grad_(False)
    features=tuple(torch.randn(1,c,h,w) for c,h,w in [(32,8,12),(64,4,6),(128,2,3)])
    scores=tuple(v[:,:1].new_full(v[:,:1].shape,.1) for v in features);cue=torch.randn(1,3,64,96)
    observer=TaskUtilityObserver();observer.attach('normal',module)
    with pytest.raises(ValueError):observer.attach('second',module)
    result=module(features,cue,scores)[0];assert all(torch.equal(a,b) for a,b in zip(result,features))
    module(features,None,scores,available=False)
    summary=observer.summary('normal');assert summary['frames']==2 and summary['missing_frames']==summary['available_frames']==1
    assert summary['levels'][0]['gate']['count']==96 and summary['levels'][0]['feature_changed']['max']==0
    observer.close();assert not module._forward_hooks


def test_observer_refuses_training_model():
    module=TaskUtilityFusion(channels=(32,64,128))
    with pytest.raises(ValueError,match='frozen'):TaskUtilityObserver().attach('wrong',module)


def test_incomplete_registry_stops_worker_before_any_GPU_setup(tmp_path,monkeypatch):
    monkeypatch.setattr(worker,'context',lambda:(None,None,None,None,tmp_path,None,tmp_path/'task_utility'))
    reached=[]
    def forbidden():reached.append(True);raise AssertionError('GPU setup must not be reached')
    monkeypatch.setattr(worker,'gpu_settings',forbidden)
    monkeypatch.setattr(sys,'argv',['worker','probe','--fit-id','TU_foreground_utility_labels_ir'])
    with pytest.raises(FileNotFoundError):worker.main()
    assert not reached and not torch.cuda.is_initialized()
