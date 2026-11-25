"""Protocol/process/artifact fixtures only; not actual GPU fitting or accuracy."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

from tools.native_pyramid_experiment import read
from tools.native_pyramid_train_data import schedule_proof
from tools.probe_task_utility_workflow import simulate_graph,verify_real_CPU_queue
from tools.task_utility_pipeline import verify_fit,verify_probe
from tools.task_utility_queue import Queue
from tools.task_utility_study import STUDY,REGISTRY,arm_path,make_arm
from tools.run_pseudo_ir_comparison import sha256
from tools.export_task_utility import validate_execution_state
import tools.task_utility_pipeline as pipeline


@pytest.mark.parametrize('stop',[6,9,12])
def test_real_orchestration_graph_keeps_common_stages_all_negative_models_and26_Test(tmp_path,stop):
    value=simulate_graph(tmp_path,read(STUDY),stop)
    assert value['stages']==[f'TRAIN_{s:02d}' for s in [6,9,12] if s<=stop]
    assert value['Dev_case_count']==20*stop+2 and value['frozen_Test_case_count']==26
    assert value['phase_job_counts']['NORMAL_GPU_ENGINEERING']==12
    assert value['phase_job_counts']['test_frozen_NN']==7 and value['phase_job_counts']['test_frozen_OFFICIAL']==24
    assert value['phase_job_counts']['INDIVIDUAL_RUNTIME']==26
    assert sum(batch==4 for name,batch in value['runtime_routes'])==1
    assert sum(batch==1 for name,batch in value['runtime_routes'])==25
    assert not value['actual_GPU_work'] and not value['actual_NN_or_Train_Test_evaluation']
    base=Path(value['artificial_base']);state=read(base/'artificial_state.json')
    state.update(study_sha256=sha256(STUDY),pipeline_registration_sha256=sha256(base/REGISTRY),through_epoch=stop)
    with pytest.raises(ValueError,match='Every original worker'):
        validate_execution_state(base,state) # Simulated jobs cannot become actual execution evidence.


def test_two_real_CPU_processes_keep_identity_exit_and_complete_phase_barrier(tmp_path):
    result=verify_real_CPU_queue(tmp_path)
    jobs=result['phases']['CPU_LIFECYCLE'];assert len(jobs)==2
    assert {j['slot'] for j in jobs.values()}=={'CPU0','CPU1'}
    assert all(j['status']=='COMPLETE' and j['exit_code']==0 and j['argv_identity_observed'] for j in jobs.values())
    assert len({j['pid'] for j in jobs.values()})==2


def test_failed_child_is_preserved_healthy_sibling_finishes_and_pending_job_does_not_start(tmp_path):
    state=dict(artificial_fixture=True,phases={});completed=[];marker=tmp_path/'healthy.txt';forbidden=tmp_path/'forbidden.txt'
    queue=Queue(tmp_path,tmp_path,state,'unused',lambda:True,poll_seconds=.01,storage_guard=lambda *a,**k:None,
        memory_guard=lambda *a,**k:None,force_CPU_for_artificial_test=True)
    tasks=[dict(id='fails',command=[sys.executable,'-u','-c','import time;time.sleep(.2);raise SystemExit(7)']),
        dict(id='healthy',command=[sys.executable,'-u','-c',f'import pathlib,time;time.sleep(.4);pathlib.Path({str(marker)!r}).write_text("ok");time.sleep(.1)']),
        dict(id='must_wait',command=[sys.executable,'-u','-c',f'import pathlib;pathlib.Path({str(forbidden)!r}).write_text("bad")'])]
    with pytest.raises(RuntimeError,match='Preserve failed child'):
        queue.jobs('FAILURE_PRESERVATION',tasks,False,lambda e:completed.append(e['id']))
    jobs=state['phases']['FAILURE_PRESERVATION']
    assert jobs['fails']['exit_code']==7 and jobs['fails']['status']=='FAILED'
    assert marker.read_text()=='ok' and jobs['healthy']['status']=='COMPLETE' and completed==['healthy']
    assert jobs['must_wait']['status']=='PENDING' and not forbidden.exists() and not queue.active


def test_source_change_stops_new_jobs_without_killing_already_launched_workers(tmp_path):
    state=dict(artificial_fixture=True,phases={});calls=[]
    def guard():
        calls.append(1)
        if len(calls)>1:raise ValueError('changed source fixture')
    queue=Queue(tmp_path,tmp_path,state,'unused',guard,poll_seconds=.01,storage_guard=lambda *a,**k:None,
        memory_guard=lambda *a,**k:None,force_CPU_for_artificial_test=True)
    jobs=[dict(id=str(i),command=[sys.executable,'-u','-c','import time;time.sleep(.2)']) for i in range(3)]
    with pytest.raises(RuntimeError,match='changed source'):
        queue.jobs('SOURCE_FAILURE',jobs,False,lambda e:True)
    rows=state['phases']['SOURCE_FAILURE']
    assert rows['0']['status']==rows['1']['status']=='COMPLETE' and rows['2']['status']=='PENDING'
    assert rows['0']['exit_code']==rows['1']['exit_code']==0 and not queue.active


def test_formal_queue_cannot_enable_CPU_fixture_override_or_other_GPUs(tmp_path):
    with pytest.raises(ValueError,match='unavailable'):
        Queue(tmp_path,tmp_path,dict(phases={}),'.',lambda:True,force_CPU_for_artificial_test=True)
    queue=Queue(tmp_path,tmp_path,dict(phases={}),'.',lambda:True)
    with pytest.raises(ValueError,match='GPU4/5'):
        queue.jobs('BAD_CARD',[dict(id='no',command=['forbidden'],physical_gpu='2')],True,lambda e:True)


def test_CPU_worker_artifact_cannot_be_used_as_actual_GPU_qualification(tmp_path):
    study=read(STUDY);fit=next(f for f in study['normal_fits'] if f['cue']=='ir');arm=make_arm(study,fit['id'],fit['variant'],fit['cue'])
    config=arm_path(tmp_path,fit['id']);config.parent.mkdir();config.write_text(json.dumps(arm))
    registration=dict(source_hashes={'artificial':'not_GPU_qualification'});(tmp_path/REGISTRY).write_text(json.dumps(registration))
    actual=read(Path(__file__).resolve().parents[1]/'docs/fusion_research/PERCEPTUAL_MFT_20261125_TASK_UTILITY_WORKER_CPU_ENGINEERING.json')
    value=dict(status='COMPLETE_TASK_UTILITY_GPU_TRAIN_B1_ENGINEERING',GPU_work=True,Dev_rows=0,Test_rows=0,peak_process_RSS_GiB=2.,
        record=copy.deepcopy(actual['actual_record']),metadata=dict(seed=0,fit_id=fit['id'],variant=arm['variant'],cue=arm['cue'],
            study_sha256=sha256(STUDY),fit_config_sha256=sha256(config),registry_sha256=sha256(tmp_path/REGISTRY),
            source_hashes=registration['source_hashes'],source_RGB_sha256=study['source_RGB_sha256'],
            protocol_sha256=study['protocol_sha256'],calibration_sha256=study['calibration_sha256'],
            initial_native_sha256=study['architecture']['initial_state_sha256'],task_supervision=arm['task_supervision'],
            physical_gpu=fit['physical_gpu'],GPU_uuid='not_reached'))
    path=tmp_path/'engineering'/fit['id']/'proof.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='Twelve actual B1'):
        verify_probe(tmp_path,fit['id'],study,registration)


def test_fit_validation_reloads_tensor_snapshot_and_requires_exact_complete_progress_history(tmp_path):
    from test_task_utility_model_loading import artifact
    study,arm,registration,saved,row=artifact(tmp_path);fit_id=arm['fit_id'];directory=tmp_path/'fits'/fit_id
    original=saved['history'][0];history=[]
    for epoch in range(1,7):
        value=copy.deepcopy(original);delivery=schedule_proof(arm['Train_delivery']['lengths'],epoch)
        delivery.update(actual_address_sha256=delivery['address_sha256'],complete_all_Train_frames=True,delivered_clips=6814)
        value.update(epoch=epoch,delivery=delivery,delivered_clips_total=6814*epoch,actual_optimizer_updates=6814*epoch)
        history.append(value);snapshot=copy.deepcopy(saved);snapshot.update(completed_epochs=epoch,history=copy.deepcopy(history))
        torch.save(snapshot,directory/f'epoch_{epoch:02d}.pt')
    progress=dict(completed_full_Train_epochs=6,registered_through_epoch=6,metadata=saved['metadata'],history=history)
    path=directory/'progress.json';path.write_text(json.dumps(progress))
    assert verify_fit(tmp_path,fit_id,6,study,registration)['checkpoint_sha256']==sha256(directory/'epoch_06.pt')
    progress['history'][-1]['mean_clip_losses']['loss']+=.01;path.write_text(json.dumps(progress))
    with pytest.raises(ValueError,match='histories differ'):verify_fit(tmp_path,fit_id,6,study,registration)


@pytest.mark.parametrize('exit_code,process_exited',[(1,True),(0,False)])
def test_registration_requires_actual_CPU_proof_exit0_and_observed_process_exit(tmp_path,monkeypatch,exit_code,process_exited):
    import tools.neck72_runtime_study as runtime_study
    identity=dict(pid=123,start_ticks=456,command_sha256='fake_not_a_real_process')
    base=tmp_path/'CPU_proof';base.mkdir();proof=base/'complete.json';proof.write_text(json.dumps(dict(identity=identity)))
    log=tmp_path/'proof.log';log.write_text('artificial log')
    base.with_suffix('.execution.json').write_text(json.dumps(dict(exit_code=exit_code,identity=identity,pid=123,
        log=str(log),log_sha256=sha256(log),RSS_limit_exceeded=False)))
    monkeypatch.setattr(pipeline,'context',lambda:({}, {}, {}, {},tmp_path,None,tmp_path/'never_registered'))
    monkeypatch.setattr(pipeline,'current_source_hashes',lambda:{'artificial':'none'})
    monkeypatch.setattr(runtime_study,'exited',lambda *a,**k:process_exited)
    with pytest.raises(ValueError,match='finish and exit0'):
        pipeline.register(proof)
    assert not (tmp_path/'never_registered').exists()
