import json
from pathlib import Path
import sys
import time

import pytest

from tools import launch_fusion_factorial as launch
from tools.native_pyramid_train_data import schedule_proof
from tools.run_pseudo_ir_comparison import dump,sha256


class GraphQueue:
    def __init__(self,base,authorization,fail=None):
        self.base,self.authorization,self.fail=base,authorization,fail
        self.state=dict(phases={});self.events=[]

    def save(self):pass

    def jobs(self,phase,tasks,gpu,check):
        self.events.append(phase)
        if phase==self.fail:raise RuntimeError('simulated preserved worker failure')
        auth=self.authorization
        if phase=='GPU_ENGINEERING':
            dump(self.base/'engineering.json',dict(status='COMPLETE_REAL_GPU_JOINT_FACTORIAL_ENGINEERING',
                authorization_sha256=sha256(self.base/'authorization.json'),record=dict(
                    initial_native_sha256=auth['initial_native_sha256'],inference_B1_frames=8,
                    source_zero_missing_checks=[dict(direct_B1_packets_exact=True,zero_full_head_exact=True,
                        trained_missing_full_head_exact=self.fail!='INVALID_PROBE') for _ in range(8)],
                    delivered_Train_frames=8,actual_updates=2,optimizer_rows=[
                        dict(optimizer_updated=True,native_stem_gradient_L1=v) for v in [0.,1.]],
                    frozen_RGB_parameters_and_BN_unchanged_before_explicit_head_precision_conversion=True,
                    inference_RGB_state_unchanged=True),peak_process_RSS_GiB=1.))
        elif phase=='FULL_TRAIN':
            study=launch.read(launch.STUDY)
            history=[]
            for epoch in range(1,auth['stage']+1):
                schedule=schedule_proof(study['Train_delivery']['lengths'],epoch)
                history.append(dict(frozen_RGB_exact_unchanged=True,delivery=dict(schedule,
                    actual_address_sha256=schedule['address_sha256'],delivered_clips=6814,complete_all_Train_frames=True)))
            dump(self.base/'fit/progress.json',dict(test_fixture=True,completed_full_Train_epochs=auth['stage'],
                registered_through_epoch=auth['stage'],frozen_RGB_exact_unchanged=True,history=history,
                metadata=dict(seed=0,cue='ir',study_sha256=sha256(launch.STUDY),
                    source_RGB_sha256=study['source_RGB_sha256'],initial_native_sha256=auth['initial_native_sha256'])))
            dump(self.base/'fit_complete.json',dict(status='COMPLETE_ONE_JOINT_DISABLE_FULL_TRAIN_FIT',
                stage=auth['stage'],authorization_sha256=sha256(self.base/'authorization.json'),
                progress_sha256=sha256(self.base/'fit/progress.json')))
        elif phase=='INDIVIDUAL_RUNTIME':
            freeze=json.loads((self.base/'Test_selection_frozen.json').read_text())
            dump(self.base/'individual_runtime.json',dict(status='COMPLETE_FROZEN_FACTORIAL_INDIVIDUAL_PREFIX_RUNTIME',
                Test_selection_sha256=sha256(self.base/'Test_selection_frozen.json'),
                rows=[dict(id=launch.FRESH['id'])]+freeze['selected_unique_models']))
        check(tasks[0])


def configure(monkeypatch,tmp_path,fail=None):
    auth=dict(status='AUTHORIZED_ONE_JOINT_DISABLE_FIT_FROM_COMPLETE_PARENT_DEV_ONLY',
        stage=6,mode='quality',initial_native_sha256='initial',cell_fits={'00':{}})
    dump(tmp_path/'authorization.json',auth)
    queue=GraphQueue(tmp_path,auth,fail)
    monkeypatch.setattr(launch,'prepare_authorization',lambda registration:auth)
    monkeypatch.setattr(launch,'model_row',lambda cell,e,fit,mode:dict(id=f'FQ_{cell}_{e}',cell=cell,epoch=e))
    def trial(queue,models,split):
        queue.events.append(split)
        if fail==split:raise RuntimeError('incomplete '+split)
        if split=='test':assert (tmp_path/'Test_selection_frozen.json').exists()
    monkeypatch.setattr(launch,'run_trial',trial)
    def freeze(authorization):
        queue.events.append('DEV_FREEZE')
        value=dict(complete_all_four_Dev=True,Test_accuracy_used_for_selection=False,
            Dev_improvement_is_negative=True,selected_unique_models=[dict(id='FQ_00_3')])
        dump(tmp_path/'Test_selection_frozen.json',value);return value
    monkeypatch.setattr(launch,'complete_dev_and_freeze',freeze)
    def reuse(freeze):
        queue.events.append('ARTIFACT_REUSE');return dict(missing_models=freeze['selected_unique_models'])
    monkeypatch.setattr(launch,'plan_test_reuse',reuse)
    return queue


def test_negative_parent_dev_never_launches_an_extra_fit(monkeypatch,tmp_path):
    queue=GraphQueue(tmp_path,{})
    monkeypatch.setattr(launch,'prepare_authorization',lambda _:dict(
        status='FACTORIAL_EXTENSION_NOT_TRIGGERED_BY_COMPLETE_PARENT_DEV',Test_accuracy_used=False))
    launch.execute(queue,{})
    assert queue.events==[] and (tmp_path/'not_triggered.json').exists()
    assert queue.state['status']=='COMPLETE_FACTORIAL_EXTENSION_NOT_TRIGGERED_BY_DEV'


def test_negative_ablation_dev_does_not_skip_frozen_full_test_or_cost(monkeypatch,tmp_path):
    queue=configure(monkeypatch,tmp_path);launch.execute(queue,{})
    assert queue.events==['GPU_ENGINEERING','FULL_TRAIN','dev','DEV_FREEZE','ARTIFACT_REUSE','test','INDIVIDUAL_RUNTIME']
    assert queue.state['status']=='COMPLETE_FACTORIAL_EXECUTION_PENDING_SCIENTIFIC_EXPORT'


@pytest.mark.parametrize('failure,forbidden',[('GPU_ENGINEERING','FULL_TRAIN'),('FULL_TRAIN','dev'),('dev','DEV_FREEZE'),('test','INDIVIDUAL_RUNTIME')])
def test_failed_prerequisite_cannot_be_treated_as_completed_evidence(monkeypatch,tmp_path,failure,forbidden):
    queue=configure(monkeypatch,tmp_path,failure)
    with pytest.raises(RuntimeError):launch.execute(queue,{})
    assert forbidden not in queue.events


def test_eight_reported_frames_with_a_failed_missing_ir_check_cannot_authorize_fitting(monkeypatch,tmp_path):
    queue=configure(monkeypatch,tmp_path,'INVALID_PROBE')
    with pytest.raises(AssertionError):launch.execute(queue,{})
    assert 'FULL_TRAIN' not in queue.events


def test_failed_cpu_child_does_not_terminate_healthy_sibling_or_start_more_jobs(monkeypatch,tmp_path):
    monkeypatch.setattr(launch,'registered',lambda work:{})
    monkeypatch.setattr(launch,'require_storage',lambda *a,**k:None)
    monkeypatch.setattr(launch,'require_host_memory',lambda *a,**k:None)
    real_sleep=time.sleep;monkeypatch.setattr(launch.time,'sleep',lambda _:real_sleep(.01))
    marker=tmp_path/'healthy_done.txt';pending=tmp_path/'must_not_start.txt'
    tasks=[dict(id='bad',command=[sys.executable,'-c','raise SystemExit(7)']),
        dict(id='good',command=[sys.executable,'-c',
            'import pathlib,time;time.sleep(.15);pathlib.Path('+repr(str(marker))+').write_text("done")']),
        dict(id='pending',command=[sys.executable,'-c','import pathlib;pathlib.Path('+repr(str(pending))+').write_text("wrong")'])]
    queue=launch.Queue(tmp_path,dict(phases={}),str(tmp_path))
    with pytest.raises(RuntimeError):queue.jobs('CPU_FIXTURE',tasks,False,lambda entry:marker.exists())
    entries=queue.state['phases']['CPU_FIXTURE']
    assert entries['bad']['exit_code']==7 and entries['bad']['status']=='FAILED'
    assert entries['good']['exit_code']==0 and entries['good']['status']=='COMPLETE' and marker.read_text()=='done'
    assert entries['pending']['status']=='PENDING' and not pending.exists()
