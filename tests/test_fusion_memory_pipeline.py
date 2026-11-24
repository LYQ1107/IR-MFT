"""CPU-only controller integration; synthetic observations and no GPU/IR efficacy."""
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

from test_fusion_memory_evaluation import complete_fixture,official
from test_fusion_memory_results import receipt_fixture
from tools.fusion_memory_dev_freeze import normal_dev_freeze,final_dev_freeze,load_dev_files
from tools.fusion_memory_gpu_checks import known_plane_ROI
from tools.fusion_memory_evaluation import read
from tools.fusion_memory_observation import write_json
from tools.fusion_memory_pipeline import (Actions,ProcessQueue,COMPLETE,execute,worker_environment)
from tools.fusion_memory_results import analysis_roles
from tools.export_fusion_memory import validate_execution_state
import tools.fusion_memory_pipeline as pipeline


class SyntheticActions:
    def __init__(self,root,official,eligible=True):
        self.root,self.official,self.eligible=root,official,eligible
        self.paths,self.specs,self.events={},{},[]
        if eligible:
            self.plan,self.normal,self.additional,self.original_choice=complete_fixture(root/'artificial_files',official)

    def prepare(self):self.events.append('prepare');return self.eligible

    def specify(self,phase):
        self.events.append('specify_'+phase)
        if phase=='dev_normal':cases=self.plan['normal_cases']
        elif phase=='dev_additional':cases=read(self.root/'normal_Dev_selection.json')['additional_cases']
        else:cases=read(self.root/'frozen_Test_case_plan.json')['cases']
        value=dict(split='test' if phase=='test_frozen' else 'dev',cases=cases)
        path=self.root/'specifications'/(phase+'.json');write_json(path,value)
        self.paths[phase],self.specs[phase]=path,value
        return path

    def commands(self,phase,trial):return Actions.commands(self,phase,trial)
    def check(self,phase,trial,key,entry):self.events.append('checked_'+phase);return True

    def freeze_normal(self):
        self.events.append('freeze_normal')
        normal_dev_freeze(self.plan,self.normal,self.official[0],self.root/'normal_Dev_selection.json')

    def freeze_all(self):
        self.events.append('freeze_all')
        final_dev_freeze(self.plan,self.root/'normal_Dev_selection.json',self.normal,self.additional,
            self.official[0],self.root/'frozen_Test_case_plan.json')

    def export(self,ineligible=False):
        self.events.append('export_ineligible' if ineligible else 'export_complete')
        if not ineligible:
            normal,_,_=load_dev_files(self.plan,self.plan['normal_cases'],self.normal,self.official[0])
            additions=read(self.root/'normal_Dev_selection.json')['additional_cases']
            extra,_,_=load_dev_files(self.plan,additions,self.additional,self.official[0])
            roles=analysis_roles(self.plan,read(self.root/'frozen_Test_case_plan.json'),{**normal,**extra})
            assert not roles['Test_input']


class SyntheticQueue:
    def __init__(self):
        self.state=dict(phases={},fixed_seed=0,physical_gpus=['4','5'],maximum_GPU_workers=2,
                        worker_python=sys.executable,goal_met=False)
        self.saved=[]
    def save(self):self.saved.append(self.state['status'])
    def jobs(self,phase,commands,gpu,check):
        assert phase not in self.state['phases']
        entries={key:dict(command=command,status='COMPLETE',exit_code=0,pid=2_000_000_000,start_ticks=1)
                 for key,command in commands.items()}
        self.state['phases'][phase]=entries
        for key,entry in entries.items():assert check(key,entry) is not False


def test_complete169_job_controller_graph_freezes_real_synthetic_Dev_files_before_Test(tmp_path,official):
    actions=SyntheticActions(tmp_path,official);queue=SyntheticQueue()
    execute(queue,actions)
    assert queue.state['status']==COMPLETE and len(queue.state['phases'])==8
    assert sum(len(p) for p in queue.state['phases'].values())==169
    assert len(queue.state['phases']['TEST_EVALUATE'])==78
    assert actions.events.index('freeze_normal')<actions.events.index('specify_dev_additional')
    assert actions.events.index('freeze_all')<actions.events.index('specify_test_frozen')
    assert actions.events[-1]=='export_complete'
    assert not queue.state['goal_met']


def test_no_eligible_parent_has_no_F4_observation_Test_or_GPU_phase(tmp_path,official):
    actions=SyntheticActions(tmp_path,official,False);queue=SyntheticQueue()
    execute(queue,actions)
    assert queue.state['status']=='COMPLETE_F4_NOT_ELIGIBLE_NO_F4_EXPERIMENT'
    assert not queue.state['phases'] and actions.events==['prepare','export_ineligible']


def test_failed_Dev_freeze_stops_before_any_Test_command(tmp_path,official):
    actions=SyntheticActions(tmp_path,official);queue=SyntheticQueue()
    def failed():raise ValueError('Artificial incomplete Dev evidence')
    actions.freeze_all=failed
    with pytest.raises(ValueError,match='incomplete Dev'):execute(queue,actions)
    assert not any(name.startswith('TEST_') for name in queue.state['phases'])
    assert 'export_complete' not in actions.events


def test_known_plane_CPU_geometry_is_exact_but_cannot_claim_actual_CUDA():
    result=known_plane_ROI('cpu')
    assert result['actual_raw_ROI_average']==[24.,12.] and result['exact']
    assert not result['actual_CUDA'] and not result['authorization_established_here']


def workflow(root):
    return SimpleNamespace(base=root/'queue',work=root,plan=dict(data_root=str(root/'DATA')),
                           registered=lambda:dict(artificial=True))


@pytest.mark.parametrize('gpu,slots',[(True,['4','5']),(False,['CPU0','CPU1'])])
def test_worker_environment_is_bounded_and_uses_the_exact_physical_slots(tmp_path,gpu,slots):
    value=workflow(tmp_path)
    for slot in slots:
        environment=worker_environment(value,slot,gpu)
        assert environment['CUDA_VISIBLE_DEVICES']==(slot if gpu else '')
        assert environment['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
        assert environment['TORCHINDUCTOR_CACHE_DIR']==str(value.base/'kernel_cache/inductor')
        assert environment['TRITON_CACHE_DIR']==str(value.base/'kernel_cache/triton')
    with pytest.raises(ValueError):worker_environment(value,'2',gpu)


def test_real_CPU_child_handles_maximum_two_exact_exits_and_private_logs(tmp_path):
    value=workflow(tmp_path);state=dict(phases={});observed=[]
    queue=ProcessQueue(value,state,resources=False)
    def sleep(_):observed.append(len(queue.active));time.sleep(.02)
    queue.sleep=sleep
    code='import os,time;time.sleep(.15);print(repr(os.environ.get("CUDA_VISIBLE_DEVICES")))'
    commands={str(i):[sys.executable,'-c',code] for i in range(4)}
    queue.jobs('ARTIFICIAL_CPU',commands,False,lambda key,entry:True)
    assert max(observed)==2 and not queue.active
    for key,job in state['phases']['ARTIFICIAL_CPU'].items():
        assert job['exit_code']==0 and job['status']=='COMPLETE' and job['start_ticks']>0
        assert Path(job['log']).read_text().strip()=="''" and job['command']==commands[key]
        assert job['worker_wall_upper_seconds']>=job['worker_wall_lower_seconds']>=0


def test_real_failed_CPU_child_preserves_healthy_sibling_and_does_not_launch_pending(tmp_path):
    value=workflow(tmp_path);state=dict(phases={});queue=ProcessQueue(value,state,resources=False,sleep=lambda _:time.sleep(.02))
    commands=dict(failed=[sys.executable,'-c','import sys,time;time.sleep(.1);sys.exit(7)'],
        healthy=[sys.executable,'-c','import time;time.sleep(.25);print("complete")'],
        pending=[sys.executable,'-c','raise RuntimeError("must not run")'])
    with pytest.raises(RuntimeError,match='Preserve failed'):queue.jobs('ARTIFICIAL_FAILURE',commands,False,lambda key,entry:True)
    entries=state['phases']['ARTIFICIAL_FAILURE']
    assert entries['failed']['exit_code']==7 and entries['healthy']['status']=='COMPLETE'
    assert entries['pending']['status']=='PENDING' and 'pid' not in entries['pending']
    assert not queue.active and Path(entries['healthy']['log']).read_text().strip()=='complete'


def test_real_resource_wait_does_not_create_a_child_or_restart_a_phase(tmp_path):
    value=workflow(tmp_path);state=dict(phases={});queue=ProcessQueue(value,state,resources=False)
    attempts=[]
    def unavailable(count):
        attempts.append(count)
        if len(attempts)==1:raise RuntimeError('Artificial resource reserve wait')
    queue.admission=unavailable;queue.sleep=lambda _:time.sleep(.02)
    queue.jobs('ARTIFICIAL_WAIT',{'one':[sys.executable,'-c','import time;time.sleep(.1)']},False,lambda *a:True)
    assert attempts==[1,1] and state['phases']['ARTIFICIAL_WAIT']['one']['status']=='COMPLETE'
    with pytest.raises(ValueError,match='no automatic restart'):
        queue.jobs('ARTIFICIAL_WAIT',{'one':[sys.executable,'-c','pass']},False,lambda *a:True)


def test_actual_action_metric_callback_uses_registered_driver_and_full7_files(tmp_path,official,monkeypatch):
    _,path,case,context,_=receipt_fixture(tmp_path,official)
    config=read(Path(__file__).resolve().parents[1]/'configs/perceptual_mft_20261122.json')
    value=SimpleNamespace(base=tmp_path,plan=config)
    actions=Actions(value);actions.paths['test_frozen']=Path(context['specification']);actions.specs['test_frozen']=context['spec']
    monkeypatch.setattr(pipeline,'metric_context',lambda *a:context)
    actions.check('TEST_EVALUATE','test_frozen',case['id'],{})
    driver=path.parent/'driver_validation.json';data=read(driver);data['protocol_GT_file_sha256']={};write_json(driver,data)
    with pytest.raises(ValueError,match='driver/GT receipt'):
        actions.check('TEST_EVALUATE','test_frozen',case['id'],{})


def test_actual_action_callback_rejects_CPU_geometry_as_GPU_qualification(tmp_path):
    value=workflow(tmp_path);value.base.mkdir(parents=True)
    spec=value.base/'spec.json';write_json(spec,dict(artificial=True))
    actions=Actions(value);actions.paths['dev_normal']=spec;actions.specs['dev_normal']=read(spec)
    write_json(value.base/'gpu_engineering.json',dict(status='COMPLETE_REAL_GPU_F4_SOURCE_AND_MEMORY_ENGINEERING',
        actual_GPU_tensors_verified=True,actual_GPU_ROI_coordinate_check=known_plane_ROI('cpu')))
    with pytest.raises(ValueError,match='GPU source/ROI proof'):
        actions.check('GPU_ENGINEERING','dev_normal','actual',{})
