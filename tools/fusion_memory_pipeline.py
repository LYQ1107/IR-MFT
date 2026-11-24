"""Complete F4 CPU controller, registration and bounded child queue.

No implicit CUDA, NN fitting, Test selection or automatic failed-job restart.
CPU integration uses explicit fakes; real CLI always uses the actual executor.
"""
from datetime import datetime,timezone
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.export_fusion_memory import (EXECUTION_FINISHED, REPORT, export_ineligible, metric_context,
    validate_execution_state)
from tools.export_perceptual_metrics47 import official_schema
from tools.fusion_memory_dev_freeze import normal_dev_freeze,final_dev_freeze
from tools.fusion_memory_evaluation import read,sha,validate_mot
from tools.fusion_memory_observation import LENGTHS,write_json
from tools.fusion_memory_parent_io import ROOT,DevParentFiles,require
from tools.fusion_memory_results import checked_metric
from tools.fusion_memory_workflow import FusionMemoryWorkflow,STUDY,WORKFLOW
from tools.neck72_runtime import require_host_memory,require_storage
from tools.neck72_runtime_study import exited

WAITING='WAITING_FOR_COMPLETE_SPATIAL_EXPORT_AND_ACTUAL_EXITS'
COMPLETE='COMPLETE_F4_STUDY_REQUIRES_GLOBAL_SCIENTIFIC_ASSESSMENT'


def process_identity(pid=None):
    pid=os.getpid() if pid is None else pid
    proc=Path('/proc')/str(pid)
    return dict(pid=pid,start_ticks=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19]),
        command_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest())


def frozen_json(path,value):
    path=Path(path)
    if path.exists():require(read(path)==value,'Preserve original registered/controller artifact: '+str(path))
    else:write_json(path,value,exclusive=True)


def source_files():
    workflow=FusionMemoryWorkflow()
    previous=read(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261125_F4_RESULT_EXPORT_COMPONENTS_SOURCE_AUDIT.json')
    names=set()
    for row in previous['existing_registrations']:
        require(sha(row['registration'])==row['registration_sha256'],'Original predecessor registration changed')
        registered=read(row['registration'])
        require(all(sha(ROOT/p)==s for p,s in registered['source_hashes'].items()),'Original predecessor source changed')
        names.update(registered['source_hashes'])
    patterns=['tools/fusion_memory*.py','tools/probe_f4*.py','tests/test_fusion_memory*.py',
        'configs/perceptual_f4*.json','configs/perceptual_fusion_memory*.json']
    names.update(str(p.relative_to(ROOT)) for pattern in patterns for p in ROOT.glob(pattern))
    names.update(workflow.config['complete_pipeline_entrypoints'])
    names.update(['tools/probe_f4_pipeline_cpu.py','yolox/tracker/fusion_memory_tracker.py',str(WORKFLOW.relative_to(ROOT))])
    require(all((ROOT/p).is_file() for p in names),'Complete F4 source/entrypoint file missing')
    return sorted(names)


def current_source_hashes():
    return {p:sha(ROOT/p) for p in source_files()}


def register(proof_path):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Register F4 on CPU only')
    workflow=FusionMemoryWorkflow();proof=read(proof_path)
    hashes=current_source_hashes()
    require(proof['status']=='COMPLETE_F4_WORKFLOW_CPU_INTEGRATION' and proof['source_hashes']==hashes and
        proof['actual_GPU_qualification'] is False and proof['actual_parent_eligibility'] is False,
        'Current complete CPU integration proof required; no implicit real GPU/parent qualification')
    require(all(sha(p)==s for p,s in proof['artifact_hashes'].items()),'CPU integration dependencies changed')
    require_storage(workflow.work,growth_bytes=int(.5*2**30))
    value=dict(status='REGISTERED_COMPLETE_F4_PIPELINE_PENDING_REAL_GPU_ENGINEERING',study_sha256=sha(STUDY),
        workflow_sha256=sha(WORKFLOW),source_hashes=hashes,
        artifact_hashes=dict(proof['artifact_hashes'],**{str(Path(proof_path).resolve()):sha(proof_path)}),
        complete_workflow_CPU_proof=str(Path(proof_path).resolve()),complete_workflow_CPU_proof_sha256=sha(proof_path),
        fixed_seed=0,physical_gpus=['4','5'],maximum_GPU_workers=2,maximum_CPU_workers=2,
        new_NN_fits=0,worker_RSS_bound_GiB=8,host_reserve_GiB=40,work_cap_GiB=64,
        real_GPU_engineering_complete=False,actual_parent_eligibility=False,
        supported_innovations=0,goal_met=False)
    frozen_json(workflow.base/'pipeline_registered.json',value)
    workflow.registered()
    return value


def worker_environment(workflow,slot,gpu):
    require(slot in (['4','5'] if gpu else ['CPU0','CPU1']),'Only registered bounded worker slots')
    return dict(os.environ,CUDA_VISIBLE_DEVICES=slot if gpu else '',MFT26IR_ROOT=workflow.plan['data_root'],
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        TORCHINDUCTOR_CACHE_DIR=str(workflow.base/'kernel_cache/inductor'),
        TRITON_CACHE_DIR=str(workflow.base/'kernel_cache/triton'),TORCH_EXTENSIONS_DIR=str(workflow.base/'kernel_cache/extensions'))


class ProcessQueue:
    def __init__(self,workflow,state,*,sleep=time.sleep,popen=subprocess.Popen,resources=True):
        self.workflow,self.state=workflow,state
        self.sleep,self.popen,self.resources=sleep,popen,resources
        self.active={}

    def save(self):write_json(self.workflow.base/'state.json',self.state)

    def admission(self,count):
        if self.resources:
            require_storage(self.workflow.work,growth_bytes=int(.1*2**30))
            require_host_memory(8*2**30,maximum_workers=count)

    def _rss(self,pid):
        try:
            for line in (Path('/proc')/str(pid)/'status').read_text().splitlines():
                if line.startswith('VmRSS:'):return int(line.split()[1])*1024
        except FileNotFoundError:return None

    def jobs(self,phase,commands,gpu,check):
        require(phase not in self.state['phases'] and commands and not self.active,
                'Preserve previous phases and actual live handles; no automatic restart')
        entries={key:dict(command=list(command),status='PENDING') for key,command in commands.items()}
        self.state['phases'][phase]=entries;self.state['status']=phase;self.save()
        failure=None
        while True:
            try:self.workflow.registered()
            except Exception as error:failure=repr(error)
            for slot,(key,process) in list(self.active.items()):
                entry=entries[key];observed=time.monotonic();code=process.poll()
                if code is None:
                    entry['last_live_monotonic']=observed
                    rss=self._rss(process.pid)
                    if rss is not None:
                        entry['observed_peak_RSS_bytes']=max(entry.get('observed_peak_RSS_bytes',0),rss)
                        if rss>8*2**30:
                            process.terminate();failure='Own F4 child exceeded8GiB; preserve healthy sibling and partial outputs'
                    continue
                del self.active[slot]
                entry.update(exit_code=code,completion_observed_UTC=datetime.now(timezone.utc).isoformat(),
                    worker_wall_upper_seconds=time.monotonic()-entry['started_monotonic'],
                    worker_wall_lower_seconds=max(0.,entry.get('last_live_monotonic',entry['started_monotonic'])-entry['started_monotonic']))
                if code:
                    entry['status']='FAILED';failure='Preserve failed F4 child: '+entry['log']
                else:
                    try:
                        require(check(key,entry) is not False,'Completed worker lacks valid complete output')
                        entry['status']='COMPLETE'
                    except Exception as error:
                        entry.update(status='FAILED_EVIDENCE',error=repr(error));failure=repr(error)
            if failure is None:
                for slot in (['4','5'] if gpu else ['CPU0','CPU1']):
                    if slot in self.active:continue
                    pending=[(k,v) for k,v in entries.items() if v['status']=='PENDING']
                    if not pending:continue
                    try:self.admission(min(2,len(pending)) if not self.active else 1)
                    except RuntimeError as error:
                        self.state['resource_wait_reason']=str(error);continue
                    key,entry=pending[0]
                    require(all(c.isalnum() or c in '_-' for c in key),'Safe registered worker ID required')
                    log=self.workflow.base/'logs'/(phase+'_'+key+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                    started=time.monotonic()
                    with log.open('xb',buffering=0) as output:
                        process=self.popen(entry['command'],cwd=ROOT,env=worker_environment(self.workflow,slot,gpu),
                            stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                    self.active[slot]=(key,process)
                    entry.update(status='RUNNING',slot=slot,log=str(log),started_monotonic=started,
                        started_UTC=datetime.now(timezone.utc).isoformat(),**process_identity(process.pid))
                    self.state.pop('resource_wait_reason',None)
            if failure:self.state['failure']=failure
            self.save()
            if failure and not self.active:raise RuntimeError(failure)
            if not self.active and all(e['status']=='COMPLETE' for e in entries.values()):return
            self.sleep(5)


class Actions:
    def __init__(self,workflow):
        self.workflow=workflow;self.paths={};self.specs={}

    def prepare(self):
        evidence=DevParentFiles().load()
        if evidence['selection']['chosen_family'] is None:return False
        frozen_json(self.workflow.base/'parent_Dev_files_verified.json',evidence)
        self.workflow.freeze_parent_snapshot()
        return True

    def specify(self,phase):
        self.paths[phase]=self.workflow.write_specification(phase)
        self.specs[phase]=read(self.paths[phase])
        return self.paths[phase]

    def commands(self,phase,trial):
        path,spec=self.paths[trial],self.specs[trial];python=sys.executable
        if phase=='GPU_ENGINEERING':return {'actual':[python,'-u','tools/probe_f4_gpu_engineering.py','--specification',str(path)]}
        if phase=='RUNTIME':return {'all_frozen':[python,'-u','tools/benchmark_fusion_memory.py','--specification',str(path)]}
        if phase.endswith('_PRODUCE'):
            return {s:[python,'-u','tools/observe_fusion_memory.py','produce','--specification',str(path),'--sequence',s]
                for s in LENGTHS[spec['split']]}
        return {c['id']:[python,'-u','tools/observe_fusion_memory.py','evaluate','--specification',str(path),'--case',c['id']]
                for c in spec['cases']}

    def check(self,phase,trial,key,entry):
        path,spec=self.paths[trial],self.specs[trial]
        if phase=='GPU_ENGINEERING':
            proof=read(self.workflow.base/'gpu_engineering.json')
            require(proof['status']=='COMPLETE_REAL_GPU_F4_SOURCE_AND_MEMORY_ENGINEERING' and
                proof['actual_GPU_tensors_verified'] is True and proof['actual_GPU_ROI_coordinate_check']['actual_CUDA'] is True and
                proof['normal_specification_sha256']==sha(path) and
                proof['pipeline_registration_sha256']==sha(self.workflow.base/'pipeline_registered.json') and
                proof['source_versions']==self.workflow.registered()['source_hashes'],'Actual registered GPU source/ROI proof required')
        elif phase=='RUNTIME':
            value=read(self.workflow.base/'individual_runtime.json')
            require(value['status']=='COMPLETE_F4_INDIVIDUAL_SOURCE_MATCHED_PREFIX_RUNTIME' and
                value['specification_sha256']==sha(path) and [r['case'] for r in value['rows']]==spec['cases'] and
                all(r['measured_frames']==168 for r in value['rows']),'Every frozen individual cost required')
        elif phase.endswith('_PRODUCE'):
            value=read(Path(spec['stream_directory'])/'sequence_manifests'/(key+'.json'))
            require(value['status']=='COMPLETE_F4_CAUSAL_SEQUENCE_OBSERVATIONS' and value['frames']==LENGTHS[spec['split']][key] and
                value['cases']==spec['cases'] and value['GT_read'] is False and value['future_frame_input'] is False and
                value['provenance']['specification_sha256']==sha(path),'Complete registered causal sequence required')
            for stream in value['streams'].values():
                require(sha(stream['track_path'])==stream['track_sha256'],'Final producer MOT changed')
                validate_mot(stream['track_path'],value['frames'],stream['track_rows'])
        else:
            case=next(c for c in spec['cases'] if c['id']==key)
            schema,_=official_schema(self.workflow.plan['evaluation']['TrackEval_root'])
            checked_metric(self.workflow.base/'evaluation'/trial/key/'metrics.json',case,metric_context(self.workflow,path),schema)

    def metric_files(self,trial):
        return {c['id']:str(self.workflow.base/'evaluation'/trial/c['id']/'metrics.json') for c in self.specs[trial]['cases']}

    def freeze_normal(self):
        schema,_=official_schema(self.workflow.plan['evaluation']['TrackEval_root'])
        return normal_dev_freeze(self.workflow.parents.plan,self.metric_files('dev_normal'),schema,
            self.workflow.base/'normal_Dev_selection.json')

    def freeze_all(self):
        schema,_=official_schema(self.workflow.plan['evaluation']['TrackEval_root'])
        return final_dev_freeze(self.workflow.parents.plan,self.workflow.base/'normal_Dev_selection.json',
            self.metric_files('dev_normal'),self.metric_files('dev_additional'),schema,self.workflow.base/'frozen_Test_case_plan.json')

    def export(self,ineligible=False):
        command=[sys.executable,'-u','tools/export_fusion_memory.py','ineligible' if ineligible else 'complete']
        log=self.workflow.base/'logs'/'FINAL_EXPORT.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('xb',buffering=0) as output:
            process=subprocess.Popen(command,cwd=ROOT,env=worker_environment(self.workflow,'CPU0',False),
                stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        identity=process_identity(process.pid)
        while process.poll() is None:
            try:
                for line in (Path('/proc')/str(process.pid)/'status').read_text().splitlines():
                    if line.startswith('VmRSS:') and int(line.split()[1])*1024>8*2**30:process.terminate()
            except FileNotFoundError:pass
            time.sleep(5)
        require(process.returncode==0,'Preserve failed final F4 exporter: '+str(log))
        value=read(REPORT.with_suffix('.json'))
        expected='F4_NOT_ELIGIBLE_FROM_ORIGINAL_COMPLETE_DEV_NOT_AN_F4_EXPERIMENT' if ineligible else 'COMPLETE_F4_FORMAL_EXPORT_REQUIRES_GLOBAL_SCIENTIFIC_ASSESSMENT'
        require(value['status']==expected and value['goal_met'] is False,'F4 report scope/status differs')
        frozen_json(self.workflow.base/'final_export_execution.json',dict(command=command,identity=identity,exit_code=0,
            log=str(log),log_sha256=sha(log),report=str(REPORT.with_suffix('.json')),report_sha256=sha(REPORT.with_suffix('.json'))))


def execute(queue,actions):
    if not actions.prepare():
        actions.export(ineligible=True)
        queue.state['status']='COMPLETE_F4_NOT_ELIGIBLE_NO_F4_EXPERIMENT';queue.save();return
    actions.specify('dev_normal')
    def phase(name,trial,gpu):
        queue.jobs(name,actions.commands(name,trial),gpu,lambda key,entry:actions.check(name,trial,key,entry))
    phase('GPU_ENGINEERING','dev_normal',True)
    phase('DEV_NORMAL_PRODUCE','dev_normal',True);phase('DEV_NORMAL_EVALUATE','dev_normal',False)
    actions.freeze_normal();actions.specify('dev_additional')
    phase('DEV_ADDITIONAL_PRODUCE','dev_additional',True);phase('DEV_ADDITIONAL_EVALUATE','dev_additional',False)
    actions.freeze_all();actions.specify('test_frozen')
    phase('TEST_PRODUCE','test_frozen',True);phase('TEST_EVALUATE','test_frozen',False);phase('RUNTIME','test_frozen',True)
    queue.state['status']=EXECUTION_FINISHED;queue.save()
    validate_execution_state(queue.state,actions.paths,actions.specs)
    actions.export()
    queue.state['status']=COMPLETE;queue.save()


def run():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','F4 controller must remain CPU-only')
    workflow=FusionMemoryWorkflow();workflow.registered();workflow.base.mkdir(parents=True,exist_ok=True)
    lock=(workflow.base/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not (workflow.base/'state.json').exists(),'Inspect prior state/real handles; no automatic duplicate controller or restart')
    identity=process_identity();frozen_json(workflow.base/'controller_identity.json',identity)
    state=dict(status=WAITING,controller_pid=identity['pid'],controller_identity=identity,
        pipeline_registration_sha256=sha(workflow.base/'pipeline_registered.json'),study_sha256=sha(STUDY),phases={},
        worker_python=sys.executable,fixed_seed=0,physical_gpus=['4','5'],maximum_GPU_workers=2,
        maximum_CPU_workers=2,goal_met=False,supported_innovations=0)
    queue=ProcessQueue(workflow,state);queue.save()
    try:
        while not workflow.predecessor_ready():
            workflow.registered();require_storage(workflow.work);time.sleep(10)
        execute(queue,Actions(workflow))
    except Exception as error:
        state.update(status='FAILED_PRESERVE_F4_EXECUTION_AND_OUTPUTS',failure=repr(error));queue.save();raise
