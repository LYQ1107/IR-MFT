#!/usr/bin/env python3
"""Independent, Dev-gated one-fit factorial follow-up after the existing queue."""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.evaluate_fusion_factorial import FRESH,validate_observation_contract
from tools.evaluate_native_pyramid import lengths
from tools.fusion_factorial_selection import complete_dev_and_freeze,model_row,plan_test_reuse,prepare_authorization
from tools.fusion_factorial_workflow import REGISTRY,SCIENCE,STUDY,context,predecessor_ready,read,registered,validate_history
from tools.native_pyramid_experiment import frozen_json
from tools.neck72_runtime import require_host_memory,require_storage
from tools.run_pseudo_ir_comparison import dump,sha256


def source_files(work):
    names=set()
    for name in ['native_pyramid_pipeline_registered.json','neck72_fresh_runtime_registered.json']:
        names.update(read(Path(work)/name)['source_hashes'])
    names.update(['tools/fusion_factorial_rules.py','tools/fusion_factorial_workflow.py',
        'tools/fusion_factorial_selection.py','tools/train_fusion_factorial.py',
        'tools/evaluate_fusion_factorial.py','tools/benchmark_fusion_factorial.py',
        'tools/launch_fusion_factorial.py','tools/export_fusion_factorial.py',
        'tools/probe_fusion_factorial_workflow.py',str(SCIENCE.relative_to(ROOT)),
        'tools/export_perceptual_metrics47.py','tests/test_fusion_factorial_rules.py',
        'tests/test_fusion_factorial_workflow.py','tests/test_fusion_factorial_observations.py',
        'tests/test_fusion_factorial_runtime_prefix.py','tests/test_fusion_factorial_selection.py',
        'tests/test_fusion_factorial_controller.py','tests/test_fusion_factorial_export.py'])
    return sorted(names)


def register():
    plan,study,protocol,calibration,work,checkpoint,base=context()
    assert not (work/REGISTRY).exists()
    proof_path=work/'fusion_factorial_complete_CPU_proof.json';proof=read(proof_path)
    sources={n:sha256(ROOT/n) for n in source_files(work)}
    assert proof['status']=='COMPLETE_CPU_FACTORIAL_WORKFLOW_AND_REAL_TRAIN_CANARIES'
    assert proof['source_hashes']==sources and proof['complete_followup_pipeline_implemented']
    assert not proof['GPU_work'] and not proof['new_accuracy']
    assert len(proof['real_Train_worker_canaries'])==2
    artifacts={str(p):sha256(p) for p in [SCIENCE,proof_path,
        work/'native_pyramid_pipeline_registered.json',work/'neck72_fresh_runtime_registered.json',
        work/'fusion_factorial_training_worker_CPU_proof.json',work/'correlation_cache/calibration.json']}
    artifacts.update(proof['official_metric_schema_source_hashes'])
    table=ROOT/proof['real_existing_seed0_Dev_CSV_provenance']['source']
    assert sha256(table)==proof['real_existing_seed0_Dev_CSV_provenance']['sha256']
    artifacts[str(table)]=sha256(table)
    value=dict(status='REGISTERED_COMPLETE_FACTORIAL_PIPELINE_PENDING_DEV_GATE_AND_GPU_ENGINEERING',
        science_sha256=sha256(SCIENCE),source_hashes=sources,artifact_hashes=artifacts,
        complete_GPU_Dev_Test_pipeline_implemented=True,actual_GPU_engineering=False,
        maximum_extra_formal_fits=1,complete_parent_Dev_gate_mandatory=True,
        fixed_seed=0,additional_seed_work=False,physical_gpus=['4','5'],maximum_workers=2,
        parent_versions_unchanged=True,goal_met=False,supported_innovations=0)
    require_storage(work,int(.5*2**30));frozen_json(work/REGISTRY,value)
    print(value['status'],len(sources),'sources',flush=True)


class Queue:
    def __init__(self,work,state,data_root):
        self.work,self.state,self.data_root=Path(work),state,data_root
        self.base=self.work/'fusion_factorial';self.path=self.base/'state.json';self.active={}

    def save(self):dump(self.path,self.state)

    def jobs(self,phase,tasks,gpu,check):
        assert phase not in self.state['phases'] and not self.active
        entries={t['id']:dict(t,status='PENDING') for t in tasks}
        assert entries and len(entries)==len(tasks)
        self.state['phases'][phase]=entries;self.state['status']=phase;self.save()
        failure=None
        while True:
            try:registered(self.work)
            except Exception as error:failure=repr(error)
            for slot,(key,process) in list(self.active.items()):
                code=process.poll();entry=entries[key]
                if code is None:
                    try:
                        lines=(Path('/proc')/str(process.pid)/'status').read_text().splitlines()
                        rss=next(int(line.split()[1])*1024 for line in lines if line.startswith('VmRSS:'))
                        entry['observed_peak_RSS_bytes']=max(entry.get('observed_peak_RSS_bytes',0),rss)
                        if rss>8*2**30:
                            entry['resource_failure']='Own child exceeded8GiB RSS';process.terminate()
                            failure=entry['resource_failure']
                    except (FileNotFoundError,StopIteration):pass
                    continue
                del self.active[slot];entry['exit_code']=code
                if code:
                    entry['status']='FAILED';failure='Preserve failed child: '+entry['log']
                else:
                    try:
                        result=check(entry)
                        if result is False:raise ValueError('Required evidence failed')
                        entry['status']='COMPLETE'
                    except Exception as error:
                        entry.update(status='FAILED_EVIDENCE',error=repr(error));failure=repr(error)
            if failure is None:
                for slot in (['4','5'] if gpu else ['CPU0','CPU1']):
                    if slot in self.active:continue
                    pending=[(k,v) for k,v in entries.items() if v['status']=='PENDING' and
                        (not gpu or v.get('physical_gpu',slot)==slot)]
                    if not pending:continue
                    try:
                        require_storage(self.work,int(.5*2**30))
                        require_host_memory(8*2**30,maximum_workers=2 if not self.active and len(pending)>1 else 1)
                    except RuntimeError as error:
                        self.state['resource_wait_reason']=str(error);continue
                    key,entry=pending[0];log=self.base/'logs'/(phase+'_'+key+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                    environment=dict(os.environ,CUDA_VISIBLE_DEVICES=slot if gpu else '',
                        MFT26IR_ROOT=self.data_root,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',
                        MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                    with log.open('xb',buffering=0) as stream:
                        process=subprocess.Popen(entry['command'],cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
                            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    proc=Path('/proc')/str(process.pid)
                    ticks=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19])
                    entry.update(status='RUNNING',pid=process.pid,start_ticks=ticks,slot=slot,log=str(log),
                        command_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest())
                    self.active[slot]=(key,process);self.state.pop('resource_wait_reason',None)
            if failure:self.state['failure']=failure
            self.save()
            if failure and not self.active:raise RuntimeError(failure)
            if not self.active and all(e['status']=='COMPLETE' for e in entries.values()):return
            time.sleep(5)


def task(identifier,script,*arguments,physical=None):
    row=dict(id=identifier,command=[sys.executable,'-u','tools/'+script,*arguments])
    if physical is not None:row['physical_gpu']=physical
    return row


def run_trial(queue,models,split):
    plan,study,protocol,calibration,work,checkpoint,base=context()
    authorization=read(base/'authorization.json')
    anchor=work/'native_pyramid_evaluation'/split/'NP_source_fresh_RGB/metrics.json'
    specification=dict(status='FROZEN_FACTORIAL_OBSERVATIONS',split=split,models=[dict(FRESH)]+models,
        science_sha256=sha256(SCIENCE),registry_sha256=sha256(work/REGISTRY),
        authorization_sha256=sha256(base/'authorization.json'),stream_directory=str(base/'streams'/split),
        fresh_anchor_metrics=str(anchor),fresh_anchor_metrics_sha256=sha256(anchor))
    freeze=None
    if split=='test':
        freeze=read(base/'Test_selection_frozen.json')
        specification['Test_selection_sha256']=sha256(base/'Test_selection_frozen.json')
    validate_observation_contract(specification,split,authorization,freeze)
    spec_path=base/'specs'/(split+'.json');frozen_json(spec_path,specification)
    expected=lengths(plan,protocol,split)
    tasks=[task(seq,'evaluate_fusion_factorial.py','produce','--split',split,'--sequence',seq,
                '--specification',str(spec_path)) for seq in sorted(expected,key=lambda s:-expected[s])]
    def check_producer(entry):
        path=base/'streams'/split/'sequence_manifests'/(entry['id']+'.json');value=read(path)
        assert value['status']=='COMPLETE_FACTORIAL_CURRENT_IMAGE_STREAM_WITHOUT_GT'
        assert value['specification_sha256']==sha256(spec_path) and not value['GT_read']
        assert value['frames']==expected[entry['id']] and value['split']==split
        assert value['fresh_B1_packets_tracks_exact_parent'] and value['actual_RGB_IR_bytes_digest_exact_parent']
        assert set(value['streams'])=={r['id'] for r in specification['models']}
        for stream in value['streams'].values():assert sha256(stream['track_path'])==stream['track_sha256']
        entry['evidence_sha256']=sha256(path)
    queue.jobs(split.upper()+'_OBSERVATIONS',tasks,True,check_producer)
    tasks=[]
    for model in models:
        case=dict(id=model['id'],model=model,split=split,specification=str(spec_path),specification_sha256=sha256(spec_path))
        path=base/'cases'/split/(model['id']+'.json');frozen_json(path,case)
        tasks.append(task(model['id'],'evaluate_fusion_factorial.py','evaluate','--split',split,'--config',str(path)))
    def check_evaluation(entry):
        path=base/'evaluation'/split/entry['id']/'metrics.json';value=read(path)
        assert value['status']=='COMPLETE' and len(value['trackeval'])==39
        assert value['manifest']['frames']=={'dev':4834,'test':15985}[split]
        assert value['manifest']['config_sha256']==sha256(base/'cases'/split/(entry['id']+'.json'))
        entry['evidence_sha256']=sha256(path)
    queue.jobs(split.upper()+'_OFFICIAL',tasks,False,check_evaluation)


def execute(queue,registration):
    """Kept separate so failed canaries/incomplete Dev can be exercised on CPU."""
    authorization=prepare_authorization(registration);base=queue.base
    if authorization['status']=='FACTORIAL_EXTENSION_NOT_TRIGGERED_BY_COMPLETE_PARENT_DEV':
        frozen_json(base/'not_triggered.json',authorization)
        queue.state['status']='COMPLETE_FACTORIAL_EXTENSION_NOT_TRIGGERED_BY_DEV';queue.save()
        return authorization
    def check_probe(entry):
        value=read(base/'engineering.json');assert value['status']=='COMPLETE_REAL_GPU_JOINT_FACTORIAL_ENGINEERING'
        assert value['authorization_sha256']==sha256(base/'authorization.json')
        assert value['record']['initial_native_sha256']==authorization['initial_native_sha256']
        assert value['record']['inference_B1_frames']==8
        assert len(value['record']['source_zero_missing_checks'])==8
        assert all(r['direct_B1_packets_exact'] and r['zero_full_head_exact'] and r['trained_missing_full_head_exact']
                   for r in value['record']['source_zero_missing_checks'])
        record=value['record']
        assert record['delivered_Train_frames']==8 and record['actual_updates']==2
        assert len(record['optimizer_rows'])==2 and all(r['optimizer_updated'] for r in record['optimizer_rows'])
        assert record['optimizer_rows'][0]['native_stem_gradient_L1']==0 and record['optimizer_rows'][1]['native_stem_gradient_L1']>0
        assert record['frozen_RGB_parameters_and_BN_unchanged_before_explicit_head_precision_conversion']
        assert record['inference_RGB_state_unchanged'] and value['peak_process_RSS_GiB']<=8
    queue.jobs('GPU_ENGINEERING',[task('joint00','train_fusion_factorial.py','gpu-probe',physical='4')],True,check_probe)
    def check_fit(entry):
        value=read(base/'fit_complete.json');assert value['status']=='COMPLETE_ONE_JOINT_DISABLE_FULL_TRAIN_FIT'
        assert value['stage']==authorization['stage'] and value['authorization_sha256']==sha256(base/'authorization.json')
        assert value['progress_sha256']==sha256(base/'fit/progress.json')
        progress=read(base/'fit/progress.json')
        assert progress['completed_full_Train_epochs']==progress['registered_through_epoch']==authorization['stage']
        assert progress['frozen_RGB_exact_unchanged']
        metadata=validate_history(dict(completed_epochs=progress['completed_full_Train_epochs'],
            history=progress['history'],metadata=progress['metadata']),authorization['stage'],read(STUDY))
        assert metadata['initial_native_sha256']==authorization['initial_native_sha256']
    queue.jobs('FULL_TRAIN',[task('joint00','train_fusion_factorial.py','fit',physical='4')],True,check_fit)
    models=[model_row('00',e,authorization['cell_fits']['00'],authorization['mode']) for e in range(1,authorization['stage']+1)]
    run_trial(queue,models,'dev')
    freeze=complete_dev_and_freeze(authorization)
    coverage=plan_test_reuse(freeze)
    run_trial(queue,coverage['missing_models'],'test')
    def check_runtime(entry):
        value=read(base/'individual_runtime.json')
        assert value['status']=='COMPLETE_FROZEN_FACTORIAL_INDIVIDUAL_PREFIX_RUNTIME'
        assert value['Test_selection_sha256']==sha256(base/'Test_selection_frozen.json')
        assert {r['id'] for r in value['rows']}=={FRESH['id']}|{r['id'] for r in freeze['selected_unique_models']}
    queue.jobs('INDIVIDUAL_RUNTIME',[task('cost','benchmark_fusion_factorial.py',physical='4')],True,check_runtime)
    queue.state['status']='COMPLETE_FACTORIAL_EXECUTION_PENDING_SCIENTIFIC_EXPORT';queue.save()
    return freeze


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','The controller is CPU-only'
    plan,study,protocol,calibration,work,checkpoint,base=context();registration=registered(work)
    base.mkdir(parents=True,exist_ok=True)
    lock=(base/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=base/'state.json';assert not path.exists(),'Preserve and inspect any prior controller before resuming'
    proc=Path('/proc')/str(os.getpid())
    identity=dict(pid=os.getpid(),start_ticks=int((proc/'stat').read_text().rsplit(')',1)[1].split()[19]),
        command_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest())
    frozen_json(base/'controller_identity.json',identity)
    state=dict(status='WAITING_FOR_COMPLETE_PRIOR_RUNTIME_AND_ACTUAL_PROCESS_EXITS',controller_pid=os.getpid(),
        science_sha256=sha256(SCIENCE),registry_sha256=sha256(work/REGISTRY),phases={},
        physical_gpus=['4','5'],maximum_workers=2,fixed_seed=0,goal_met=False,supported_innovations=0)
    queue=Queue(work,state,plan['data_root']);queue.save()
    try:
        while not predecessor_ready(work):registered(work);time.sleep(10)
        execute(queue,registration)
        subprocess.run([sys.executable,'tools/export_fusion_factorial.py'],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
        if state['status']!='COMPLETE_FACTORIAL_EXTENSION_NOT_TRIGGERED_BY_DEV':
            state['status']='COMPLETE_FACTORIAL_STUDY_REQUIRES_SCIENTIFIC_ASSESSMENT';queue.save()
    except Exception as error:
        state.update(status='FAILED_PRESERVE_FACTORIAL_STUDY',error=repr(error));queue.save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['register','run'])
    {'register':register,'run':run}[parser.parse_args().action]()
