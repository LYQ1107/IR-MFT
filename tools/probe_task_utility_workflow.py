#!/usr/bin/env python3
"""CPU-only experiment-graph/real-process checks; never a GPU experiment proof."""
import argparse
from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.native_pyramid_experiment import frozen_json,read
from tools.neck72_runtime import require_host_memory,require_storage
from tools.run_pseudo_ir_comparison import dump,sha256
import tools.task_utility_controller as controller
import tools.task_utility_selection as selection
from tools.task_utility_design import CANONICAL,FRESH
from tools.task_utility_pipeline import CPU_PROOF,current_source_hashes
from tools.task_utility_queue import Queue,process_identity
from tools.task_utility_study import STUDY,REGISTRY,arm_path,context,make_arm,predecessor_ready
from tools.fusion_memory_observation import LENGTHS


class ArtificialQueue:
    """Record real command recipes,write only clearly isolated artificial fixtures."""
    def __init__(self,work,base,study,registration,stop):
        self.work,self.base,self.study,self.registration=Path(work),Path(base),study,registration
        self.stop=stop;self.state=dict(artificial_fixture=True,phases={},source_scope='Graph simulation;no NN/GPUcommands launched')
        self.commands=[];self.metric_rows={};self.Test_ever_read=False
    def save(self):dump(self.base/'artificial_state.json',self.state)
    def metric(self,item):
        case=read(item['config']);return self.metric_rows[case['split'],case['id']]
    def original_rows(self,plan,protocol,base,cases):
        rows=[];files={}
        for item in cases:
            value=self.metric(item)
            if value['split']!='dev':raise ValueError('Artificial Test data cannot choose Train budget')
            rows.append(value)
            for path in [item['config'],value['source_metrics']]:files[str(path)]=sha256(path)
        return rows,files
    def _score(self,model):
        if model.get('baseline') or model.get('intervention')=='unavailable':return 60.
        epoch=model['epoch'];gain=(1 if epoch>=4 and self.stop>=9 else 0)+(1 if epoch>=7 and self.stop==12 else 0)
        variants=['no_aux_labels','foreground_labels','utility_labels','foreground_utility_labels']
        return {'rgb':60.,'gray':59.,'ir':45.}[model['cue']]+gain+.01*variants.index(model['variant'])-(.3 if model['intervention']=='IR_image_spatial_roll' else 0)
    def _metrics(self,entry):
        case=read(entry['config']);model=case['model'];score=self._score(model)
        path=self.base/'evaluation'/case['split']/case['id']/'metrics.json'
        data=dict(artificial_fixture=True,metrics39={'HOTA':score},combined47={'HOTA':score},
            per_sequence47=[{'artificial_HOTA':score}],official_CSV_sha256=hashlib.sha256(str(score).encode()).hexdigest())
        dump(path,data)
        self.metric_rows[case['split'],case['id']]=dict(model,split=case['split'],
            sequences=list(LENGTHS[case['split']]),frames=sum(LENGTHS[case['split']].values()),
            selection_metrics=dict(HOTA=score,AssA=40.,IDF1=50.),metrics39=data['metrics39'],
            source_metrics=str(path),source_metrics_sha256=sha256(path),official_CSV_sha256=data['official_CSV_sha256'])
    def _observe(self,entry):
        command=entry['command'];specpath=Path(command[command.index('--specification')+1]);spec=read(specpath)
        sequence=entry['id'];count=LENGTHS[spec['split']][sequence];directory=Path(spec['stream_directory']);streams={}
        for model in spec['models']:
            path=directory/model['id']/'track_results'/(sequence+'.txt');path.parent.mkdir(parents=True,exist_ok=True);path.write_text('')
            streams[model['id']]=dict(track_path=str(path),track_sha256=sha256(path),track_rows=0,frames=count)
        gates={m['id']:dict(frames=count,available_frames=count,missing_frames=0,levels=[{}, {}, {}])
            for m in spec['models'] if not m.get('baseline')}
        value=dict(artificial_fixture=True,status='COMPLETE_TASK_UTILITY_B1_SOURCE_MATCHED_OBSERVATIONS_WITHOUT_GT',
            split=spec['split'],sequence=sequence,frames=count,complete_source_sequence_frames=count,
            formal_Dev_Test_sequence_coverage=True,models=spec['models'],GT_read=False,future_frame_input=False,
            previous_outputs_rewritten=False,independent_input_batch_size=1,feature_or_detection_disk_cache=False,
            provenance=dict(source_versions=self.registration['source_hashes'],specification_sha256=sha256(specpath),
                pipeline_registration_sha256=spec['pipeline_registration_sha256'],study_sha256=sha256(STUDY),
                physical_gpu=entry['slot'],parent_fresh_packets_and_tracks_exact=True),streams=streams,gate_statistics=gates)
        dump(directory/'sequence_manifests'/(sequence+'.json'),value)
    def jobs(self,phase,tasks,gpu,check):
        if phase in self.state['phases'] or not tasks:raise ValueError('Artificial graph phase repeated/empty')
        rows={t['id']:dict(t,status='ARTIFICIAL_NOT_LAUNCHED',slot=['4','5'][i%2] if gpu else 'CPU0',
            argv_identity_observed=False,actual_GPU_work=False) for i,t in enumerate(tasks)}
        self.state['phases'][phase]=rows;self.commands.extend(t['command'] for t in tasks)
        if phase.startswith('TRAIN_'):
            stage=int(phase.split('_')[1])
            for fit in self.study['normal_fits']:
                directory=self.base/'fits'/fit['id'];directory.mkdir(parents=True,exist_ok=True)
                for epoch in range(1,stage+1):
                    path=directory/f'epoch_{epoch:02d}.pt'
                    if not path.exists():path.write_bytes(f'ARTIFICIAL_NOT_A_TORCH_MODEL:{fit["id"]}:{epoch}'.encode())
            for row in rows.values():check(row)
        elif phase.endswith('_NN'):
            for row in rows.values():self._observe(row);check(row)
        elif phase.endswith('_CANONICAL'):
            for entry in rows.values():
                cmd=entry['command'];specpath=Path(cmd[cmd.index('--specification')+1]);spec=read(specpath)
                for seq,n in LENGTHS[spec['split']].items():
                    dump(Path(spec['stream_directory'])/'sequence_manifests'/(seq+'.canonical.json'),
                        dict(artificial_fixture=True,status='COMPLETE_TASK_UTILITY_CANONICAL_REPLAY_WITHOUT_GT',frames=n,
                            specification_sha256=sha256(specpath),GT_read=False,future_frame_input=False,previous_outputs_rewritten=False))
                check(entry)
        elif phase.endswith('_OFFICIAL'):
            for entry in rows.values():self._metrics(entry);check(entry)
        elif phase in ['NORMAL_GPU_ENGINEERING','INDIVIDUAL_RUNTIME']:
            for entry in rows.values():check(entry)
        else:raise ValueError('Unexpected synthetic graph phase '+phase)
        self.save()


def simulate_graph(root,study,stop):
    """Actual orchestration/recipe/selector code with artificial data services."""
    root=Path(root);base=root/'task_utility';base.mkdir(parents=True);arms={}
    for fit in study['normal_fits']:
        path=arm_path(base,fit['id']);frozen_json(path,make_arm(study,fit['id'],fit['variant'],fit['cue']));arms[fit['id']]=sha256(path)
    registration=dict(status='ARTIFICIAL_NOT_REGISTERED_FOR_GPU',normal_arms=arms,source_hashes={'fixture':'NOT_REAL_SOURCES'},study_sha256=sha256(STUDY))
    frozen_json(base/REGISTRY,registration);queue=ArtificialQueue(root,base,study,registration,stop)
    with ExitStack() as stack:
        stack.enter_context(patch.object(controller,'verify_probe',lambda *args:{'artificial_only':True}))
        stack.enter_context(patch.object(controller,'verify_fit',lambda *args:{'artificial_only':True}))
        stack.enter_context(patch.object(controller,'verify_runtime',lambda *args:{'artificial_only':True}))
        stack.enter_context(patch.object(controller,'verify_observations',lambda *args:{'artificial_only':True}))
        stack.enter_context(patch.object(controller,'metric_row',lambda base,item,protocol:queue.metric(item)))
        stack.enter_context(patch.object(selection,'original_Dev_rows',queue.original_rows))
        controller.execute(queue,{},study,dict(dev=list(LENGTHS['dev']),historical_test=list(LENGTHS['test'])) ,registration)
    frozen=read(base/'Test_selection_frozen.json');stage=read(base/f'complete_Dev_stage_{stop:02d}.json')
    if stage['budget_decision']['extend_all'] or frozen['through_epoch']!=stop:raise ValueError('Synthetic common budget stopped incorrectly')
    fits=[p for p in queue.state['phases'] if p.startswith('TRAIN_')]
    if fits!=[f'TRAIN_{s:02d}' for s in [6,9,12] if s<=stop]:raise ValueError('The actual controller skipped/added a common Train stage')
    if len(frozen['cases'])!=26 or len(stage['cases'])!=20*stop+2:raise ValueError('Complete negative conditions/epoch curves were dropped')
    ids={c['id'] for c in frozen['cases']}
    if sum(name.startswith('TU_') for name in ids)!=26:raise ValueError('Missing original frozen Test identities')
    return dict(artificial_fixture=True,stopped_at=stop,normal_arm_hashes=arms,
        stages=fits,Dev_case_count=len(stage['cases']),frozen_Test_case_count=26,
        runtime_routes=[(t['id'],t['NN_batch_size']) for t in queue.state['phases']['INDIVIDUAL_RUNTIME'].values()],
        phase_job_counts={k:len(v) for k,v in queue.state['phases'].items()},
        command_intents_recorded_not_launched=queue.commands,artificial_base=str(base),
        actual_GPU_work=False,actual_NN_or_Train_Test_evaluation=False,formal_worker_validators_not_bypassed_in_production=True)


def verify_real_CPU_queue(root):
    root=Path(root);state=dict(artificial_fixture=True,phases={});completed=[]
    queue=Queue(root,root,state,'unused',lambda:True,poll_seconds=.01,storage_guard=lambda *a,**k:None,
        memory_guard=lambda *a,**k:None,force_CPU_for_artificial_test=True)
    tasks=[]
    for i in range(2):
        output=root/f'worker_{i}.json'
        program="import json,os,pathlib,time;time.sleep(.2);pathlib.Path("+repr(str(output))+ ").write_text(json.dumps({'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')}));time.sleep(.1)"
        tasks.append(dict(id='actual_CPU_'+str(i),command=[sys.executable,'-u','-c',program]))
    def check(item):
        number=item['id'].rsplit('_',1)[1];value=read(root/f'worker_{number}.json')
        if value['CUDA_VISIBLE_DEVICES']!='':raise ValueError('Artificial queue touched GPU')
        completed.append(item['id'])
    queue.jobs('CPU_LIFECYCLE',tasks,False,check)
    if len(completed)!=2 or not all(v['argv_identity_observed'] for v in state['phases']['CPU_LIFECYCLE'].values()):
        raise ValueError('Actual CPU subprocess identities/exit/evidence checks did not finish')
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pytest-receipt',type=Path,required=True);args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU graph proof only')
    if torch.cuda.is_initialized():raise ValueError('CPU workflow proof cannot start with CUDA initialized')
    started=time.monotonic();sources=current_source_hashes();unit=read(args.pytest_receipt)
    if unit['exit_code']!=0 or not unit['all_sources_unchanged'] or sha256(unit['log'])!=unit['log_sha256']:
        raise ValueError('Actual current-source unit receipt required')
    if any(sources[p]!=s for p,s in unit['source_hashes'].items()):raise ValueError('Unit-tested controller source changed')
    plan,study,protocol,calibration,work,checkpoint,base=context();require_host_memory(8*2**30,maximum_workers=1);require_storage(work)
    output=work/Path(CPU_PROOF).parent;output.mkdir(exist_ok=False);identity=process_identity(os.getpid())
    graphs=[simulate_graph(output/f'artificial_stop_{stage}',study,stage) for stage in [6,9,12]]
    lifecycle=verify_real_CPU_queue(output/'real_CPU_subprocesses')
    if torch.cuda.is_initialized():raise ValueError('CPU workflow proof must never initialize CUDA')
    if any(g['actual_GPU_work'] for g in graphs) or any(g['normal_arm_hashes']!=graphs[0]['normal_arm_hashes'] for g in graphs):
        raise ValueError('The graph fixtures must never create GPU execution qualification')
    if current_source_hashes()!=sources:raise ValueError('Source closure changed during CPU graph proof')
    artifacts={str(args.pytest_receipt.resolve()):sha256(args.pytest_receipt),unit['log']:sha256(unit['log'])}
    for name in ['WORKER_CPU_ENGINEERING','RUNTIME_COMPONENTS']:
        path=ROOT/f'docs/fusion_research/PERCEPTUAL_MFT_20261125_TASK_UTILITY_{name}.json';artifacts[str(path)]=sha256(path)
    result=dict(status='COMPLETE_TASK_UTILITY_CPU_WORKFLOW_GRAPH_NOT_GPU_EXECUTION',identity=identity,
        source_hashes=sources,study_sha256=sha256(STUDY),normal_arm_hashes=graphs[0]['normal_arm_hashes'],
        component_artifact_hashes=artifacts,all_registered_budget_branches_exercised=True,
        real_CPU_subprocess_queue_lifecycle_verified=True,graphs=graphs,actual_CPU_process_lifecycle=lifecycle,
        actual_predecessor_ready=predecessor_ready(work),new_GPU_work=False,formal_GPU_eligibility=False,
        formal_fit_started=False,new_tracking_accuracy=False,goal_met=False,supported_innovations=0,
        seconds=time.monotonic()-started,
        scope='Actual orchestration/recipe/common-budget/Devfreeze/Testplan/cost-order graph on isolated artificial data;12fit/GPUvalidators and metrics replaced only inside fixture context. '
            'Actual2CPUchildprocess identities/exit/evidence verified. No neural qualification,Train/Dev/Testaccuracy orcompleteGPUexecution. Actual prior worker/runtime NN evidence remains separate.')
    dump(output/'complete.json',result);print(json.dumps(dict(status=result['status'],seconds=result['seconds'],sources=len(sources))),flush=True)


if __name__=='__main__':main()
