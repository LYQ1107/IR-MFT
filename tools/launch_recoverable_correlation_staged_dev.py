#!/usr/bin/env python3
"""Evaluate each unchanged24E removal after completion; all4 snapshots retained."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.run_pseudo_ir_comparison import dump,sha256

ROOT=Path(__file__).resolve().parents[1]


def main():
    main=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(main['work_root'])
    fusion=ROOT/'configs/perceptual_correlation_recoverability_20261123.json'
    original=ROOT/'configs/perceptual_recoverable_correlation_ablation_eval_20261123.json';spec=json.loads(original.read_text())
    source=ROOT/'tools/extract_recoverable_correlation_stage.py';path=work/'correlation_recoverability_staged_dev_state.json'
    lock=path.with_suffix('.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if path.exists():raise ValueError('Staged removals already scheduled')
    old=json.loads((work/'correlation_recoverability_ablation_dev_state.json').read_text())
    if old['status']!='SUPERSEDED_BEFORE_ANY_REMOVAL_DEV_INFERENCE':raise ValueError('Archive untouched old schedule first')
    folder=ROOT/'configs/perceptual_recoverable_correlation_staged_dev_20261123';folder.mkdir(exist_ok=False)
    spec_folder=ROOT/'configs/perceptual_recoverable_correlation_stage_eval_20261123';spec_folder.mkdir(exist_ok=False)
    jobs={};stages={}
    for stage in ['aligned','no_curriculum','blanket_counterfactual']:
        for seq in ['BT-005','SN-013']:
            check=json.loads((work/f'correlation_recoverability_removal_{stage}/dev'/seq/'source_check_8/complete.json').read_text())
            if(check['status']!='SMOKE_COMPLETE'or check['source_sha256']!=sha256(source)or len(check['outputs'])!=7
                    or not all(x['array_exact']for x in check['fresh_RGB_parity'])):
                raise ValueError('Each stage needs both actual8frame source checks')
        subset=dict(spec,models=[row for row in spec['models']if row['variant']==stage],
            scheduling='Same models/settings, now per completed24E variant. Every4checkpoint and both policies retained; never stop fitting early.')
        if len(subset['models'])!=4:raise ValueError('Unchanged four checkpoints per removal required')
        target=spec_folder/(stage+'.json');dump(target,subset)
        stages[stage]=dict(status='WAITING_FOR_FULL24_FIT',specification=str(target),sha256=sha256(target),
                           producers={seq:dict(status='PENDING')for seq in ['BT-005','SN-013']})
        for row in subset['models']:
            for policy in ['priority','raw']:
                item=dict(row,id=row['id']+('_raw'if policy=='raw'else''),output_policy=policy)
                item.update(proposal_key=item['id'],producer_directory='correlation_recoverability_removal_'+stage,
                    fusion_plan_sha256=sha256(fusion),specification_sha256=sha256(target),
                    registration='Same registered removal, all24E fitting completed before fullDev; all4snapshots/bothpolicies. Scheduling amendment only, no new settings or Test.')
                config=folder/(item['id']+'.json');dump(config,item)
                jobs[item['id']]=dict(status='PENDING',stage=stage,config=str(config),sha256=sha256(config))
    state=dict(status='RUNNING_STAGED_DEV',jobs=jobs,stages=stages,producer_source_sha256=sha256(source),
        original_specification_sha256=sha256(original),fusion_plan_sha256=sha256(fusion),Test_read=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MFT26IR_ROOT=main['data_root'])
    gpu=None;cpus={}
    while True:
        if gpu is not None:
            stage,seq,proc=gpu;code=proc.poll()
            if code is not None:stages[stage]['producers'][seq].update(status='COMPLETE'if code==0 else'FAILED',exit_code=code);gpu=None
        for key,proc in list(cpus.items()):
            code=proc.poll()
            if code is not None:jobs[key].update(status='COMPLETE'if code==0 else'FAILED',exit_code=code);del cpus[key]
        failed=any(j['status']=='FAILED'for j in jobs.values())or any(j['status']=='FAILED'for s in stages.values()for j in s['producers'].values())
        fitting=json.loads((work/'correlation_recoverability_ablation_fit_state.json').read_text())
        if gpu is None and not failed:
            for stage,info in stages.items():
                if fitting['jobs'][stage]['status']!='COMPLETE':continue
                info['status']='PRODUCING_FULL_DEV'
                pending=[seq for seq,j in info['producers'].items()if j['status']=='PENDING']
                if not pending:continue
                seq=pending[0]
                if sha256(source)!=state['producer_source_sha256']or sha256(info['specification'])!=info['sha256']:
                    raise ValueError('Frozen staged inference changed')
                command=[sys.executable,'-u',str(source),'--stage',stage,'--sequence',seq,'--specification',info['specification']]
                log=work/f'{stage}_{seq}.staged_removal_inference.log'
                with log.open('ab',buffering=0)as handle:
                    proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES='3'),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                info['producers'][seq].update(status='RUNNING',pid=proc.pid,physical_gpu='3',command=command,log=str(log));gpu=(stage,seq,proc);break
        for key,job in jobs.items():
            if len(cpus)>=2 or failed:break
            if job['status']!='PENDING'or not all(j['status']=='COMPLETE'for j in stages[job['stage']]['producers'].values()):continue
            if sha256(job['config'])!=job['sha256']:raise ValueError('Frozen removal replay changed')
            command=[sys.executable,'-u','tools/replay_recoverable_correlation.py','--config',job['config'],
                '--output',str(work/'correlation_recoverability_staged_dev'/key)]
            log=work/(key+'.staged_removal_replay.log')
            with log.open('ab',buffering=0)as handle:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            cpus[key]=proc;job.update(status='RUNNING',pid=proc.pid,command=command,log=str(log))
        for stage,info in stages.items():
            if all(j['status']=='COMPLETE'for j in info['producers'].values()):
                info['status']='COMPLETE'if all(j['status']=='COMPLETE'for j in jobs.values()if j['stage']==stage)else'RUNNING_FULL_DEV_REPLAYS'
        if all(j['status']=='COMPLETE'for j in jobs.values()):state['status']='COMPLETE'
        elif failed and gpu is None and not cpus:state['status']='FAILED'
        dump(path,state)
        if state['status']!='RUNNING_STAGED_DEV':break
        time.sleep(5)


if __name__=='__main__':main()
