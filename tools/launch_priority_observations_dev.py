#!/usr/bin/env python3
"""Complete all registered priority-integration controls without GPU forwards."""
import argparse
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
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['plain','regions','causal'],required=True);a=p.parse_args()
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(plan['work_root'])
    config_path=ROOT/'configs/perceptual_foveated_priority_20261123.json';config=json.loads(config_path.read_text())
    parent=json.loads((ROOT/config['parents'][a.family]['plan']).read_text());modes=parent.get('attention_modes',['rgb','ir'])
    label='foveated_priority_'+a.family+'_dev';state_path=work/(label+'_state.json')
    lock=state_path.with_suffix('.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if state_path.exists():raise ValueError('Priority family already scheduled')
    directory='foveated_priority_'+a.family+'_proposals'
    expected=2*(len(modes)*len(parent['attention_scales'])*2+1)
    for seq in ['BT-005','SN-013']:
        smoke=json.loads((work/directory/'dev'/seq/'smoke_4/complete.json').read_text())
        if (smoke['status']!='SMOKE_COMPLETE' or smoke['config_sha256']!=sha256(config_path)
                or len(smoke['outputs'])!=expected or not smoke['high_global_coordinates_and_scores_immutable']):
            raise ValueError('Both exact reconstruction/source checks required')
    folder=ROOT/f'configs/perceptual_priority_{a.family}_dev_20261123';folder.mkdir(exist_ok=False)
    keys=[f'{arm}_{cue}_{scale}_k{count}_priority'for arm in ['rgb','new']for cue in modes
          for scale in parent['attention_scales']for count in [2,4]]
    keys+=[f'{arm}_uniform_k4_priority'for arm in ['rgb','new']]+['baseline_rgb','baseline_new']
    jobs={}
    for key in keys:
        name=f'V_{a.family}_'+key;path=folder/(name+'.json')
        dump(path,dict(id=name,proposal_key=key,fovea_plan='configs/perceptual_foveated_priority_20261123.json',
             fovea_plan_sha256=sha256(config_path),producer_directory=directory,
             registration='All parent cue/scale/count/global-front controls with the same high-confidence global observation protection. Original ByteTrack and thresholds, no Test selection.'))
        jobs[name]=dict(status='PENDING',config=str(path),sha256=sha256(path))
    state=dict(status='RECOMBINING_EXACT_RAW_OBSERVATIONS',family=a.family,Test_started=False,jobs=jobs,producers={})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',MFT26IR_ROOT=plan['data_root'],OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    active={}
    for seq in ['BT-005','SN-013']:
        command=[sys.executable,'-u','tools/recombine_priority_observations.py','--family',a.family,'--sequence',seq]
        log=work/(seq+'.priority_'+a.family+'_full_dev.log')
        with log.open('ab',buffering=0)as handle:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        active[seq]=proc;state['producers'][seq]=dict(status='RUNNING',pid=proc.pid,command=command,log=str(log))
    while active:
        for key,proc in list(active.items()):
            code=proc.poll()
            if code is not None:state['producers'][key].update(status='COMPLETE'if code==0 else'FAILED',exit_code=code);del active[key]
        dump(state_path,state);time.sleep(5)
    if any(j['status']=='FAILED'for j in state['producers'].values()):
        state['status']='FAILED';dump(state_path,state);return
    state['status']='RUNNING'
    while True:
        for key,proc in list(active.items()):
            code=proc.poll()
            if code is not None:jobs[key].update(status='COMPLETE'if code==0 else'FAILED',exit_code=code);del active[key]
        for key,job in jobs.items():
            if len(active)>=2 or any(j['status']=='FAILED'for j in jobs.values()):break
            if job['status']!='PENDING':continue
            if sha256(job['config'])!=job['sha256']:raise ValueError('Registered priority control changed')
            command=[sys.executable,'-u','tools/replay_foveated_observations.py','--config',job['config'],'--split','dev','--output',str(work/label/key)]
            log=work/(label+'_'+key+'.log')
            with log.open('ab',buffering=0)as handle:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[key]=proc;job.update(status='RUNNING',pid=proc.pid,command=command,log=str(log))
        if not active:
            if all(j['status']=='COMPLETE'for j in jobs.values()):state['status']='COMPLETE'
            elif any(j['status']=='FAILED'for j in jobs.values()):state['status']='FAILED'
        dump(state_path,state)
        if state['status']!='RUNNING':break
        time.sleep(5)


if __name__=='__main__':main()
