#!/usr/bin/env python3
"""Prepare bounded Train crops, then fit paired native encoders on GPUs2/3."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['crops','train'],required=True);a=p.parse_args()
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());w=Path(plan['work_root'])
    label='native_crop' if a.stage=='crops' else 'native_fit_e36'
    lock=(w/(label+'.lock')).open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=w/(label+'_state.json')
    if path.exists():raise ValueError('Existing native cue campaign')
    state=dict(status='RUNNING',stage=a.stage,jobs={})
    def save():
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,path)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps','/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    if a.stage=='crops':
        order=['MSK-002','BT-003','SN-015','SN-001','BT-001','PF-001']
        state['jobs']={sequence:dict(status='PENDING') for sequence in order};active={}
        while True:
            for name,proc in list(active.items()):
                rc=proc.poll()
                if rc is not None:state['jobs'][name].update(status='COMPLETE' if rc==0 else 'FAILED',exit_code=rc);del active[name]
            for name,job in state['jobs'].items():
                if len(active)>=2 or any(x['status']=='FAILED' for x in state['jobs'].values()):break
                if job['status']!='PENDING':continue
                command=[sys.executable,'-u','tools/prepare_native_cue_crops.py','--sequence',name];log=w/f'native_{name}.crop.log'
                with log.open('ab',buffering=0) as handle:
                    proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                active[name]=proc;job.update(status='RUNNING',pid=proc.pid,command=command,log=str(log),device='CPU crop preparation')
            if not active:
                if all(x['status']=='COMPLETE' for x in state['jobs'].values()):state['status']='COMPLETE'
                elif any(x['status']=='FAILED' for x in state['jobs'].values()):state['status']='FAILED'
            save()
            if state['status']!='RUNNING':return
            time.sleep(5)
    else:
        while True:
            dependency=w/'native_crop_state.json'
            if dependency.exists():
                status=json.loads(dependency.read_text())['status']
                if status=='COMPLETE':break
                if status=='FAILED':state['status']='DEPENDENCY_FAILED';save();return
            state['status']='WAITING_FOR_TRAIN_CROPS';save();time.sleep(5)
        state['status']='RUNNING'
        for seed in [0,1,2]:
            active={}
            for cue,gpu in [('rgb',2),('ir',3)]:
                name=f'{cue}_s{seed}';command=[sys.executable,'-u','tools/train_native_cue_encoder.py','--cue',cue,'--seed',str(seed)];log=w/f'native_{name}.fit.log'
                with log.open('ab',buffering=0) as handle:
                    proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                active[name]=proc;state['jobs'][name]=dict(status='RUNNING',pid=proc.pid,command=command,log=str(log),physical_gpu=gpu)
            save()
            while active:
                for name,proc in list(active.items()):
                    rc=proc.poll()
                    if rc is not None:state['jobs'][name].update(status='COMPLETE' if rc==0 else 'FAILED',exit_code=rc);del active[name]
                save();time.sleep(5)
            if any(x['status']=='FAILED' for x in state['jobs'].values()):state['status']='FAILED';save();return
        state['status']='COMPLETE';save()


if __name__=='__main__':main()
