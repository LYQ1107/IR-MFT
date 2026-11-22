#!/usr/bin/env python3
"""Three paired seeds of the registered association head, physical GPUs2/3."""
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
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=18);p.add_argument('--stage',default='e18');p.add_argument('--resume',action='store_true');a=p.parse_args()
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());w=Path(plan['work_root'])
    lock=(w/f'association_fit_{a.stage}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=w/f'association_fit_{a.stage}_state.json'
    if path.exists():raise ValueError('Existing association campaign: inspect before restarting')
    state=dict(status='WAITING_FOR_TRAIN_BANK',epochs=a.epochs,jobs={})
    def save():
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,path)
    while True:
        dependency=json.loads((w/'association_train_state.json').read_text())
        if dependency['status']=='COMPLETE':break
        if dependency['status']=='FAILED':state['status']='DEPENDENCY_FAILED';save();return
        save();time.sleep(5)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps','/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    state['status']='RUNNING'
    for seed in [0,1,2]:
        active={}
        for cue,gpu in [('rgb',2),('ir',3)]:
            name=f'{cue}_s{seed}';log=w/f'association_{name}_{a.stage}.fit.log'
            command=[sys.executable,'-u','tools/train_association_calibrator.py','--cue',cue,'--seed',str(seed),'--epochs',str(a.epochs)]
            if a.resume:command.append('--resume')
            with log.open('ab',buffering=0) as handle:
                proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[name]=proc;state['jobs'][name]=dict(status='RUNNING',pid=proc.pid,physical_gpu=gpu,command=command,log=str(log))
        save()
        while active:
            for name,proc in list(active.items()):
                rc=proc.poll()
                if rc is not None:state['jobs'][name].update(status='COMPLETE' if rc==0 else 'FAILED',exit_code=rc);del active[name]
            save();time.sleep(5)
        if any(j['status']=='FAILED' for j in state['jobs'].values()):state['status']='FAILED';save();return
    state['status']='COMPLETE';save()


if __name__=='__main__':main()
