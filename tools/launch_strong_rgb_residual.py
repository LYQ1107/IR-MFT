#!/usr/bin/env python3
"""Matched cue residuals, restricted to physical GPUs2/3 with a memory guard."""
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
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=18);p.add_argument('--resume',action='store_true');p.add_argument('--stage',default='e18');a=p.parse_args()
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());w=Path(plan['work_root'])
    lock=(w/f'strong_rgb_fit_{a.stage}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=w/f'strong_rgb_fit_{a.stage}_state.json'
    if path.exists():raise RuntimeError('Existing strong-RGB residual campaign')
    state=dict(status='RUNNING',epochs=a.epochs,jobs={})
    def save():
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,path)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps','/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    for seed in [0,1,2]:
        while True:
            lines=subprocess.check_output(['nvidia-smi','-i','2,3','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
            free={int(s.split(',')[0]):int(s.split(',')[1]) for s in lines}
            if all(free[g]>=12*1024 for g in [2,3]):break
            state['status']='WAITING_FOR_ALLOWED_GPU_MEMORY';save();time.sleep(10)
        state['status']='RUNNING';active={}
        for cue,gpu in [('rgb',2),('ir',3)]:
            name=f'{cue}_s{seed}';command=[sys.executable,'-u','tools/train_strong_rgb_residual.py','--cue',cue,'--seed',str(seed),'--epochs',str(a.epochs)]
            if a.resume:command.append('--resume')
            log=w/f'strong_rgb_{name}_{a.stage}.fit.log'
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
