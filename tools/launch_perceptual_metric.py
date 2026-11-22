#!/usr/bin/env python3
"""Train paired RGB/new-IR controls for seeds 0/1/2 on physical GPUs 2/3."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());w=Path(plan['work_root'])
    lock=(w/'metric_fit.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=w/'metric_fit_state.json'
    if state_path.exists():raise RuntimeError('Existing metric campaign: inspect before restarting')
    state=dict(status='RUNNING',epochs=18,jobs={})
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps','/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    def save():
        tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');os.replace(tmp,state_path)
    for seed in [0,1,2]:
        active={}
        for cue,gpu in [('rgb',2),('ir',3)]:
            name=f'{cue}_s{seed}';log=w/(name+'.metric.process.log')
            command=[sys.executable,'-u','tools/train_perceptual_metric.py','--cue',cue,'--seed',str(seed),'--epochs','18','--wait-for-bank']
            with log.open('ab',buffering=0) as handle:
                proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            state['jobs'][name]=dict(status='RUNNING',pid=proc.pid,physical_gpu=gpu,command=command,log=str(log));active[name]=proc
        save()
        while active:
            for name,proc in list(active.items()):
                code=proc.poll()
                if code is not None:
                    state['jobs'][name].update(status='COMPLETE' if code==0 else 'FAILED',exit_code=code)
                    del active[name]
            save();time.sleep(5)
        if any(j['status']=='FAILED' for j in state['jobs'].values()):state['status']='FAILED';save();return
    state['status']='COMPLETE';save()


if __name__=='__main__':main()
