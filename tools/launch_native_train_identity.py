#!/usr/bin/env python3
"""Stream immutable final native-model feature jobs on the two allowed cards."""
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
    lock=(w/'native_train_features.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=w/'native_train_features_state.json'
    if path.exists():raise ValueError('Native feature campaign exists')
    order=[('reference',0)]+[(cue,seed) for seed in [0,1,2] for cue in ['ir','rgb']]
    state=dict(status='RUNNING',jobs={f'{cue}_s{seed}':dict(status='PENDING',cue=cue,seed=seed) for cue,seed in order})
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps','/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    active={}
    while True:
        for gpu,(name,proc) in list(active.items()):
            rc=proc.poll()
            if rc is not None:state['jobs'][name].update(status='COMPLETE' if rc==0 else 'FAILED',exit_code=rc);del active[gpu]
        for gpu in [3,2]:
            if gpu in active or any(x['status']=='FAILED' for x in state['jobs'].values()):continue
            for name,job in state['jobs'].items():
                if job['status']!='PENDING':continue
                if job['cue']!='reference' and not (w/'native_cue'/name/'complete.json').exists():continue
                command=[sys.executable,'-u','tools/extract_native_train_identity.py','--cue',job['cue'],'--seed',str(job['seed'])];log=w/f'{name}.native_features.log'
                with log.open('ab',buffering=0) as handle:
                    proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=(name,proc);job.update(status='RUNNING',pid=proc.pid,physical_gpu=gpu,command=command,log=str(log));break
        if not active:
            if all(x['status']=='COMPLETE' for x in state['jobs'].values()):state['status']='COMPLETE'
            elif any(x['status']=='FAILED' for x in state['jobs'].values()):state['status']='FAILED'
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,path)
        if state['status']!='RUNNING':break
        time.sleep(5)


if __name__=='__main__':main()
