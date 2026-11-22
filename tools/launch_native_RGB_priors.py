#!/usr/bin/env python3
"""Two CPU workers for the six deterministic ordinary RGB calibrations."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());w=Path(plan['work_root']);path=w/'native_RGB_prior_state.json'
    lock=(w/'native_RGB_prior.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if path.exists():raise ValueError('RGB query priors already scheduled')
    jobs={f'{front}_s{seed}':dict(status='PENDING',frontend=front,seed=seed) for front in ['new','rgb'] for seed in [0,1,2]};state=dict(status='RUNNING',jobs=jobs);active={}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps','/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    while True:
        for name,proc in list(active.items()):
            rc=proc.poll()
            if rc is not None:jobs[name].update(status='COMPLETE' if rc==0 else 'FAILED',exit_code=rc);del active[name]
        for name,job in jobs.items():
            if len(active)>=2 or any(x['status']=='FAILED' for x in jobs.values()):break
            if job['status']!='PENDING':continue
            command=[sys.executable,'-u','tools/fit_native_RGB_prior.py','--frontend',job['frontend'],'--seed',str(job['seed'])];log=w/(name+'.native_RGB_prior.log')
            with log.open('ab',buffering=0) as handle:proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[name]=proc;job.update(status='RUNNING',pid=proc.pid,command=command,log=str(log))
        if not active:
            if all(x['status']=='COMPLETE' for x in jobs.values()):state['status']='COMPLETE'
            elif any(x['status']=='FAILED' for x in jobs.values()):state['status']='FAILED'
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,path)
        if state['status']!='RUNNING':break
        time.sleep(5)


if __name__=='__main__':main()
