#!/usr/bin/env python3
"""Bounded unlabeled feature preparation, exclusively on physical GPUs 2/3."""
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
    p=argparse.ArgumentParser();p.add_argument('--plan',default='configs/perceptual_mft_20261122.json')
    p.add_argument('--split',choices=['dev','test'],required=True)
    p.add_argument('--workers-per-device',type=int,choices=[1,2],default=1)
    a=p.parse_args();plan=json.loads((ROOT/a.plan).read_text());protocol=json.loads((ROOT/plan['protocol']).read_text())
    work=Path(plan['work_root']);lock=(work/f'features_{a.split}.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_file=work/f'features_{a.split}_queue.json'
    sequences=protocol['historical_test' if a.split=='test' else 'dev']
    state=dict(status='RUNNING',split=a.split,scope='Unlabeled feature computation only; no GT or Test accuracy selection',jobs={})
    for seq in sequences:
        complete=work/'features'/a.split/seq/'complete.json'
        state['jobs'][seq]=dict(status='COMPLETE' if complete.exists() else 'PENDING')
    active={}
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps','/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    while True:
        for seq,process in list(active.items()):
            code=process.poll()
            if code is not None:
                state['jobs'][seq].update(status='COMPLETE' if code==0 else 'FAILED',exit_code=code,finished_unix=time.time())
                del active[seq]
        failed=any(x['status']=='FAILED' for x in state['jobs'].values())
        for seq,job in state['jobs'].items():
            if failed or job['status']!='PENDING':continue
            used=[state['jobs'][s]['physical_gpu'] for s in active]
            available=[g for g in [2,3] if used.count(g)<a.workers_per_device]
            if not available:continue
            gpu=min(available,key=used.count);log=work/f'feature_{a.split}_{seq}.process.log'
            command=[sys.executable,'-u','tools/extract_perceptual_mft_features.py','--plan',a.plan,'--split',a.split,'--sequence',seq]
            with log.open('ab',buffering=0) as h:
                proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
            active[seq]=proc;job.update(status='RUNNING',pid=proc.pid,physical_gpu=gpu,command=command,log=str(log),started_unix=time.time())
        if failed and not active:state['status']='FAILED'
        if all(x['status']=='COMPLETE' for x in state['jobs'].values()):state['status']='COMPLETE'
        tmp=state_file.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');os.replace(tmp,state_file)
        if state['status']!='RUNNING':break
        time.sleep(5)
    print(json.dumps(state),flush=True)


if __name__=='__main__':main()
