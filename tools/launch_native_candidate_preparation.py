#!/usr/bin/env python3
"""Two bounded CPU workers for exact indices and derived candidate crops."""
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
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['index','crops'],required=True);a=p.parse_args()
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());w=Path(plan['work_root']);label='native_candidate_'+('index' if a.stage=='index' else 'crop')
    lock=(w/(label+'.lock')).open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);path=w/(label+'_state.json')
    if path.exists():raise ValueError('Existing candidate preparation')
    script='tools/reconstruct_native_candidate_index.py' if a.stage=='index' else 'tools/prepare_native_candidate_crops.py'
    folder='native_candidate_index' if a.stage=='index' else 'native_candidate_crops';expected='COMPLETE_EXACT_TRAIN_INDEX' if a.stage=='index' else 'COMPLETE_EXACT_TRAIN_CANDIDATE_CROPS'
    order=['MSK-002','BT-003','SN-015','SN-001','BT-001','PF-001'];jobs={name:dict(status='PENDING') for name in order};state=dict(status='RUNNING',stage=a.stage,jobs=jobs)
    for name in order:
        complete=w/folder/name/'complete.json'
        if complete.exists():
            if json.loads(complete.read_text())['status']!=expected:raise ValueError('Invalid existing candidate producer')
            jobs[name]=dict(status='COMPLETE',existing_verified_producer=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps','/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    active={}
    while True:
        for name,proc in list(active.items()):
            rc=proc.poll()
            if rc is not None:jobs[name].update(status='COMPLETE' if rc==0 else 'FAILED',exit_code=rc);del active[name]
        for name,job in jobs.items():
            if len(active)>=2 or any(x['status']=='FAILED' for x in jobs.values()):break
            if job['status']!='PENDING':continue
            if a.stage=='crops' and not (w/'native_candidate_index'/name/'complete.json').exists():continue
            command=[sys.executable,'-u',script,'--sequence',name];log=w/f'{name}.{label}.log'
            with log.open('ab',buffering=0) as handle:proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[name]=proc;job.update(status='RUNNING',pid=proc.pid,command=command,log=str(log),device='CPU')
        if not active:
            if all(x['status']=='COMPLETE' for x in jobs.values()):state['status']='COMPLETE'
            elif any(x['status']=='FAILED' for x in jobs.values()):state['status']='FAILED'
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,path)
        if state['status']!='RUNNING':break
        time.sleep(5)


if __name__=='__main__':main()
