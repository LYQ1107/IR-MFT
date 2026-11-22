#!/usr/bin/env python3
"""Build matched native-query features with two bounded CPU workers."""
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
    lock=(w/'native_neck_mismatch_query_bank.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);path=w/'native_neck_mismatch_query_bank_state.json'
    if path.exists():raise ValueError('Native query bank already scheduled')
    order=['MSK-002','BT-003','SN-015','SN-001','BT-001','PF-001'];jobs={sequence:dict(status='PENDING') for sequence in order};state=dict(status='RUNNING',jobs=jobs);active={}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps','/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    while True:
        for name,proc in list(active.items()):
            rc=proc.poll()
            if rc is not None:jobs[name].update(status='COMPLETE' if rc==0 else 'FAILED',exit_code=rc);del active[name]
        for name,job in jobs.items():
            if len(active)>=2 or any(j['status']=='FAILED' for j in jobs.values()):break
            if job['status']!='PENDING' or not all((w/'native_neck_candidate_features'/name/cue/'complete.json').exists() for cue in ['rgb','ir']):continue
            command=[sys.executable,'-u','tools/build_native_neck_mismatch_bank.py','--sequence',name];log=w/(name+'.native_neck_mismatch_query_bank.log')
            with log.open('ab',buffering=0) as handle:proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[name]=proc;job.update(status='RUNNING',pid=proc.pid,command=command,log=str(log))
        dependency=json.loads((w/'native_neck_candidate_features_state.json').read_text())
        if not active:
            if all(j['status']=='COMPLETE' for j in jobs.values()):state['status']='COMPLETE'
            elif any(j['status']=='FAILED' for j in jobs.values()) or dependency['status']=='FAILED':state['status']='FAILED'
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,path)
        if state['status']!='RUNNING':break
        time.sleep(5)


if __name__=='__main__':main()
