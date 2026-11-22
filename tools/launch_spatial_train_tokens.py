#!/usr/bin/env python3
"""One bounded frozen-encoder worker on each permitted physical GPU."""
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
    protocol=json.loads((ROOT/plan['protocol']).read_text())
    lock=(w/'spatial_train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=w/'spatial_train_state.json'
    if path.exists():raise ValueError('Existing spatial Train extraction')
    for sequence,gpu in [('BT-001','2'),('SN-015','3')]:
        smoke=json.loads((w/f'spatial_tokens/train/{sequence}/smoke_16/complete.json').read_text())
        if smoke['status']!='SMOKE_COMPLETE' or smoke['physical_gpu']!=gpu:raise ValueError('Both allowed GPU smoke checks are required')
    state=dict(status='RUNNING',jobs={sequence:dict(status='PENDING') for sequence in protocol['train']})
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps','/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    active={}
    while True:
        for gpu,(sequence,proc) in list(active.items()):
            rc=proc.poll()
            if rc is not None:state['jobs'][sequence].update(status='COMPLETE' if rc==0 else 'FAILED',exit_code=rc);del active[gpu]
        for gpu in [2,3]:
            if gpu in active or any(x['status']=='FAILED' for x in state['jobs'].values()):continue
            sequence=next((s for s in ['MSK-002','BT-003','SN-015','SN-001','BT-001','PF-001'] if state['jobs'][s]['status']=='PENDING'),None)
            if sequence is None:continue
            log=w/f'spatial_{sequence}.process.log'
            command=[sys.executable,'-u','tools/extract_spatial_train_tokens.py','--sequence',sequence]
            with log.open('ab',buffering=0) as handle:
                proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[gpu]=(sequence,proc);state['jobs'][sequence].update(status='RUNNING',pid=proc.pid,physical_gpu=gpu,command=command,log=str(log))
        if not active:
            if all(j['status']=='COMPLETE' for j in state['jobs'].values()):state['status']='COMPLETE'
            elif any(j['status']=='FAILED' for j in state['jobs'].values()):state['status']='FAILED'
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,path)
        if state['status']!='RUNNING':break
        time.sleep(5)


if __name__=='__main__':main()
