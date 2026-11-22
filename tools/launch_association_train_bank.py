#!/usr/bin/env python3
"""Build only Train correspondence supervision, one worker per allowed GPU."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root'])
    lock=(w/'association_train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=w/'association_train_state.json'
    if path.exists():raise RuntimeError('Existing training-bank campaign')
    # Dense fish scenes first, so the last GPU need not wait for a long scene.
    order=['MSK-002','SN-015','SN-001','BT-001','BT-003','PF-001']
    if set(order)!=set(protocol['train']):raise ValueError('Frozen Train mismatch')
    jobs={s:dict(status='PENDING') for s in order};state=dict(status='RUNNING',scope='Train-only supervised pair preparation, no Dev/Test GT or fitting rows',jobs=jobs)
    active={};env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps','/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    while True:
        for sequence,process in list(active.items()):
            rc=process.poll()
            if rc is not None:
                jobs[sequence].update(status='COMPLETE' if rc==0 else 'FAILED',exit_code=rc);del active[sequence]
        failed=any(j['status']=='FAILED' for j in jobs.values())
        for sequence,job in jobs.items():
            if failed or job['status']!='PENDING':continue
            used={jobs[s]['physical_gpu'] for s in active};available=[g for g in [2,3] if g not in used]
            if not available:break
            manifest=w/'association_train'/sequence/'complete.json'
            if manifest.exists():
                if json.loads(manifest.read_text())['status']!='COMPLETE_TRAIN_ONLY':raise ValueError('Incomplete existing bank')
                job['status']='COMPLETE';continue
            gpu=available[0];log=w/(sequence+'.association_train.log')
            command=[sys.executable,'-u','tools/extract_association_train_pairs.py','--sequence',sequence]
            with log.open('ab',buffering=0) as handle:
                proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[sequence]=proc;job.update(status='RUNNING',pid=proc.pid,physical_gpu=gpu,command=command,log=str(log))
        if all(j['status']=='COMPLETE' for j in jobs.values()):state['status']='COMPLETE'
        elif failed and not active:state['status']='FAILED'
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,path)
        if state['status']!='RUNNING':break
        time.sleep(5)


if __name__=='__main__':main()
