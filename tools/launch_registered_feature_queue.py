#!/usr/bin/env python3
"""Bounded registered-IR point preparation; preserve original RGB-only fields."""
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
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','test'],required=True)
    a=p.parse_args();plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root'])
    sequences=protocol['historical_test' if a.split=='test' else 'dev']
    lock=(w/f'transport_registered_{a.split}_queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=w/f'transport_registered_{a.split}_queue.json'
    if state_path.exists():raise RuntimeError('Existing point queue; inspect before restarting')
    jobs={s:dict(status='PENDING') for s in sequences}
    state=dict(status='RUNNING',split=a.split,scope='Unlabeled classical pixel correspondences; no GT, Test accuracy or model selection',jobs=jobs)
    active={};env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps','/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    while True:
        for sequence,proc in list(active.items()):
            rc=proc.poll()
            if rc is not None:
                jobs[sequence].update(status='COMPLETE' if rc==0 else 'FAILED',exit_code=rc);del active[sequence]
        failed=any(j['status']=='FAILED' for j in jobs.values())
        for sequence,job in jobs.items():
            if failed or len(active)>=2:break
            if job['status']!='PENDING':continue
            manifest=w/'transport_registered'/a.split/sequence/'complete.json'
            if manifest.exists():
                if json.loads(manifest.read_text())['status']!='COMPLETE':raise ValueError('Malformed existing producer')
                job['status']='COMPLETE';continue
            original=w/'transport'/a.split/sequence/'complete.json'
            if not original.exists():continue
            if json.loads(original.read_text())['status']!='COMPLETE':raise ValueError('Original RGB producer is not complete')
            command=[sys.executable,'-u','tools/extract_registered_correspondences.py','--split',a.split,'--sequence',sequence]
            log=w/f'{a.split}_{sequence}.registered_transport.process.log'
            with log.open('ab',buffering=0) as handle:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[sequence]=proc;job.update(status='RUNNING',pid=proc.pid,command=command,log=str(log),device='CPU classical RGB LK')
        if all(j['status']=='COMPLETE' for j in jobs.values()):state['status']='COMPLETE'
        elif failed and not active:state['status']='FAILED'
        tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');os.replace(tmp,state_path)
        if state['status']!='RUNNING':break
        time.sleep(5)


if __name__=='__main__':main()
