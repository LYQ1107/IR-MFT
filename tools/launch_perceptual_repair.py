#!/usr/bin/env python3
"""Replay registered Dev repairs after the initial screen, at most two jobs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--label',default='repair1')
    p.add_argument('--config-dir',default='configs/perceptual_repair1_20261122')
    p.add_argument('--wait-state',default='screen_state.json')
    args=p.parse_args()
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work=Path(plan['work_root']);lock=(work/(args.label+'.lock')).open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_file=work/(args.label+'_state.json')
    if state_file.exists():raise RuntimeError('Existing repair state: inspect before restarting')
    jobs={p.stem:dict(status='PENDING',config=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted((ROOT/args.config_dir).glob('*.json'))}
    state=dict(status='WAITING_FOR_INITIAL_SCREEN',jobs=jobs)
    def save():
        tmp=state_file.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');os.replace(tmp,state_file)
    while True:
        prior=json.loads((work/args.wait_state).read_text())
        if prior['status']=='COMPLETE':break
        if prior['status']=='FAILED':
            state['status']='DEPENDENCY_FAILED';save();return
        save();time.sleep(5)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps','/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    state['status']='RUNNING';active={}
    while True:
        for name,proc in list(active.items()):
            rc=proc.poll()
            if rc is not None:
                jobs[name].update(status='COMPLETE' if rc==0 else 'FAILED',exit_code=rc)
                del active[name]
        for name,job in jobs.items():
            if len(active)>=2 or any(j['status']=='FAILED' for j in jobs.values()):break
            if job['status']!='PENDING':continue
            config=Path(job['config'])
            if hashlib.sha256(config.read_bytes()).hexdigest()!=job['sha256']:raise ValueError('Registered configuration changed')
            command=[sys.executable,'-u','tools/replay_perceptual_mft.py','--config',str(config),'--split','dev','--output',str(work/args.label/name)]
            log=work/(name+'.process.log')
            with log.open('ab',buffering=0) as h:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
            active[name]=proc;job.update(status='RUNNING',pid=proc.pid,command=command,log=str(log))
        if not active:
            if all(j['status']=='COMPLETE' for j in jobs.values()):state['status']='COMPLETE'
            elif any(j['status']=='FAILED' for j in jobs.values()):state['status']='FAILED'
        save()
        if state['status']!='RUNNING':break
        time.sleep(5)
    print(json.dumps(state),flush=True)


if __name__=='__main__':main()
