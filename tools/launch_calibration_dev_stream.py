#!/usr/bin/env python3
"""Evaluate registered snapshots when complete, without stopping early fitting."""
import argparse
import fcntl
import hashlib
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--label',default='calibrated_dev')
    p.add_argument('--config-dir',default='configs/perceptual_calibrated_dev_20261123')
    p.add_argument('--fit-state',default='association_fit_e18_state.json');p.add_argument('--expected-jobs',type=int,default=36)
    a=p.parse_args()
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());w=Path(plan['work_root'])
    lock=(w/(a.label+'_stream.lock')).open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=w/(a.label+'_state.json')
    if path.exists():raise RuntimeError('Existing Dev state: inspect before restarting')
    jobs={p.stem:dict(status='PENDING',config=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted((ROOT/a.config_dir).glob('*.json'))}
    if len(jobs)!=a.expected_jobs:raise ValueError('All registered snapshots are required')
    state=dict(status='RUNNING',scope='Complete Dev evaluation at registered6/12/18 snapshots; all six fits still execute minimum18 epochs; no Test selection',jobs=jobs)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps','/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    import torch
    active={}
    while True:
        for name,proc in list(active.items()):
            rc=proc.poll()
            if rc is not None:
                jobs[name].update(status='COMPLETE' if rc==0 else 'FAILED',exit_code=rc);del active[name]
        fit=json.loads((w/a.fit_state).read_text())
        failed=any(j['status']=='FAILED' for j in jobs.values()) or fit['status'] in {'FAILED','DEPENDENCY_FAILED'}
        for name,job in jobs.items():
            if failed or len(active)>=2:break
            if job['status']!='PENDING':continue
            config=Path(job['config'])
            if hashlib.sha256(config.read_bytes()).hexdigest()!=job['sha256']:raise ValueError('Registered Dev configuration changed')
            cfg=json.loads(config.read_text());checkpoint=Path(cfg['association_checkpoint'])
            if not checkpoint.exists():continue
            try:
                candidate=torch.load(checkpoint,map_location='cpu',weights_only=False)
            except (EOFError,RuntimeError,OSError,pickle.UnpicklingError):
                # A small snapshot may be observed just before torch.save
                # writes its final ZIP directory. Retry; never use partial data.
                continue
            if candidate['cue']!=cfg['calibration_cue'] or candidate['config_sha256']!=cfg['association_plan_sha256'] or candidate['prior_plan_sha256']!=cfg['association_prior_sha256'] or candidate['capacity_plan_sha256']!=cfg['association_capacity_sha256']:raise ValueError('Snapshot provenance mismatch')
            if candidate['epoch']!=int(checkpoint.stem.replace('epoch','')):raise ValueError('Snapshot epoch mismatch')
            if cfg.get('RGB_base_selection_sha256',candidate.get('base_selection_sha256'))!=candidate.get('base_selection_sha256'):raise ValueError('Frozen RGB reference selection mismatch')
            command=[sys.executable,'-u','tools/replay_perceptual_mft.py','--config',str(config),'--split','dev','--output',str(w/a.label/name)]
            log=w/(name+'.process.log')
            with log.open('ab',buffering=0) as handle:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[name]=proc;job.update(status='RUNNING',pid=proc.pid,command=command,log=str(log),snapshot_epoch=candidate['epoch'])
        if not active:
            if failed:state['status']='FAILED'
            elif all(j['status']=='COMPLETE' for j in jobs.values()) and fit['status']=='COMPLETE':state['status']='COMPLETE'
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,path)
        if state['status']!='RUNNING':break
        time.sleep(5)


if __name__=='__main__':main()
