#!/usr/bin/env python3
"""Persistent two-device extraction and CPU Dev screen with explicit dependencies."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write(path, obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',default='configs/perceptual_mft_20261122.json')
    a=p.parse_args();plan=json.loads((ROOT/a.plan).read_text());work=Path(plan['work_root'])
    state_path=work/'screen_state.json'
    if state_path.exists():raise RuntimeError('Existing campaign: inspect state, do not double-launch')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps','/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    jobs=[]
    for gpu,seq in [(2,'BT-005'),(3,'SN-013')]:
        jobs.append(dict(id='extract_'+seq,resource=gpu,dependencies=[],
                         command=[sys.executable,'-u','tools/extract_perceptual_mft_features.py','--plan',a.plan,'--split','dev','--sequence',seq]))
    for config in sorted((ROOT/'configs/perceptual_screen_20261122').glob('*.json')):
        c=json.loads(config.read_text());deps=[] if c['backend']=='byte' else ['extract_BT-005','extract_SN-013']
        jobs.append(dict(id=config.stem,resource='cpu',dependencies=deps,
                         command=[sys.executable,'-u','tools/replay_perceptual_mft.py','--plan',a.plan,'--config',str(config),'--split','dev','--output',str(work/'screen'/config.stem)]))
    state=dict(status='RUNNING',started_unix=time.time(),jobs={j['id']:dict(j,status='PENDING') for j in jobs})
    active={}
    while True:
        for name,proc in list(active.items()):
            code=proc.poll()
            if code is not None:
                state['jobs'][name].update(status='COMPLETE' if code==0 else 'FAILED',exit_code=code,finished_unix=time.time())
                del active[name]
        failed=[name for name,j in state['jobs'].items() if j['status']=='FAILED']
        for job in state['jobs'].values():
            if job['status']!='PENDING':continue
            if not all(state['jobs'][d]['status']=='COMPLETE' for d in job['dependencies']):continue
            resource=job['resource']
            used=[state['jobs'][n]['resource'] for n in active]
            if (resource=='cpu' and used.count('cpu')>=2) or (resource!='cpu' and resource in used):continue
            if failed:continue
            log=work/(job['id']+'.process.log')
            with log.open('ab',buffering=0) as handle:
                process=subprocess.Popen(job['command'],cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES='' if resource=='cpu' else str(resource)),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[job['id']]=process
            job.update(status='RUNNING',pid=process.pid,log=str(log),started_unix=time.time())
        if failed and not active:
            state.update(status='FAILED',failed=failed)
        elif all(j['status']=='COMPLETE' for j in state['jobs'].values()):state['status']='COMPLETE'
        write(state_path,state)
        if state['status']!='RUNNING':
            print(json.dumps(dict(status=state['status'],completed=sum(j['status']=='COMPLETE' for j in state['jobs'].values()),total=len(jobs))),flush=True);break
        time.sleep(5)


if __name__=='__main__':main()
