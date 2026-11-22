#!/usr/bin/env python3
"""Two current-frame encoder workers, optionally waiting for earlier features."""
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
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','test'],default='dev');p.add_argument('--wait-state');a=p.parse_args()
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root']);label='native_neck_eval_'+a.split
    lock=(w/(label+'.lock')).open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);path=w/(label+'_state.json')
    if path.exists():raise ValueError('Native evaluation features already scheduled')
    if not a.wait_state:raise ValueError('Explicit dependency on completed neck Train features required')
    sequences=protocol['dev' if a.split=='dev' else 'historical_test']
    jobs={f'{cue}_{sequence}':dict(status='PENDING',cue=cue,sequence=sequence) for cue in ['ir','rgb'] for sequence in reversed(sequences)}
    state=dict(status='WAITING_FOR_DEPENDENCY' if a.wait_state else 'RUNNING',split=a.split,jobs=jobs,scope='Unlabeled current-image representations only, no GT or Test accuracy selection')
    def save():
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,path)
    if a.wait_state:
        while True:
            dependency=w/a.wait_state
            if dependency.exists():
                status=json.loads(dependency.read_text())['status']
                if status=='COMPLETE':break
                if status in {'FAILED','DEPENDENCY_FAILED'}:state['status']='DEPENDENCY_FAILED';save();return
            save();time.sleep(5)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps','/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    state['status']='RUNNING';active={}
    while True:
        for gpu,(name,proc) in list(active.items()):
            rc=proc.poll()
            if rc is not None:jobs[name].update(status='COMPLETE' if rc==0 else 'FAILED',exit_code=rc);del active[gpu]
        for gpu in [2,3]:
            if gpu in active or any(x['status']=='FAILED' for x in jobs.values()):continue
            name=next((key for key,value in jobs.items() if value['status']=='PENDING'),None)
            if name is None:continue
            job=jobs[name];command=[sys.executable,'-u','tools/extract_native_neck_eval_features.py','--split',a.split,'--cue',job['cue'],'--sequence',job['sequence']];log=w/f'{a.split}_{name}.native_neck_eval.log'
            with log.open('ab',buffering=0) as handle:proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[gpu]=(name,proc);job.update(status='RUNNING',pid=proc.pid,physical_gpu=gpu,command=command,log=str(log))
        if not active:
            if all(x['status']=='COMPLETE' for x in jobs.values()):state['status']='COMPLETE'
            elif any(x['status']=='FAILED' for x in jobs.values()):state['status']='FAILED'
        save()
        if state['status']!='RUNNING':break
        time.sleep(5)


if __name__=='__main__':main()
