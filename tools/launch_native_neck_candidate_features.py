#!/usr/bin/env python3
"""Stream fixed-model Train feature extraction as exact crop banks finish."""
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
    if json.loads((w/'native_neck_train_features_state.json').read_text())['status']!='COMPLETE':raise ValueError('All immutable FP32 encoders required')
    lock=(w/'native_neck_candidate_features.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);path=w/'native_neck_candidate_features_state.json'
    if path.exists():raise ValueError('Native candidate features already scheduled')
    order=['MSK-002','BT-003','SN-015','SN-001','BT-001','PF-001'];jobs={f'{sequence}_{cue}':dict(status='PENDING',sequence=sequence,cue=cue) for sequence in order for cue in ['rgb','ir']}
    state=dict(status='RUNNING',jobs=jobs);active={}
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps','/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    while True:
        for gpu,(name,proc) in list(active.items()):
            rc=proc.poll()
            if rc is not None:jobs[name].update(status='COMPLETE' if rc==0 else 'FAILED',exit_code=rc);del active[gpu]
        for gpu in [2,3]:
            if gpu in active or any(j['status']=='FAILED' for j in jobs.values()):continue
            for name,job in jobs.items():
                if job['status']!='PENDING' or not (w/'native_candidate_crops'/job['sequence']/'complete.json').exists():continue
                command=[sys.executable,'-u','tools/extract_native_neck_candidate_features.py','--sequence',job['sequence'],'--cue',job['cue']];log=w/(name+'.native_neck_candidates.log')
                with log.open('ab',buffering=0) as handle:proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=(name,proc);job.update(status='RUNNING',pid=proc.pid,physical_gpu=gpu,command=command,log=str(log));break
        crop_state=json.loads((w/'native_candidate_crop_state.json').read_text())
        if not active:
            if all(j['status']=='COMPLETE' for j in jobs.values()):state['status']='COMPLETE'
            elif any(j['status']=='FAILED' for j in jobs.values()) or crop_state['status']=='FAILED':state['status']='FAILED'
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,path)
        if state['status']!='RUNNING':break
        time.sleep(5)


if __name__=='__main__':main()
