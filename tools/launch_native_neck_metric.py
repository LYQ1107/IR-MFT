#!/usr/bin/env python3
"""Only physical GPU2/3: matched endpoint metric repairs on cached Train crops."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    work=Path(json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())['work_root'])
    path=work/'native_neck_metric_state.json'
    lock=path.with_suffix('.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if path.exists():raise ValueError('Neck metric repair already scheduled')
    state=dict(status='RUNNING',endpoint_epochs=36,cumulative_encoder_epochs=72,jobs={})
    def save():
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,path)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps',
        '/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages',
        '/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    for seed in [0,1,2]:
        active={}
        for cue,gpu in [('rgb',2),('ir',3)]:
            name=f'{cue}_s{seed}'
            command=[sys.executable,'-u','tools/train_native_neck_metric.py','--cue',cue,'--seed',str(seed)]
            log=work/(name+'.native_neck_metric.fit.log')
            with log.open('ab',buffering=0) as handle:
                proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),
                    stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[name]=proc;state['jobs'][name]=dict(status='RUNNING',pid=proc.pid,physical_gpu=gpu,command=command,log=str(log))
        save()
        while active:
            for name,proc in list(active.items()):
                code=proc.poll()
                if code is not None:
                    state['jobs'][name].update(status='COMPLETE' if code==0 else 'FAILED',exit_code=code)
                    del active[name]
            save();time.sleep(5)
        if any(job['status']=='FAILED' for job in state['jobs'].values()):
            state['status']='FAILED';save();return
    state['status']='COMPLETE';save()


if __name__=='__main__':main()
