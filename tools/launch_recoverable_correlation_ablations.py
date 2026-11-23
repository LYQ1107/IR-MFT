#!/usr/bin/env python3
"""Registered removals on the allowed spare GPU, yielding to primary Dev."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tools.run_pseudo_ir_comparison import dump,sha256

ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(plan['work_root'])
    spec_path=ROOT/'configs/perceptual_recoverable_correlation_ablations_20261123.json';spec=json.loads(spec_path.read_text())
    if sha256(ROOT/spec['parent'])!=spec['parent_sha256']:raise ValueError('Registered parent changed')
    path=work/'correlation_recoverability_ablation_fit_state.json';lock=path.with_suffix('.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if path.exists():raise ValueError('Removal fitting already scheduled')
    files=['tools/train_recoverable_correlation.py','yolox/models/recoverable_curriculum.py',
           'yolox/models/local_correlation_repair.py','yolox/models/frozen_head_repair.py']
    sources={name:sha256(ROOT/name)for name in files}
    jobs={variant:dict(status='PENDING',physical_gpu='2',paused_seconds=0.,pauses=[])for variant in spec['variants']}
    state=dict(status='FITTING_REGISTERED_REMOVALS',jobs=jobs,specification_sha256=sha256(spec_path),source_sha256=sources,Test_read=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MFT26IR_ROOT=plan['data_root'])
    active=None;paused_at=None
    while True:
        main_dev=json.loads((work/'correlation_recoverability_pilot_dev_state.json').read_text())
        yield_primary=any(j['status']=='RUNNING'for j in main_dev.get('producers',{}).values())
        if active is not None:
            key,proc=active;job=jobs[key];code=proc.poll()
            if code is not None:
                job.update(status='COMPLETE'if code==0 else'FAILED',exit_code=code);active=None
            elif yield_primary and paused_at is None:
                os.kill(proc.pid,signal.SIGSTOP);paused_at=time.time();job['status']='PAUSED_FOR_PRIMARY_DEV'
                job['pauses'].append(dict(start_UNIX=paused_at,reason='Primary full-Dev observation producer'))
            elif not yield_primary and paused_at is not None:
                os.kill(proc.pid,signal.SIGCONT);now=time.time();job['paused_seconds']+=now-paused_at
                job['pauses'][-1]['end_UNIX']=now;paused_at=None;job['status']='RUNNING'
        if active is None and not yield_primary and not any(j['status']=='FAILED'for j in jobs.values()):
            pending=[key for key,job in jobs.items()if job['status']=='PENDING']
            if pending:
                if sha256(spec_path)!=state['specification_sha256']or any(sha256(ROOT/n)!=s for n,s in sources.items()):
                    raise ValueError('Registered removal source changed')
                key=pending[0];command=[sys.executable,'-u','tools/train_recoverable_correlation.py','--cue','ir','--seed','0','--variant',key]
                log=work/(key+'.correlation_recoverability_ablation.log')
                with log.open('ab',buffering=0)as handle:
                    proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                jobs[key].update(status='RUNNING',pid=proc.pid,command=command,log=str(log));active=(key,proc)
        if active is None:
            if all(j['status']=='COMPLETE'for j in jobs.values()):state['status']='COMPLETE'
            elif any(j['status']=='FAILED'for j in jobs.values()):state['status']='FAILED'
        dump(path,state)
        if state['status']!='FITTING_REGISTERED_REMOVALS':break
        time.sleep(5)


if __name__=='__main__':main()
