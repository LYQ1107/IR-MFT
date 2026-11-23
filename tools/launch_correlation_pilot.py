#!/usr/bin/env python3
"""Matched full-budget IR/RGB/gray pilot, only the two allowed physical GPUs."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.run_pseudo_ir_comparison import dump,sha256

ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(plan['work_root'])
    config_path=ROOT/'configs/perceptual_correlation_repair_20261123.json'
    path=work/'correlation_pilot_fit_state.json';lock=path.with_suffix('.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if path.exists():raise ValueError('Correlation pilot already scheduled')
    sources={name:sha256(ROOT/name)for name in ['tools/train_correlation_repair.py',
        'yolox/models/local_correlation_repair.py','yolox/models/frozen_head_repair.py']}
    state=dict(status='MATCHED_REAL_INPUT_GRADIENT_CHECKS',config_sha256=sha256(config_path),source_sha256=sources,
        jobs={},seed=0,epochs=24,Test_read=False,formal_contribution_claim=False)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MFT26IR_ROOT=plan['data_root'])
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps',
        '/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    def run(smoke):
        pending=['gray','ir','rgb'];active={}
        while pending or active:
            for key,proc in list(active.items()):
                code=proc.poll()
                if code is not None:
                    state['jobs'][key].update(status='COMPLETE'if code==0 else'FAILED',exit_code=code);del active[key]
            failed=any(j['status']=='FAILED'for j in state['jobs'].values())
            occupied={state['jobs'][key]['physical_gpu']for key in active}
            for gpu in ['2','3']:
                if not pending or failed:break
                if gpu in occupied:continue
                if sha256(config_path)!=state['config_sha256']or any(sha256(ROOT/name)!=value for name,value in sources.items()):
                    raise ValueError('Registered fitting source changed')
                cue=pending.pop(0);key=cue+('_smoke4'if smoke else'_full24')
                command=[sys.executable,'-u','tools/train_correlation_repair.py','--cue',cue,'--seed','0']
                if smoke:command+=['--smoke-steps','4']
                log=work/(key+'.correlation_pilot.log')
                with log.open('ab',buffering=0)as handle:
                    proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=gpu),stdin=subprocess.DEVNULL,
                        stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                active[key]=proc;state['jobs'][key]=dict(status='RUNNING',pid=proc.pid,physical_gpu=gpu,command=command,log=str(log))
            dump(path,state)
            if failed and not active:return False
            time.sleep(5)
        return True
    if not run(True):state['status']='FAILED';dump(path,state);return
    checks=[]
    for cue in ['rgb','gray','ir']:
        check=json.loads((work/'correlation_fit'/f'{cue}_full_s0/smoke_4/complete.json').read_text())
        if check['status']!='SMOKE_COMPLETE'or check['successful_steps_this_invocation']!=4:
            raise ValueError('Four actual gradient steps required for all modalities')
        checks.append(check)
    if len({c['history'][0]['sample_digest']for c in checks})!=1 or len({c['trainable_parameters']for c in checks})!=1:
        raise ValueError('Matched capacity and sample draws required')
    state['status']='WAITING_FOR_ALL_TRAIN_CACHES'
    while True:
        cache=json.loads((work/'correlation_cache_state.json').read_text())
        if cache['status']=='FAILED':state['status']='DEPENDENCY_FAILED';dump(path,state);return
        if cache['status']=='COMPLETE':break
        dump(path,state);time.sleep(5)
    state['status']='FITTING_MATCHED24_EPOCH_PILOTS'
    if not run(False):state['status']='FAILED';dump(path,state);return
    state['status']='COMPLETE';dump(path,state)


if __name__=='__main__':main()
