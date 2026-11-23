#!/usr/bin/env python3
"""Two CPU calibration workers followed by exactly physical GPU2/3 caching."""
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
    protocol=json.loads((ROOT/plan['protocol']).read_text());config_path=ROOT/'configs/perceptual_correlation_repair_20261123.json'
    state_path=work/'correlation_cache_state.json';lock=state_path.with_suffix('.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if state_path.exists():raise ValueError('Correlation cache already scheduled')
    state=dict(status='TRAIN_ONLY_PIXEL_CALIBRATION',config_sha256=sha256(config_path),jobs={},Test_read=False)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MFT26IR_ROOT=plan['data_root'])
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps',
        '/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    def run_stage(stage,requests):
        active={};pending=list(requests)
        while pending or active:
            for key,proc in list(active.items()):
                code=proc.poll()
                if code is not None:
                    state['jobs'][key].update(status='COMPLETE'if code==0 else'FAILED',exit_code=code);del active[key]
            failed=any(j['status']=='FAILED'for j in state['jobs'].values())
            while pending and len(active)<2 and not failed:
                sequence,gpu,smoke=pending.pop(0);key=f'{stage}_{sequence or "all"}'+(f'_smoke{smoke}'if smoke else'')
                command=[sys.executable,'-u','tools/build_correlation_cache.py','--stage',stage]
                if sequence:command+=['--sequence',sequence]
                if smoke:command+=['--max-frames',str(smoke)]
                log=work/(key+'.correlation_cache.log')
                with log.open('ab',buffering=0)as handle:
                    proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=gpu),stdin=subprocess.DEVNULL,
                        stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                active[key]=proc;state['jobs'][key]=dict(status='RUNNING',pid=proc.pid,command=command,log=str(log),physical_gpu=gpu)
            dump(state_path,state)
            if failed and not active:return False
            time.sleep(5)
        return True
    if not run_stage('statistics',[(seq,'',0)for seq in protocol['train']]):
        state['status']='FAILED';dump(state_path,state);return
    if not run_stage('merge',[(None,'',0)]):
        state['status']='FAILED';dump(state_path,state);return
    state['status']='REAL_INPUT_FEATURE_SOURCE_CHECKS'
    if not run_stage('features',[('BT-001','2',2),('BT-003','3',2)]):
        state['status']='FAILED';dump(state_path,state);return
    for seq in ['BT-001','BT-003']:
        check=json.loads((work/'correlation_cache'/seq/'feature_smoke_2/complete.json').read_text())
        if check['status']!='SMOKE_COMPLETE'or len(check['outputs'])!=9 or check['config_sha256']!=sha256(config_path):
            raise ValueError('Matched real shared-encoder source checks failed')
    state['status']='CACHING_TRAIN_FROZEN_FEATURES'
    # Same-sequence caches are never written by two workers. Each pair has
    # exactly one worker mapped to each of the two allowed physical GPUs.
    for first,second in [('BT-001','BT-003'),('MSK-002','PF-001'),('SN-001','SN-015')]:
        if not run_stage('features',[(first,'2',0),(second,'3',0)]):
            state['status']='FAILED';dump(state_path,state);return
    state['status']='COMPLETE';dump(state_path,state)


if __name__=='__main__':main()
