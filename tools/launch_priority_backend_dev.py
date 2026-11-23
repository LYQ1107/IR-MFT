#!/usr/bin/env python3
"""Frozen priority-observation pipeline x matched point cues x three seeds."""
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
    plan_path=ROOT/'configs/perceptual_priority_backend_20261123.json';plan=json.loads(plan_path.read_text())
    main=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(main['work_root'])
    evidence=Path(plan['work_root']);path=work/'priority_best_backend_dev_state.json'
    lock=path.with_suffix('.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if path.exists():raise ValueError('Priority backend already scheduled')
    for sequence in ['BT-005','SN-013']:
        feature=json.loads((evidence/'features/dev'/sequence/'complete.json').read_text())
        point=json.loads((evidence/'transport/dev'/sequence/'complete.json').read_text())
        if feature['status']!='COMPLETE'or point['status']!='COMPLETE'or feature['GT_read']or point['GT_read']:
            raise ValueError('Complete unlabeled evidence required')
    folder=ROOT/'configs/perceptual_priority_backend_dev_20261123';folder.mkdir(exist_ok=False);jobs={}
    for alias in ['new','rgb']:
        for cue in ['rgb','ir']:
            for seed in range(3):
                source=ROOT/f'configs/perceptual_pair_selected_20261123/K_{alias}_{cue}_s{seed}.json'
                config=json.loads(source.read_text());name=f'PB_{alias}_{cue}_s{seed}'
                config.update(id=name,priority_backend_plan_sha256=sha256(plan_path),
                    observation_role='Paired-new-global/RGB-guided-priority'if alias=='new'else'Original RGB-global control',
                    registration='Whole complete20-row Dev selection, unchanged original Train-fitted point weights, all3 seeds and both RGB/IR point cues. Global paired capacity differs and is not novelty evidence alone. No Test selection.')
                out=folder/(name+'.json');dump(out,config);jobs[name]=dict(status='PENDING',config=str(out),sha256=sha256(out))
    state=dict(status='RUNNING',jobs=jobs,plan_sha256=sha256(plan_path),Test_read=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MFT26IR_ROOT=main['data_root'])
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps',
        '/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    active={}
    while True:
        for key,proc in list(active.items()):
            code=proc.poll()
            if code is not None:jobs[key].update(status='COMPLETE'if code==0 else'FAILED',exit_code=code);del active[key]
        for key,job in jobs.items():
            if len(active)>=2 or any(j['status']=='FAILED'for j in jobs.values()):break
            if job['status']!='PENDING':continue
            if sha256(job['config'])!=job['sha256']or sha256(plan_path)!=state['plan_sha256']:raise ValueError('Frozen backend changed')
            command=[sys.executable,'-u','tools/replay_perceptual_mft.py','--plan',str(plan_path),'--config',job['config'],
                '--split','dev','--output',str(work/'priority_best_backend_dev'/key)]
            log=work/(key+'.priority_best_backend.log')
            with log.open('ab',buffering=0)as handle:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[key]=proc;job.update(status='RUNNING',pid=proc.pid,command=command,log=str(log))
        if not active:
            if all(j['status']=='COMPLETE'for j in jobs.values()):state['status']='COMPLETE'
            elif any(j['status']=='FAILED'for j in jobs.values()):state['status']='FAILED'
        dump(path,state)
        if state['status']!='RUNNING':break
        time.sleep(5)


if __name__=='__main__':main()
