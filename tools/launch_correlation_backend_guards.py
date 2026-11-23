#!/usr/bin/env python3
"""Matched mechanical guard repairs for a failed fixed-weight compound."""
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
    main=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(main['work_root'])
    repair_path=ROOT/'configs/perceptual_correlation_backend_guard_repair_20261123.json';repair=json.loads(repair_path.read_text())
    plan_path=ROOT/repair['parent_plan'];plan=json.loads(plan_path.read_text())
    if sha256(plan_path)!=repair['parent_plan_sha256']:raise ValueError('Frozen data plan changed')
    completed=work/'correlation_backend_fixed_state.json'
    if json.loads(completed.read_text())['status']!='COMPLETE'or sha256(completed)!=repair['completed18_state_sha256']:
        raise ValueError('All18 original fixed controls required')
    path=work/'correlation_backend_guarded_dev_state.json';lock=path.with_suffix('.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if path.exists():raise ValueError('Guard repairs already scheduled')
    folder=ROOT/'configs/perceptual_correlation_backend_guarded_dev_20261123';folder.mkdir(exist_ok=False);jobs={}
    for setting in repair['settings']:
        for observation in ['rgb','gray','ir']:
            for cue in ['rgb','ir']:
                for seed in range(3):
                    source=ROOT/f'configs/perceptual_correlation_backend_fixed_dev_20261123/CB_{observation}_{cue}_s{seed}.json'
                    config=json.loads(source.read_text());name=f'CG_{observation}_{cue}_s{seed}_{setting["name"]}'
                    if sha256(config['association_checkpoint'])!=config['association_checkpoint_sha256']:raise ValueError('Frozen association head changed')
                    config.update(id=name,global_birth_threshold=setting['global_birth_threshold'],preserve_clear_motion=setting['preserve_clear_motion'],
                        guard_repair_sha256=sha256(repair_path),source_fixed_config_sha256=sha256(source),
                        registration='Same exact failed18 cached inputs/weights. Restore originalbirth0.7, with/without existingclear-motionguard, acrossall3observationcues/bothassociationcues/3associationseeds. No other parameter search, Test or noveltyclaim for genericguards.')
                    target=folder/(name+'.json');dump(target,config);jobs[name]=dict(status='PENDING',config=str(target),sha256=sha256(target))
    state=dict(status='RUNNING',jobs=jobs,repair_sha256=sha256(repair_path),data_plan_sha256=sha256(plan_path),Test_read=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MFT26IR_ROOT=main['data_root'])
    active={}
    while True:
        for key,proc in list(active.items()):
            code=proc.poll()
            if code is not None:jobs[key].update(status='COMPLETE'if code==0 else'FAILED',exit_code=code);del active[key]
        for key,job in jobs.items():
            if len(active)>=2 or any(j['status']=='FAILED'for j in jobs.values()):break
            if job['status']!='PENDING':continue
            if sha256(job['config'])!=job['sha256']or sha256(repair_path)!=state['repair_sha256']:raise ValueError('Frozen guard repair changed')
            command=[sys.executable,'-u','tools/replay_perceptual_mft.py','--plan',str(plan_path),'--config',job['config'],
                '--split','dev','--output',str(work/'correlation_backend_guarded_dev'/key)]
            log=work/(key+'.correlation_backend_guarded.log')
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
