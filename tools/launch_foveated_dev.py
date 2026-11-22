#!/usr/bin/env python3
"""Two GPU observation producers, then two CPU full-Dev evaluation workers."""
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
    path=work/'foveated_dev_state.json'
    lock=path.with_suffix('.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if path.exists():raise ValueError('Foveated Dev already scheduled')
    config_path=ROOT/'configs/perceptual_foveated_mft_20261123.json';config=json.loads(config_path.read_text())
    for sequence,gpu in [('BT-005','2'),('SN-013','3')]:
        smoke=json.loads((work/'foveated_proposals/dev'/sequence/'smoke_4/complete.json').read_text())
        if (smoke['status']!='SMOKE_COMPLETE' or smoke['config_sha256']!=sha256(config_path)
                or smoke['physical_gpu']!=gpu or smoke['GT_read'] or len(smoke['outputs'])!=36):
            raise ValueError('Both matching real-image source checks are required')
    folder=ROOT/'configs/perceptual_foveated_dev_20261123'
    if folder.exists():raise ValueError('Foveated Dev registration already exists')
    folder.mkdir();keys=[f'{arm}_{cue}_{scale}_k{count}_{fusion}' for arm in ['rgb','new']
        for cue in ['rgb','ir'] for scale in config['attention_scales'] for count in [2,4] for fusion in ['refine','nms']]
    keys += [f'{arm}_uniform_k4_{fusion}' for arm in ['rgb','new'] for fusion in ['refine','nms']]
    keys += ['baseline_rgb','baseline_new']
    jobs={}
    for key in keys:
        name='Z_'+key;cfg=dict(id=name,proposal_key=key,fovea_plan_sha256=sha256(config_path),
            registration='Complete deterministic current-image observations, unchanged original ByteTrack. Equal crop model/count and scale controls, no Test selection.')
        p=folder/(name+'.json');dump(p,cfg);jobs[name]=dict(status='PENDING',config=str(p),sha256=sha256(p))
    state=dict(status='PRODUCING_CURRENT_OBSERVATIONS',split='dev',producers={},jobs=jobs)
    env=dict(os.environ,MFT26IR_ROOT='/data3/liuyeqiang/datasets/mft26-ir',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps',
        '/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    active={}
    for sequence,gpu in [('BT-005',2),('SN-013',3)]:
        command=[sys.executable,'-u','tools/extract_foveated_observations.py','--split','dev','--sequence',sequence]
        log=work/(sequence+'.foveated_full_dev.log')
        with log.open('ab',buffering=0) as handle:
            proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdin=subprocess.DEVNULL,
                stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        active[sequence]=proc;state['producers'][sequence]=dict(status='RUNNING',pid=proc.pid,physical_gpu=gpu,command=command,log=str(log))
    while active:
        for name,proc in list(active.items()):
            code=proc.poll()
            if code is not None:
                state['producers'][name].update(status='COMPLETE' if code==0 else 'FAILED',exit_code=code);del active[name]
        dump(path,state);time.sleep(5)
    if any(value['status']=='FAILED' for value in state['producers'].values()):
        state['status']='FAILED';dump(path,state);return
    state['status']='RUNNING';active={}
    while True:
        for name,proc in list(active.items()):
            code=proc.poll()
            if code is not None:jobs[name].update(status='COMPLETE' if code==0 else 'FAILED',exit_code=code);del active[name]
        for name,job in jobs.items():
            if len(active)>=2 or any(value['status']=='FAILED' for value in jobs.values()):break
            if job['status']!='PENDING':continue
            if sha256(job['config'])!=job['sha256']:raise ValueError('Registered detector policy changed')
            log=work/(name+'.foveated_dev.log');command=[sys.executable,'-u','tools/replay_foveated_observations.py',
                '--config',job['config'],'--split','dev','--output',str(work/'foveated_dev'/name)]
            with log.open('ab',buffering=0) as handle:
                proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,
                    stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[name]=proc;job.update(status='RUNNING',pid=proc.pid,command=command,log=str(log))
        if not active:
            if all(value['status']=='COMPLETE' for value in jobs.values()):state['status']='COMPLETE'
            elif any(value['status']=='FAILED' for value in jobs.values()):state['status']='FAILED'
        dump(path,state)
        if state['status']!='RUNNING':break
        time.sleep(5)


if __name__=='__main__':main()
