#!/usr/bin/env python3
"""Two CPU workers for the registered native-query Dev snapshots."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.run_pseudo_ir_comparison import sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='native_query_dev')
    parser.add_argument('--config-dir', default='configs/perceptual_native_query_dev_20261123')
    parser.add_argument('--fit-state', default='native_query_fit_e18_state.json')
    args = parser.parse_args()
    work = Path(json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())['work_root'])
    path = work/(args.label+'_state.json')
    lock = (work/(args.label+'.lock')).open('w');fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if path.exists():
        raise ValueError('Native query Dev already scheduled')
    files = list((ROOT/args.config_dir).glob('*.json'))
    def order(path):
        cfg = json.loads(path.read_text())
        return (0 if cfg['detection_arm']=='new' else 1,
                0 if 'e18' in path.stem else 1,
                0 if cfg.get('disable_native_increment') else 1,
                cfg['native_seed'], 0 if cfg['native_cue']=='ir' else 1, path.stem)
    jobs = {p.stem: dict(status='PENDING', config=str(p), sha256=sha256(p)) for p in sorted(files,key=order)}
    if not jobs:
        raise ValueError('No registered Dev jobs')
    state = dict(status='RUNNING', scope='Complete two-video Dev only; no Test selection', jobs=jobs)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', MFT26IR_ROOT='/data3/liuyeqiang/datasets/mft26-ir',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = ':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps',
        '/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages',
        '/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    active = {}
    while True:
        for name, process in list(active.items()):
            code = process.poll()
            if code is not None:
                jobs[name].update(status='COMPLETE' if code==0 else 'FAILED', exit_code=code)
                del active[name]
        fit = json.loads((work/args.fit_state).read_text())
        failed = fit['status'] in {'FAILED','DEPENDENCY_FAILED'} or any(j['status']=='FAILED' for j in jobs.values())
        for name, job in jobs.items():
            if failed or len(active)>=2:
                break
            if job['status'] != 'PENDING':
                continue
            config = Path(job['config'])
            if sha256(config) != job['sha256']:
                raise ValueError('Registered configuration changed')
            cfg = json.loads(config.read_text())
            if not Path(cfg['native_query_checkpoint']).exists():
                continue
            command = [sys.executable,'-u','tools/replay_perceptual_mft.py','--config',str(config),
                       '--split','dev','--output',str(work/args.label/name)]
            log = work/(args.label+'_'+name+'.process.log')
            with log.open('ab',buffering=0) as handle:
                process = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                    stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[name]=process
            job.update(status='RUNNING',pid=process.pid,command=command,log=str(log))
        if not active:
            if failed: state['status']='FAILED'
            elif all(j['status']=='COMPLETE' for j in jobs.values()) and fit['status']=='COMPLETE':
                state['status']='COMPLETE'
        temporary = path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,path)
        if state['status'] != 'RUNNING':
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
