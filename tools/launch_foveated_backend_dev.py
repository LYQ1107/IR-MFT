#!/usr/bin/env python3
"""Produce fresh evidence for selected boxes, then run the full cue factorial."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.run_pseudo_ir_comparison import dump, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    a = p.parse_args()
    path = Path(a.plan)
    plan = json.loads(path.read_text())
    registration = plan['foveated_backend_registration']
    base = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work = Path(base['work_root'])
    name = registration['name']
    label = 'foveated_backend_'+name+'_dev'
    state_path = work/(label+'_state.json')
    lock = state_path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if state_path.exists():
        raise ValueError('Backend already scheduled')
    front = registration['roles']['new']['global_frontend']
    folder = ROOT/'configs/perceptual_foveated_backend_dev_20261123'/name
    folder.mkdir(parents=True, exist_ok=False)
    jobs = {}
    for alias in ['new', 'rgb']:
        for cue in ['rgb', 'ir']:
            for seed in range(3):
                source = ROOT/f'configs/perceptual_pair_selected_20261123/K_{front}_{cue}_s{seed}.json'
                config = json.loads(source.read_text())
                config['id'] = f'F_{name}_{alias}_{cue}_s{seed}'
                config['detection_arm'] = alias
                config['foveated_plan'] = str(path)
                config['foveated_plan_sha256'] = sha256(path)
                config['source_global_frontend'] = front
                config['source_attention_role'] = registration['roles'][alias]['role']
                config['registration'] = ('Dev-selected observations, unchanged original Train-fitted F0 weights. '
                    'Full IR/RGB attention x IR/RGB point-cue factorial, all three seeds, no refitting or Test selection.')
                out = folder/(config['id']+'.json')
                dump(out, config)
                jobs[config['id']] = dict(status='PENDING', config=str(out), sha256=sha256(out))
    state = dict(status='PRODUCING_FRESH_EVIDENCE', plan=str(path), plan_sha256=sha256(path),
                 Test_started=False, jobs=jobs, producers={})
    dump(state_path, state)
    env = dict(os.environ, MFT26IR_ROOT=base['data_root'], OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = ':'.join([str(ROOT), str(ROOT/'exps'), '/data3/liuyeqiang/.codex/pydeps',
        '/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages',
        '/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    active = {}
    for sequence, gpu in [('BT-005', '2'), ('SN-013', '3')]:
        for kind, script, device in [('appearance', 'tools/extract_perceptual_mft_features.py', gpu),
                                     ('points', 'tools/extract_foveated_correspondences.py', '')]:
            key = sequence+'_'+kind
            command = [sys.executable, '-u', script, '--plan', str(path), '--split', 'dev', '--sequence', sequence]
            log = work/(label+'_'+key+'.log')
            with log.open('ab', buffering=0) as handle:
                proc = subprocess.Popen(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=device),
                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            active[key] = proc
            state['producers'][key] = dict(status='RUNNING', pid=proc.pid, log=str(log), command=command,
                                           physical_gpu=device)
    while active:
        for key, proc in list(active.items()):
            code = proc.poll()
            if code is not None:
                state['producers'][key].update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code)
                del active[key]
        dump(state_path, state)
        time.sleep(5)
    if any(j['status'] == 'FAILED' for j in state['producers'].values()):
        state['status'] = 'FAILED'
        dump(state_path, state)
        return
    state['status'] = 'RUNNING'
    while True:
        for key, proc in list(active.items()):
            code = proc.poll()
            if code is not None:
                jobs[key].update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code)
                del active[key]
        for key, job in jobs.items():
            if len(active) >= 2 or any(j['status'] == 'FAILED' for j in jobs.values()):
                break
            if job['status'] != 'PENDING':
                continue
            if sha256(job['config']) != job['sha256'] or sha256(path) != state['plan_sha256']:
                raise ValueError('Frozen backend registration changed')
            command = [sys.executable, '-u', 'tools/replay_perceptual_mft.py', '--plan', str(path),
                       '--config', job['config'], '--split', 'dev', '--output', str(work/label/key)]
            log = work/(label+'_'+key+'.log')
            with log.open('ab', buffering=0) as handle:
                proc = subprocess.Popen(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=''),
                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            active[key] = proc
            job.update(status='RUNNING', pid=proc.pid, log=str(log), command=command)
        if not active:
            if all(j['status'] == 'COMPLETE' for j in jobs.values()):
                state['status'] = 'COMPLETE'
            elif any(j['status'] == 'FAILED' for j in jobs.values()):
                state['status'] = 'FAILED'
        dump(state_path, state)
        if state['status'] != 'RUNNING':
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
