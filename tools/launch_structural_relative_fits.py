#!/usr/bin/env python3
"""Single-seed repairs/ablations, following each allowed card's current producer."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tools.correlation_geometry_checks import fresh_RGB_audit
from tools.single_seed_geometry_checks import previous_stream_checks, all_unavailable_checks, factorial_provenance_checks
from tools.run_pseudo_ir_comparison import dump, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text()); work = Path(plan['work_root'])
    config_path = ROOT/'configs/perceptual_structural_relative_20261123.json'; config = json.loads(config_path.read_text())
    policy_path = ROOT/'configs/perceptual_single_seed_policy_20261123.json'; policy = json.loads(policy_path.read_text())
    assert policy['fixed_seed'] == 0 and not policy['additional_seed_fits']
    assert policy['structural_base_config_sha256'] == sha256(config_path)
    path = work/'structural_relative_fit_state.json'; lock = path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); assert not path.exists()
    schedule = {'2': [('rgb', 'full'), ('ir', 'full'), ('ir', 'no_cycle')],
                '3': [('gray', 'full'), ('ir', 'no_structure'), ('ir', 'no_centering')]}
    jobs = {}
    for physical, variants in schedule.items():
        for cue, variant in variants:
            assert not (work/'structural_relative_fit'/f'{cue}_{variant}_s0').exists()
            jobs[f'{cue}_{variant}_s0'] = dict(cue=cue, variant=variant, seed=0,
                physical_gpu=physical, status='PENDING')
    assert {(j['cue'], j['variant'], j['seed']) for j in jobs.values()} == {
        (j['cue'], j['variant'], j['seed']) for j in config['initial_fits']}
    names = ['tools/train_structural_relative.py', 'yolox/models/structural_relative_repair.py',
        'yolox/models/structural_relative_losses.py', 'yolox/models/frozen_head_repair.py',
        'yolox/models/recoverable_curriculum.py', 'tools/train_correlation_repair.py',
        'tools/structural_grid.py', 'tools/structural_correspondence.py']
    sources = {name: sha256(ROOT/name) for name in names}
    state = dict(status='WAITING_FOR_SINGLE_SEED_GEOMETRY_PRODUCERS', jobs=jobs,
        sources=sources, specification_sha256=sha256(config_path), single_seed_policy_sha256=sha256(policy_path),
        source_dependencies_per_card={}, Test_read=False, supported_innovations=0, goal_met=False)
    dump(path, state); active = {}
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', MFT26IR_ROOT=plan['data_root'])
    reference = json.loads((work/'correlation_fit/ir_full_s0/complete.json').read_text())
    expected_draws = [row['sample_digest'] for row in reference['history']]
    geometry_path = ROOT/'configs/perceptual_single_seed_geometry_eval_20261123.json'
    geometry_spec = json.loads(geometry_path.read_text())
    while True:
        for physical, (key, phase, proc) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            del active[physical]; job = jobs[key]
            job[phase+'_exit_code'] = code
            if code:
                job['status'] = 'FAILED'; continue
            folder = work/'structural_relative_fit'/key
            if phase == 'smoke':
                folder = folder/'smoke_2'
            complete = json.loads((folder/'complete.json').read_text())
            assert complete['initial_raw_RGB_head_bit_exact'] and complete['trainable_parameters'] == 145600
            assert complete['config_sha256'] == state['specification_sha256']
            if phase == 'smoke':
                assert complete['status'] == 'SMOKE_COMPLETE' and complete['successful_steps_this_invocation'] == 2
                job['status'] = 'SMOKE_PASSED'
            else:
                assert complete['status'] == 'COMPLETE_TRAIN_STRUCTURAL_RELATIVE'
                assert complete['epochs'] == 24 and complete['successful_steps_this_invocation'] == 6912
                assert [row['sample_digest'] for row in complete['history']] == expected_draws
                job['status'] = 'COMPLETE'
            job[phase+'_complete_sha256'] = sha256(folder/'complete.json')
        dependency = json.loads((work/'single_seed_geometry_dev_state.json').read_text())
        cache = json.loads((work/'structural_cache/state.json').read_text())
        for physical, sequence in [('2', 'BT-005'), ('3', 'SN-013')]:
            if physical in active or any(j['status'] == 'FAILED' for j in jobs.values()):
                continue
            if cache['status'] != 'COMPLETE' or dependency.get('producers', {}).get(sequence, {}).get('status') != 'COMPLETE':
                continue
            if physical not in state['source_dependencies_per_card']:
                base = work/'single_seed_geometry_predictions/dev'/sequence
                meta = json.loads((base/'complete.json').read_text())
                assert meta['frames'] == {'BT-005': 1834, 'SN-013': 3000}[sequence]
                assert meta['specification_sha256'] == sha256(geometry_path)
                state['source_dependencies_per_card'][physical] = dict(sequence=sequence,
                    previous30=previous_stream_checks(base, sequence, geometry_spec, work),
                    unavailable10=all_unavailable_checks(base, sequence, plan, work),
                    provenance=factorial_provenance_checks(base, geometry_spec),
                    freshRGB=fresh_RGB_audit(base, sequence, plan, work))
                dump(path, state)
            pending = [(key, j) for key, j in jobs.items() if j['physical_gpu'] == physical
                       and j['status'] in ['PENDING', 'SMOKE_PASSED']]
            if not pending:
                continue
            key, job = pending[0]; phase = 'smoke' if job['status'] == 'PENDING' else 'fit'
            assert sha256(config_path) == state['specification_sha256']
            assert sha256(policy_path) == state['single_seed_policy_sha256']
            assert all(sha256(ROOT/name) == value for name, value in sources.items())
            if shutil.disk_usage(work).free < 40*2**30:
                raise RuntimeError('Keep independent-disk reserve')
            command = [sys.executable, '-u', 'tools/train_structural_relative.py',
                '--cue', job['cue'], '--variant', job['variant'], '--seed', '0']
            if phase == 'smoke':
                command += ['--smoke-steps', '2']
            log = work/(key+'.structural_relative_'+phase+'.log')
            with log.open('ab', buffering=0) as handle:
                proc = subprocess.Popen(command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=physical),
                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            active[physical] = (key, phase, proc)
            job.update(status=phase.upper()+'_RUNNING', pid=proc.pid, command=command, log=str(log))
        state['status'] = 'FITTING_SINGLE_SEED' if active else 'WAITING_FOR_SINGLE_SEED_GEOMETRY_PRODUCERS'
        if not active and any(j['status'] == 'FAILED' for j in jobs.values()):
            state['status'] = 'FAILED'
        if all(j['status'] == 'COMPLETE' for j in jobs.values()):
            state['status'] = 'COMPLETE'
        dump(path, state)
        if state['status'] in ['COMPLETE', 'FAILED']:
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
