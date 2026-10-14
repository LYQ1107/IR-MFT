#!/usr/bin/env python3
"""First-fold controlled fits; wait for actual owned producer teardown."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--state', type=Path, required=True)
    ap.add_argument('--gpu', choices=('1', '3'), required=True)
    a = ap.parse_args()
    config_path = Path('configs/cfp/phase_vii_pixel_masked_identity_v1.json')
    config = json.loads(config_path.read_text())
    if a.state.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != a.gpu or a.root.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError('New outsideGit state and assigned physical GPU required')
    sources = [__file__, 'tools/cfp_train_pixel_masked_identity.py', 'cfp/masked_identity_checkpoint.py',
               'cfp/fish_masked_identity.py', 'cfp/dinov2_fish_encoder.py', 'cfp/dinov2_fish_data.py',
               'cfp/dinov2_identity.py', 'cfp/local_identity_training.py', 'cfp/local_identity_encoder.py',
               'cfp/crop_retrieval.py', 'cfp/reid.py', 'cfp/provenance.py', str(config_path),
               'tools/cfp_verify_pixel_masked_restart.py']
    guard = {p: sha256(p) for p in sources}
    # Continuous seed0 and a distinct real epoch2 exit/resume, then other registered controls.
    tasks = [('pixel_masked', 0, False), ('pixel_masked', 0, True)]
    tasks += [(mode, seed, False) for seed in config['seed_plan'] for mode in config['modes'] if (mode, seed) != ('pixel_masked', 0)]
    reserve = dict(free_GiB=shutil.disk_usage(a.root).free/2**30, fits=len(tasks),
                   estimated_MiB_per_fit=250, additional_other_job_margin_GiB=5,
                   minimum_required_free_GiB=42+len(tasks)*250/1024+5)
    if reserve['free_GiB'] < reserve['minimum_required_free_GiB']:
        raise OSError('Registered fit/storage margin would breach protected reserve')
    state = dict(status='WAITING_OWNED_NATIVE_PRODUCER', pid=os.getpid(), gpu=a.gpu, tasks=tasks,
                 completed=[], active=None, launches=[], source_guards=guard, storage_plan=reserve,
                 sources=snapshot_sources(a.state.with_suffix('.sources'), sources))
    write_json(a.state, state)
    predecessor = a.root/f'native_dino_matrix_worker{a.gpu}_v1.json'
    try:
        while True:
            if any(sha256(p) != h for p, h in guard.items()):
                raise ValueError('Immutable queue source drift')
            owned = json.loads(predecessor.read_text())
            try:
                os.kill(owned['pid'], 0)
                alive = True
            except ProcessLookupError:
                alive = False
            idle = subprocess.check_output(['nvidia-smi', '-i', a.gpu, '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
            memory, utilization = [int(x.strip()) for x in idle.split(',')]
            if owned['status'] == 'COMPLETE_REGISTERED_NATIVE_DINO_MATRIX' and not alive and memory <= 64 and utilization == 0:
                break
            if owned['status'] == 'HALTED_ACTUAL_FAILURE':
                raise RuntimeError('Predecessor failed; do not silently take its GPU')
            state.update(last_gpu=dict(memory_MiB=memory, utilization=utilization), predecessor_status=owned['status'], predecessor_process_alive=alive)
            write_json(a.state, state)
            time.sleep(20)
        state['status'] = 'RUNNING'; write_json(a.state, state)
        for mode, seed, restart in tasks:
            if any(sha256(p) != h for p, h in guard.items()):
                raise ValueError('Immutable training source drift')
            if shutil.disk_usage(a.root).free < 42*2**30:
                raise OSError('42GiB parallel reserve before next fit')
            name = f'pixel_masked_identity_{mode}_SN-015_k4_s{seed}_v1'+('_restarted' if restart else '')
            output = a.root/name
            if output.exists():
                raise ValueError('No overwritten/implicitly adopted training output')
            command = [sys.executable, 'tools/cfp_train_pixel_masked_identity.py', '--config', str(config_path),
                       '--crops', str(a.root/'identity_pilot_crop_cache_v1'), '--features', str(a.root/'dinov2_two_crop_evidence_v1'),
                       '--index', str(a.root/'identity_two_crop_index_v1'), '--reference', '/data3/liuyeqiang/CFP-FishMOT_references/phase7/DINOv2',
                       '--weights', str(a.root/'external_models/dinov2_vits14_v1/dinov2_vits14_pretrain.pth'),
                       '--source-proof', str(a.root/'dinov2_small_inference_source_smoke_v1.json'), '--output', str(output),
                       '--outer', 'SN-015', '--mode', mode, '--local-count', '4', '--seed', str(seed)]
            variants = [('stop', ['--stop-after-epoch', '2']), ('resume', ['--resume'])] if restart else [('continuous', [])]
            for label, suffix in variants:
                log_path = Path(str(output)+f'.{label}.log')
                with log_path.open('x') as log:
                    child = subprocess.Popen(command+suffix, stdout=log, stderr=subprocess.STDOUT)
                    launch = dict(output=str(output), phase=label, pid=child.pid, command=command+suffix, log=str(log_path))
                    state['launches'].append(launch); state['active'] = launch; write_json(a.state, state)
                    code = child.wait()
                launch.update(exit_code=code, log_sha256=sha256(log_path)); write_json(a.state, state)
                if code:
                    raise RuntimeError(f'Actual masked fit failed exit{code}: {output}')
            report = json.loads((output/'manifest.json').read_text())
            if report['status'] != 'COMPLETE' or report['guard']['outer'] != 'SN-015' or report['guard']['inner'] != 'BT-001':
                raise ValueError('Actual excluded-fold fixed-final completion required')
            state['completed'].append(dict(output=str(output), manifest_sha256=sha256(output/'manifest.json'), final_sha256=sha256(output/'final.pt')))
            state['active'] = None; write_json(a.state, state)
            if restart:
                continuous = a.root/'pixel_masked_identity_pixel_masked_SN-015_k4_s0_v1'
                subprocess.run([sys.executable, 'tools/cfp_verify_pixel_masked_restart.py', '--continuous', str(continuous),
                                '--restarted', str(output), '--stop-log', str(output)+'.stop.log', '--launch-proof', str(a.state),
                                '--output', str(a.root/'pixel_masked_identity_exact_restart_v1.json')], check=True)
        state['status'] = 'COMPLETE_FIRST_FOLD_PIXEL_MASKED_CONTROLS'; write_json(a.state, state)
    except Exception as error:
        state.update(status='HALTED_ACTUAL_FAILURE', error=str(error)); write_json(a.state, state); raise


if __name__ == '__main__':
    main()
