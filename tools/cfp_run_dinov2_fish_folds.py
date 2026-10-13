#!/usr/bin/env python3
"""Bounded serial jobs per approved GPU; all preregistered folds/seeds/K."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    ap = argparse.ArgumentParser()
    for name in ('config', 'diagnostic-config', 'crops', 'features', 'index', 'reference',
            'weights', 'source-proof', 'frozen-diagnostic', 'output-root', 'state'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--mode', required=True)
    args = ap.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    mode_gpu = {'global': '0', 'simple': '1', 'global_capacity': '2', 'patch_attention': '3'}
    if args.mode not in mode_gpu or physical != mode_gpu[args.mode] or args.state.exists():
        raise ValueError('One fixed mode/GPU0-3 and new immutable driver state required')
    config = json.loads(args.config.read_text())
    if config['state'] != 'FROZEN' or args.output_root.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError('Frozen plan and outside-Git output required')
    sources = [str(Path(__file__)), 'tools/cfp_train_dinov2_fish_identity.py',
        'tools/cfp_evaluate_dinov2_fish_identity.py', 'tools/cfp_evaluate_dinov2_crop_evidence.py',
        'cfp/dinov2_fish_encoder.py', 'cfp/dinov2_fish_data.py', 'cfp/dinov2_identity.py',
        'cfp/local_identity_encoder.py', 'cfp/local_identity_training.py', 'cfp/crop_retrieval.py',
        'cfp/reid.py', 'cfp/identity_evidence.py', 'cfp/provenance.py', str(args.config), str(args.diagnostic_config)]
    hashes = {p: sha256(p) for p in sources}
    # A full fit writes at most about0.5GiB here; 0.6GiB/job plus temporary
    # checkpoint margin is reserved before launching all four mode drivers.
    all_remaining = 0
    for mode in config['modes']:
        for k in config['local_counts']:
            for outer in config['videos']:
                for seed in config['seed_plan']:
                    directory = args.output_root/f'dinov2_fish_{mode}_{outer}_k{k}_s{seed}_v2'
                    if not (directory/'manifest.json').exists() or json.loads((directory/'manifest.json').read_text())['status'] != 'COMPLETE':
                        all_remaining += 1
    if shutil.disk_usage(args.output_root).free < (41.2+.6*all_remaining)*2**30:
        raise OSError('Reserve40GiB plus preregistered remaining-job and temporary-checkpoint estimate unavailable')
    state = dict(status='RUNNING', mode=args.mode, physical_gpu=physical,
        config_sha256=sha256(args.config), sources=snapshot_sources(args.state.with_suffix('.sources'), sources),
        source_guards=hashes, declared_total_runs_per_mode=36, completed=[], active=None,
        scope='All six leave-two-video-out folds, K4/K8 and seeds0/1/2; fixed same recipe without reading outer values for selection. Child failure halts this driver and preserves files. No native-MOT/JEPA completion claim.')
    write_json(args.state, state)
    try:
        outer_order = [config['first_outer']]+[v for v in config['videos'] if v != config['first_outer']]
        for k in config['local_counts']:
            for outer in outer_order:
                for seed in config['seed_plan']:
                    if any(sha256(p) != value for p, value in hashes.items()):
                        raise ValueError('Guarded training/diagnostic sources changed while driver active')
                    if shutil.disk_usage(args.output_root).free < 42*2**30:
                        raise OSError('40GiB reserve plus parallel temporary-checkpoint margin')
                    name = f'dinov2_fish_{args.mode}_{outer}_k{k}_s{seed}_v2'
                    directory = args.output_root/name
                    diagnostic = args.output_root/(name+'.diagnostic.json')
                    state['active'] = dict(name=name, phase='training', output=str(directory))
                    write_json(args.state, state)
                    if directory.exists():
                        manifest = json.loads((directory/'manifest.json').read_text())
                        if (manifest['status'] != 'COMPLETE' or manifest['guard']['mode'] != args.mode
                                or manifest['guard']['outer'] != outer or manifest['guard']['seed'] != seed
                                or manifest['guard']['local_count'] != k or manifest['guard']['physical_devices'] != physical
                                or manifest['guard']['config_sha256'] != hashes[str(args.config)]
                                or manifest['guard']['crops_sha256'] != sha256(args.crops/'manifest.json')
                                or manifest['guard']['features_sha256'] != sha256(args.features/'manifest.json')
                                or manifest['guard']['index_sha256'] != sha256(args.index/'manifest.json')
                                or manifest['guard']['pretrained_sha256'] != sha256(args.weights)
                                or any(sha256(p) != value for p, value in manifest['guard']['source_sha256'].items())
                                or sha256(directory/'final.pt') != manifest['checkpoint_sha256']):
                            raise ValueError('Existing producer incomplete or changed; manual real-resume audit required: '+name)
                    else:
                        command = [sys.executable, 'tools/cfp_train_dinov2_fish_identity.py']
                        for flag in ('config', 'crops', 'features', 'index', 'reference', 'weights', 'source-proof'):
                            command += ['--'+flag, str(getattr(args, flag.replace('-', '_')))]
                        command += ['--output', str(directory), '--mode', args.mode, '--outer', outer,
                                    '--seed', str(seed), '--local-count', str(k)]
                        with (args.output_root/(name+'.log')).open('x') as log:
                            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                            state['active']['pid'] = child.pid
                            write_json(args.state, state)
                            code = child.wait()
                        if code:
                            raise RuntimeError('Actual child terminal exit '+str(code)+': '+name)
                    state['active'].update(phase='diagnostic', output=str(diagnostic))
                    write_json(args.state, state)
                    if diagnostic.exists():
                        measured = json.loads(diagnostic.read_text())
                        if (measured['status'] != 'COMPLETE_MATCHED_TRAINED_DINOV2_CROP_DIAGNOSTIC'
                                or measured['training_manifest_sha256'] != sha256(directory/'manifest.json')):
                            raise ValueError('Existing diagnostic incomplete or changed: '+name)
                    else:
                        command = [sys.executable, 'tools/cfp_evaluate_dinov2_fish_identity.py',
                            '--training', str(directory), '--frozen-features', str(args.features),
                            '--frozen-diagnostic', str(args.frozen_diagnostic), '--config', str(args.diagnostic_config),
                            '--output', str(diagnostic)]
                        with (args.output_root/(name+'.diagnostic.log')).open('x') as log:
                            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                            state['active']['pid'] = child.pid
                            write_json(args.state, state)
                            code = child.wait()
                        if code:
                            raise RuntimeError('Actual diagnostic terminal exit '+str(code)+': '+name)
                    state['completed'].append(dict(name=name, training=str(directory), diagnostic=str(diagnostic),
                        training_manifest_sha256=sha256(directory/'manifest.json'), diagnostic_sha256=sha256(diagnostic)))
                    state['active'] = None
                    write_json(args.state, state)
                    print(json.dumps(dict(completed=len(state['completed']), name=name)), flush=True)
        state['status'] = 'COMPLETE_ALL_PREREGISTERED_DINOV2_FISH_FOLDS'
        write_json(args.state, state)
    except Exception as error:
        state.update(status='HALTED_ACTUAL_FAILURE', error=str(error))
        write_json(args.state, state)
        raise


if __name__ == '__main__':
    main()
