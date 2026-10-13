#!/usr/bin/env python3
"""Two CPU workers confirm training-only-predictor/context identity controls."""
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
    for name in ('features', 'index', 'legacy-features', 'config', 'output-root', 'state'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--worker', type=int, choices=(0, 1), required=True)
    args = ap.parse_args()
    if args.state.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or args.output_root.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError('New serial CPU worker, outsideGit and no CUDA allocation required')
    config = json.loads(args.config.read_text())
    assignments = {0: ['full', 'local_contrastive', 'no_ema', 'global_gru'],
                   1: ['no_prediction', 'local_temporal', 'no_id', 'local_attention', 'global']}
    modes = assignments[args.worker]
    if config['state']!='FROZEN' or set(config['modes']) != set(assignments[0]+assignments[1]):
        raise ValueError('All nine registered modes, three seeds and fixed firstfold required')
    sources = [__file__, 'tools/cfp_train_local_dense_context_identity.py', 'cfp/local_dense_jepa.py',
        'cfp/local_dense_data.py', 'cfp/local_dense_training.py', 'cfp/local_dense_evaluation.py',
        'cfp/identity_evidence.py', 'cfp/provenance.py', 'cfp/local_dense_context_identity.py', str(args.config)]
    hashes = {p: sha256(p) for p in sources}
    state = dict(status='RUNNING', worker=args.worker, modes=modes, completed=[], active=None,
        pid=os.getpid(), total_runs=len(modes)*len(config['seed_plan']),
        sources=snapshot_sources(args.state.with_suffix('.sources'), sources), source_guards=hashes,
        scope='Separate contextidentity V2 firstfoldK4 nine-label/three-seed CPU control; no_prediction/local_temporal are exact aliases, not independent confirmations. No allfold/K8/native/allrawtime/JEPA gain or finalgoal completion claim.')
    write_json(args.state, state)
    try:
        for seed in config['seed_plan']:
            for mode in modes:
                if any(sha256(p)!=digest for p, digest in hashes.items()):
                    raise ValueError('Guarded sources changed while actual CPU worker active')
                if shutil.disk_usage(args.output_root).free < 42*2**30:
                    raise OSError('40GiB reserve plus parallel checkpoint margin')
                name = f'local_dense_context_{mode}_{config["first_outer"]}_k{config["first_count"]}_s{seed}_v2'
                directory = args.output_root/name
                state['active'] = dict(name=name, output=str(directory))
                write_json(args.state, state)
                if directory.exists():
                    manifest = json.loads((directory/'manifest.json').read_text())
                    guard = manifest['guard']
                    if (manifest['status']!='COMPLETE' or guard['mode']!=mode or guard['seed']!=seed
                            or guard['outer']!=config['first_outer'] or guard['local_count']!=config['first_count']
                            or guard['config_sha256']!=hashes[str(args.config)]
                            or guard['feature_manifest_sha256']!=sha256(args.features/'manifest.json')
                            or any(sha256(p)!=value for p, value in guard['source_sha256'].items())
                            or sha256(directory/'final.pt')!=manifest['checkpoint_sha256']):
                        raise ValueError('Existing producer incomplete or changed; inspect real process/checkpoint before any restart: '+name)
                else:
                    command = [sys.executable, 'tools/cfp_train_local_dense_context_identity.py']
                    for key in ('features', 'index', 'legacy-features', 'config'):
                        command += ['--'+key, str(getattr(args, key.replace('-', '_')))]
                    command += ['--output', str(directory), '--outer', config['first_outer'], '--mode', mode,
                                '--seed', str(seed), '--local-count', str(config['first_count'])]
                    with (args.output_root/(name+'.log')).open('x') as log:
                        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                        state['active']['pid'] = child.pid
                        write_json(args.state, state)
                        code = child.wait()
                    if code:
                        raise RuntimeError('Actual child terminal exit '+str(code)+': '+name)
                manifest = json.loads((directory/'manifest.json').read_text())
                if manifest['status']!='COMPLETE' or manifest['cuda_initialized']:
                    raise ValueError('Actual completed CPU-only run required')
                state['completed'].append(dict(name=name, output=str(directory),
                    manifest_sha256=sha256(directory/'manifest.json'), checkpoint_sha256=manifest['checkpoint_sha256']))
                state['active'] = None
                write_json(args.state, state)
                print(json.dumps(dict(name=name, completed=len(state['completed']))), flush=True)
        state['status'] = 'COMPLETE_REGISTERED_LOCAL_DENSE_FIRST_FOLD_CONTROLS'
        write_json(args.state, state)
    except Exception as error:
        state.update(status='HALTED_ACTUAL_FAILURE', error=str(error))
        write_json(args.state, state)
        raise


if __name__ == '__main__':
    main()
