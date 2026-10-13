"""Compare real final model/optimizer/RNG state after a fresh-process restart."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cfp.provenance import sha256, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--continuous', type=Path, required=True)
    ap.add_argument('--restarted', type=Path, required=True)
    ap.add_argument('--stop-log', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--producer', choices=('visual_resnet', 'dinov2_fish'), default='visual_resnet')
    args = ap.parse_args()
    if args.output.exists() or args.continuous.resolve() == args.restarted.resolve():
        raise ValueError('Distinct real completed runs and new proof required')
    log = args.stop_log.read_text()
    if 'CHECKPOINTED_EARLY_STOP' not in log:
        raise ValueError('Actual deliberate checkpoint termination log required')
    stop_entries = []
    for line in log.splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and entry.get('status') == 'CHECKPOINTED_EARLY_STOP':
            stop_entries.append(entry)
    reports = [json.loads((p / 'manifest.json').read_text()) for p in (args.continuous, args.restarted)]
    if any(d['status'] != 'COMPLETE' for d in reports):
        raise ValueError('Both fresh process runs must really complete')
    saved = [torch.load(p / 'last.pt', map_location='cpu', weights_only=False) for p in (args.continuous, args.restarted)]
    if saved[0]['guard'] != saved[1]['guard'] or saved[0]['epoch'] != saved[1]['epoch']:
        raise ValueError('Same data/source/config/physical GPU/runtime/final epoch required')
    if len(reports[0]['epochs']) != len(reports[1]['epochs']) or len(reports[0]['epochs']) != saved[0]['epoch']:
        raise ValueError('Complete equal training epoch traces required')
    if args.producer == 'dinov2_fish' and (
            saved[0]['guard'].get('mode') != 'global' or saved[0]['guard'].get('seed') != 1
            or saved[0]['guard'].get('physical_devices') != '0'
            or saved[0]['epoch'] != 8 or stop_entries != [{'status': 'CHECKPOINTED_EARLY_STOP', 'epoch': 2}]):
        raise ValueError('Registered DINO proof is global seed1 GPU0 epoch2 exit versus8epoch continuous')
    execution_proofs = []
    if args.producer == 'dinov2_fish' and reports[0]['config'].get('determinism'):
        executions = []
        for directory, report in zip((args.continuous, args.restarted), reports):
            path = directory.with_suffix('.execution.json')
            execution = json.loads(path.read_text())
            if (execution['status'] != 'COMPLETE' or execution['training_manifest_sha256'] != sha256(directory/'manifest.json')
                    or not execution['guard']['deterministic_algorithms']
                    or execution['guard']['cublas_workspace_config'] != ':4096:8'):
                raise ValueError('Actual same-policy deterministic launch proof required')
            executions.append(execution)
            execution_proofs.append(dict(path=str(path), sha256=sha256(path), proof=execution))
        if executions[0]['guard'] != executions[1]['guard'] or [e['resume'] for e in executions[0]['launches']] != [False] or [e['resume'] for e in executions[1]['launches']] != [False, True]:
            raise ValueError('Continuous launch versus real stopped and fresh resumed launches required')
    differences = []
    counts = {'tensor_leaves': 0, 'array_leaves': 0, 'scalar_leaves': 0}

    def compare(a, b, path):
        if type(a) is not type(b):
            differences.append(path + ':type')
        elif isinstance(a, torch.Tensor):
            counts['tensor_leaves'] += 1
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                differences.append(path + ':tensor')
        elif isinstance(a, np.ndarray):
            counts['array_leaves'] += 1
            if a.dtype != b.dtype or not np.array_equal(a, b):
                differences.append(path + ':array')
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                differences.append(path + ':keys')
            for k in a.keys() & b.keys():
                compare(a[k], b[k], path + '/' + str(k))
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                differences.append(path + ':length')
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, path + '/' + str(i))
        else:
            counts['scalar_leaves'] += 1
            if a != b:
                differences.append(path + ':value')

    fields = ['model', 'optimizer', 'scheduler', 'scaler', 'rng_sampler', 'rng_augmentation', 'rng_python', 'rng_numpy', 'rng_torch', 'rng_cuda']
    for field in fields:
        compare(saved[0][field], saved[1][field], field)
    for i, (a, b) in enumerate(zip(reports[0]['epochs'], reports[1]['epochs'])):
        # Wall-clock time and pickle container bytes are not deterministic state.
        for field in ('epoch', 'loss_ce_temporal_frame', 'attempted_steps', 'amp_overflow_skips', 'optimizer_updates'):
            compare(a[field], b[field], f'epoch{i + 1}/{field}')
    compare(reports[0]['evaluations'], reports[1]['evaluations'], 'retrieval')
    if args.producer == 'dinov2_fish':
        for field in ('first_step_gradient_norms', 'first_step_gradients_finite',
                'first_finite_step_gradient_norms', 'first_finite_step', 'frozen_prefix_unchanged',
                'trained_backbone_l2_change', 'trainable_parameters', 'trainable_backbone_parameters',
                'trainable_head_parameters', 'classifier_keys'):
            compare(reports[0][field], reports[1][field], field)
        for i, (a, b) in enumerate(zip(reports[0]['epochs'], reports[1]['epochs'])):
            compare(a['sample_sameframe_temporal_coverage'], b['sample_sameframe_temporal_coverage'], f'epoch{i + 1}/coverage')
    proof = dict(status='EXACT_RESTART_STATE_VERIFIED' if not differences else 'RESTART_STATE_DIFFERENCES',
        continuous=str(args.continuous), restarted=str(args.restarted), epoch=saved[0]['epoch'], guard=saved[0]['guard'],
        compared_state_fields=fields, counts=counts, differences=differences, stop_log_sha256=sha256(args.stop_log),
        manifest_sha256=[sha256(p / 'manifest.json') for p in (args.continuous, args.restarted)],
        producer=args.producer,
        execution_proofs=execution_proofs,
        scope='Actual '+args.producer+' epoch2 early exit then fresh-process resume versus uninterrupted seed1/global8epoch run on same physicalGPU0. Tensor/optimizer/scaler/all RNG, training losses/update counts and retrieval compared. '+('Strict deterministic algorithms and :4096:8 cuBLAS workspace; this does not retroactively certify original uncontrolled-workspace twelve fits. ' if execution_proofs else '')+'No crosshardware determinism claim.')
    write_json(args.output, proof)
    print(json.dumps({'status': proof['status'], 'differences': len(differences), **counts}))
    if differences:
        raise RuntimeError('Restart is not exactly equivalent; see proof')


if __name__ == '__main__':
    main()
