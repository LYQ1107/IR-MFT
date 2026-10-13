#!/usr/bin/env python3
"""Verify an actual CPU JEPA teacher/student/optimizer/RNG restart."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    ap = argparse.ArgumentParser()
    for name in ('continuous', 'restarted', 'stop-log', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists() or args.continuous.resolve()==args.restarted.resolve():
        raise ValueError('Distinct actual runs and new proof required')
    stopped = []
    for line in args.stop_log.read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get('status')=='CHECKPOINTED_EARLY_STOP':
            stopped.append(value)
    if stopped != [dict(status='CHECKPOINTED_EARLY_STOP', epoch=2)]:
        raise ValueError('Actual registered epoch2 checkpoint termination log required')
    reports = [json.loads((p/'manifest.json').read_text()) for p in (args.continuous, args.restarted)]
    saved = [torch.load(p/'last.pt', map_location='cpu', weights_only=False) for p in (args.continuous, args.restarted)]
    if (any(r['status']!='COMPLETE' for r in reports) or saved[0]['guard']!=saved[1]['guard']
            or saved[0]['epoch']!=saved[1]['epoch'] or saved[0]['epoch']!=8
            or saved[0]['guard']['mode']!='full' or saved[0]['guard']['seed']!=1
            or saved[0]['guard']['cuda_visible_devices']!=''
            or not saved[0]['guard']['deterministic_algorithms']
            or any(len(r['epochs'])!=8 for r in reports)):
        raise ValueError('Actual completed full-seed1 CPU runs with identical registered source/data/runtime required')
    counts = dict(tensor_leaves=0, array_leaves=0, scalar_leaves=0)
    differences = []

    def compare(a, b, path):
        if type(a) is not type(b):
            differences.append(path+':type')
        elif isinstance(a, torch.Tensor):
            counts['tensor_leaves'] += 1
            if a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a, b):
                differences.append(path+':tensor')
        elif isinstance(a, np.ndarray):
            counts['array_leaves'] += 1
            if a.dtype!=b.dtype or a.shape!=b.shape or not np.array_equal(a, b):
                differences.append(path+':array')
        elif isinstance(a, dict):
            if a.keys()!=b.keys():
                differences.append(path+':keys')
            for k in a.keys() & b.keys():
                compare(a[k], b[k], path+'/'+str(k))
        elif isinstance(a, (tuple, list)):
            if len(a)!=len(b):
                differences.append(path+':length')
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, path+'/'+str(i))
        else:
            counts['scalar_leaves'] += 1
            if a!=b:
                differences.append(path+':value')
    fields = ('student', 'teacher', 'optimizer', 'scheduler', 'rng_sampler', 'rng_target_mask',
              'rng_torch', 'rng_numpy', 'rng_python')
    for field in fields:
        compare(saved[0][field], saved[1][field], field)
    for i, (a, b) in enumerate(zip(reports[0]['epochs'], reports[1]['epochs'])):
        for name in ('epoch', 'attempted_steps', 'optimizer_updates', 'teacher_updates', 'sampled_input_sha256', 'metrics_mean'):
            compare(a[name], b[name], f'epoch{i+1}/{name}')
    for field in ('first_gradient_norms', 'student_parameters_changed', 'local_encoder_l2_change',
                  'teacher_parameters_changed', 'teacher_has_gradient', 'actual_teacher_updates'):
        compare(reports[0][field], reports[1][field], field)
    # Identity/discrimination/support exactly compare; walltime and each run's
    # own absolute trace-file path are intentionally excluded.
    for video in reports[0]['evaluations']:
        for history in ('gt', 'detector'):
            for current in ('gt', 'detector'):
                for gap in ('1', '25', '100'):
                    a = reports[0]['evaluations'][video]['protocol'][history][current][gap]
                    b = reports[1]['evaluations'][video]['protocol'][history][current][gap]
                    for name in ('retrieval', 'discrimination', 'input_candidate_observations',
                            'observable_candidate_observations', 'unknown_candidate_observations', 'predicted_feature_variance'):
                        compare(a[name], b[name], '/'.join([video, history, current, gap, name]))
                    compare(a['query_support']['sha256'], b['query_support']['sha256'], '/'.join([video, history, current, gap, 'support_sha256']))
    proof = dict(status='EXACT_LOCAL_DENSE_RESTART_STATE_VERIFIED' if not differences else 'LOCAL_DENSE_RESTART_DIFFERENCES',
        guard=saved[0]['guard'], counts=counts, differences=differences, compared_state_fields=fields,
        continuous=str(args.continuous), restarted=str(args.restarted),
        manifest_sha256=[sha256(p/'manifest.json') for p in (args.continuous, args.restarted)],
        stop_log_sha256=sha256(args.stop_log), sources=snapshot_sources(args.output.with_suffix('.sources'), [__file__]),
        scope='Actual full localJEPA seed1 CPU uninterrupted8epochs versusepoch2exit/freshprocessresume; student/independentEMAteacher/updatecount/AdamW/scheduler/all CPU+sampler+mask RNG, losses, exact observed-input digest and identity/support metrics. Excludes walltime/container bytes/filepaths; no crosshardware guarantee.')
    write_json(args.output, proof)
    print(json.dumps(dict(status=proof['status'], differences=len(differences), counts=counts)))
    if differences:
        raise RuntimeError('Actual localJEPA restart is not equivalent')


if __name__=='__main__':
    main()
