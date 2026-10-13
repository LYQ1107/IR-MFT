#!/usr/bin/env python3
"""Fit/inner-calibration residual expressiveness, never outer-score tuning."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from cfp.joint_identity_decoder import JointFishIDDecoder, bounded_joint_cost
from cfp.joint_identity_training_data import load_joint_training, joint_frame_batch
from cfp.provenance import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Immutable targeted diagnostic required')
    meta = json.loads((args.model_root / 'manifest.json').read_text())
    checkpoint = args.model_root / 'epoch8.pt'
    if meta['status'] != 'COMPLETE' or sha256(checkpoint) != meta['checkpoint_sha256']:
        raise ValueError('Frozen complete first probe required')
    config = meta['config']
    records, banks, _ = load_joint_training(args.queries)
    if sha256(args.queries / 'manifest.json') != meta['guard']['query_manifest_sha256']:
        raise ValueError('Training query source differs')
    model = JointFishIDDecoder(dimension=config['dimension'], hidden=config['hidden'], heads=config['heads'],
        capacity=config['capacity'], architecture=meta['guard']['architecture']).eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    torch.set_num_threads(2)
    report = dict(status='RUNNING', checkpoint_sha256=sha256(checkpoint),
        query_manifest_sha256=sha256(args.queries / 'manifest.json'), source_sha256=sha256(__file__), videos={},
        scope='Fit and inner-calibration first-probe rows only; no outer/historical-test tuning. Scores measure bounded-route expressiveness at fixed native threshold, not online GT or final tracking performance.')
    write_json(args.output, report)
    rng = np.random.default_rng(1000)
    with torch.inference_mode():
        for video in config['fit_videos'] + config['calibration_videos']:
            values = defaultdict(list)
            counts = defaultdict(int)
            for record in (r for r in records if r['video'] == video):
                inputs, supervision = joint_frame_batch(record, banks, 'cpu', rng, config['capacity'])
                output = model(**inputs)
                q, i = banks[video]['queries'], record['position']
                lo, hi = q['edge_offsets'][i:i + 2]
                baseline = torch.from_numpy(q['baseline_cost'][lo:hi].reshape(inputs['geometry'].shape).copy())
                edited = bounded_joint_cost(baseline, output['residual'], output['legal'], config['maximum_delta'])
                delta = edited - baseline
                values['all_legal_delta'].extend(delta[output['legal']].tolist())
                counts['frames'] += 1
                counts['edges'] += baseline.numel()
                counts['threshold_crossings'] += int(((baseline <= config['unmatched_cost']) != (edited <= config['unmatched_cost'])).sum())
                for j, target in enumerate(supervision['target'].tolist()):
                    if target < 0:
                        continue
                    n = len(inputs['history'])
                    if target < n and output['legal'][target, j]:
                        values['known_correct_delta'].append(float(delta[target, j]))
                        values['known_correct_baseline_cost'].append(float(baseline[target, j]))
                    elif target == n and n:
                        counts['known_no_old_match_detections'] += 1
                        # A bounded route cannot reject every track if even the
                        # best old cost remains below threshold at maximum edit.
                        possible = bool((baseline[:, j] + config['maximum_delta'] > config['unmatched_cost']).all())
                        counts['null_expressible_by_uniform_maximum_delta'] += int(possible)
                    known_wrong = supervision['classes'][j, :n].clone()
                    if target < n:
                        known_wrong[target] = False
                    known_wrong &= output['legal'][:, j]
                    values['known_wrong_delta'].extend(delta[:, j][known_wrong].tolist())
            summaries = {key: dict(count=len(v), mean=float(np.mean(v)),
                                   quantiles=np.quantile(v, [0, .1, .5, .9, 1]).tolist()) if v else dict(count=0)
                         for key, v in values.items()}
            report['videos'][video] = dict(counts=counts, summaries=summaries)
            write_json(args.output, report)
            print(video, dict(counts), flush=True)
    report['status'] = 'COMPLETE'
    write_json(args.output, report)


if __name__ == '__main__':
    main()
