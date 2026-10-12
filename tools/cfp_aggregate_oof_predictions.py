#!/usr/bin/env python3
"""Aggregate one complete prediction per heldout video without copying assets."""
import argparse
import json
from pathlib import Path

from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--family', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan['state'] != 'FROZEN' or args.output.exists():
        raise ValueError('Frozen plan/new immutable aggregate required')
    jobs = [j for j in plan['jobs'] if j['family'] == args.family]
    sequences = json.loads(Path('configs/cfp/protocol_v2.json').read_text())['train']
    if sorted(j['sequence'] for j in jobs) != sorted(sequences):
        raise ValueError('Exactly one complete outer prediction for every training video')
    report = dict(status='RUNNING', family=args.family, videos={}, plan_sha256=sha256(args.plan),
                  scope='One model per outer fold, not a single deployed model. Conditional association OOF; frozen backbone pretraining exposure remains.')
    args.output.mkdir()
    report['sources'] = snapshot_sources(args.output, [__file__, args.plan])
    for job in jobs:
        sequence, run = job['sequence'], Path(job['output'])
        manifest = json.loads((run / 'manifest.json').read_text())
        if manifest['status'] != 'COMPLETE' or list(manifest['videos']) != [sequence]:
            raise ValueError('Whole outer-video replay required')
        producer = Path(manifest['config']['predictor_checkpoint'])
        metadata = json.loads(producer.with_name('manifest.json').read_text())
        if metadata['config']['validation_video'] != sequence or sequence in metadata['config']['fit_videos']:
            raise ValueError('Producer outer-video leakage')
        if sha256(producer) != manifest['checkpoint_sha256']:
            raise ValueError('Producer checkpoint drift')
        prediction = run / (sequence + '.txt')
        if sha256(prediction) != manifest['videos'][sequence]['prediction_sha256']:
            raise ValueError('Prediction drift')
        (args.output / prediction.name).symlink_to(prediction.resolve())
        report['videos'][sequence] = dict(source=str(prediction), prediction_sha256=sha256(prediction),
            replay_manifest_sha256=sha256(run / 'manifest.json'), producer_checkpoint_sha256=sha256(producer))
    report['status'] = 'COMPLETE'
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
