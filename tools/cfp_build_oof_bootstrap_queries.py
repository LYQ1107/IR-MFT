#!/usr/bin/env python3
"""GT-clean training-only bootstrap queries without a trained trace producer.

This is explicitly clean supervision, NOT claimed as actual online history.
Video-excluded ordinary models will generate genuine accepted states afterwards.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from cfp.provenance import sha256, snapshot_sources, write_json
from cfp.recovery_supervision import physical_bank


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('New immutable bootstrap indices required')
    assets = Path('/data3/liuyeqiang/IR-MFT_assets/cfp')
    if shutil.disk_usage(assets).free < 40 * 2**30:
        raise OSError('40 GiB reserve')
    args.output.mkdir(parents=True)
    protocol = json.loads(Path('configs/cfp/phase_vi_protocol_v1.json').read_text())
    report = dict(status='RUNNING', partition='train', seed=0, stride=5, videos={},
        source_snapshots=snapshot_sources(args.output, [__file__, 'cfp/recovery_supervision.py',
            'cfp/supervision.py', 'configs/cfp/phase_vi_protocol_v1.json']),
        scope='Pure known physical past training supervision, NOT actual accepted online states. '
              'No historical teacher/model defines these queries. Current/future cannot enter past. '
              'Only whole-video fit subsets may optimize; frozen detector/encoder exposures remain. '
              'Unknown current labels are excluded from negative CE supervision.')
    write_json(args.output / 'manifest.json', report)
    for seq in protocol['train_videos']:
        bank = physical_bank(seq, 'train')
        rows, prior = [], {}
        for frame in bank['frames']:
            frame = int(frame)
            lo, hi = bank['offsets'][frame - 1:frame + 1]
            candidates = np.flatnonzero(bank['scores'][lo:hi] > .58) + lo
            if (frame - 1) % 5 == 0 and len(candidates) > 1:
                for positive, index in enumerate(candidates):
                    identity = int(bank['labels'][index])
                    observed = prior.get(identity, [])
                    if identity < 0 or not observed:
                        continue
                    indices = observed[-8:]
                    if indices[0] != observed[0]:
                        indices = [observed[0]] + indices[-7:]
                    padded = np.full(8, -1, np.int64); times = np.zeros(8, np.int64)
                    padded[:len(indices)] = indices; times[:len(indices)] = bank['frame_of_proposal'][indices]
                    rows.append(dict(frame=frame, identity=identity, history_indices=padded, history_frames=times,
                        candidates=candidates.copy(), candidate_known=bank['labels'][candidates] >= 0, positive=positive))
            # Training annotation selects only clean past identity cues. The
            # update happens after indexing the current query, never before.
            for index in range(lo, hi):
                identity = int(bank['labels'][index])
                if identity >= 0:
                    prior.setdefault(identity, []).append(index)
        arrays = {key: np.array([r[key] for r in rows]) for key in ('frame', 'identity', 'history_indices', 'history_frames', 'positive')}
        arrays['candidate_offsets'] = np.r_[0, np.cumsum([len(r['candidates']) for r in rows])]
        arrays['candidates'] = np.concatenate([r['candidates'] for r in rows])
        arrays['candidate_known'] = np.concatenate([r['candidate_known'] for r in rows])
        path = args.output / (seq + '.npz')
        np.savez_compressed(path, **arrays)
        fp = assets / 'phase2/causal_train_detection_features_postneck_seed0' / (seq + '.npz')
        provenance = json.loads(fp.with_suffix('.provenance.json').read_text())
        if sha256(fp) != provenance['features_sha256']:
            raise ValueError('Feature drift')
        report['videos'][seq] = dict(queries=len(rows), sha256=sha256(path),
            feature_sha256=sha256(fp), detection_sha256=bank['detection_sha256'], annotation_sha256=bank['annotation_sha256'])
        write_json(args.output / 'manifest.json', report)
        print(seq, len(rows), flush=True)
    report['status'] = 'COMPLETE'
    write_json(args.output / 'manifest.json', report)


if __name__ == '__main__':
    main()
