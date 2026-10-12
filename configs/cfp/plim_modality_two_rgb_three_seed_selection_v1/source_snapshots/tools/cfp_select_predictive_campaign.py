#!/usr/bin/env python3
"""Select fixed settings from complete, matched three-seed dev campaigns."""
import argparse
import json
from pathlib import Path

import numpy as np

from cfp.provenance import sha256, write_json, snapshot_sources


def select_campaigns(paths):
    groups = {}
    source_hashes = {}
    for seed, path in enumerate(paths):
        summary = json.loads(path.read_text())
        if summary['status'] != 'COMPLETE':
            raise ValueError('Selection requires every declared run to complete')
        source_hashes[str(seed)] = sha256(path)
        for run_id, entry in summary['runs'].items():
            if f'_seed{seed}_' not in run_id:
                raise ValueError('Campaign seed order differs')
            family, setting = run_id.rsplit('_w', 1)
            family = family.replace(f'_seed{seed}', '')
            key = (family, float(setting))
            groups.setdefault(key, {})[seed] = (run_id, entry['metrics']['COMBINED_SEQ'])
    families = {}
    for (family, setting), seeds in groups.items():
        if set(seeds) != {0, 1, 2}:
            raise ValueError('Incomplete matched three-seed setting')
        metrics = sorted(set.intersection(*(set(v[1]) for v in seeds.values())))
        metrics = [k for k in metrics if all(isinstance(v[1][k], (int, float)) for v in seeds.values())]
        trial = dict(appearance_weight=setting,
                     mean={k: float(np.mean([seeds[s][1][k] for s in range(3)])) for k in metrics},
                     sample_std={k: float(np.std([seeds[s][1][k] for s in range(3)], ddof=1)) for k in metrics},
                     runs=[seeds[s][0] for s in range(3)])
        families.setdefault(family, []).append(trial)
    result = {}
    for family, trials in families.items():
        if {v['appearance_weight'] for v in trials} != {.1, .3, .5}:
            raise ValueError('Every family must have its original three-setting budget')
        selected = max(trials, key=lambda v: tuple(v['mean'][k] for k in ('HOTA', 'AssA', 'IDF1')))
        result[family] = dict(trials=sorted(trials, key=lambda v: v['appearance_weight']), selected=selected)
    return dict(state='MEASURED_DEV_SELECTION', final_test_frozen=False,
                selection='Highest mean combined dev HOTA across three seeds; ties AssA then IDF1. No best-seed selection.',
                uncertainty='Sample standard deviation across three seeds, not video generalization confidence.',
                source_summaries=source_hashes, families=result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaries', type=Path, nargs=3, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Measured selections are immutable')
    result = select_campaigns(args.summaries)
    result['source_snapshots'] = snapshot_sources(args.output.with_suffix(''), [__file__])
    write_json(args.output, result)
    for family, entry in result['families'].items():
        selected = entry['selected']
        print(family, selected['appearance_weight'],
              {k: round(selected['mean'][k], 4) for k in ('HOTA', 'AssA', 'IDF1', 'IDSW')},
              'HOTA_std', round(selected['sample_std']['HOTA'], 4), flush=True)


if __name__ == '__main__':
    main()
