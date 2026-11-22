#!/usr/bin/env python3
"""Register concrete native association thresholds before any Dev replay."""
import json
from pathlib import Path

from tools.run_pseudo_ir_comparison import dump, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    folder = ROOT/'configs/perceptual_native_query_dev_20261123'
    if folder.exists():
        raise ValueError('Native query Dev configurations already registered')
    folder.mkdir()
    outputs = {}
    for front in ['new', 'rgb']:
        for cue in ['ir', 'rgb']:
            for seed in [0, 1, 2]:
                core = json.loads((ROOT/f'configs/perceptual_pair_selected_20261123/K_{front}_rgb_s{seed}.json').read_text())
                for epoch in [6, 12, 18]:
                    name = f'N_{front}_{cue}_s{seed}_e{epoch:02d}'
                    cfg = dict(core, id=name, backend='native_query', native_cue=cue, native_seed=seed,
                               native_query_checkpoint=str(work/f'native_query/{front}_{cue}_s{seed}/epoch{epoch:02d}.pt'),
                               query_ranking=True, query_weight=.8, query_minimum_probability=.8,
                               query_margin=.1, query_opening_probability=.95, query_null=True,
                               prebirth_recovery=True, identity_retention=128,
                               prebirth_probability=.9, prebirth_margin=.1, prebirth_minimum_score=.3,
                               clear_identity_anchor=False,
                               registration='Frozen thresholds before native Dev replay. Compare all current proposals and null; keep RGB point fallback. Prebirth consults only previously returned IDs before a new ID is first returned. No Test selection.')
                    path = folder/(name+'.json');dump(path, cfg)
                    outputs[name] = dict(path=str(path), sha256=sha256(path))
                    if cue == 'ir' and epoch == 18:
                        name += '_zero'
                        cfg = dict(cfg, id=name, disable_native_increment=True)
                        path = folder/(name+'.json');dump(path, cfg)
                        outputs[name] = dict(path=str(path), sha256=sha256(path))
    dump(work/'native_query_dev_registration.json', dict(status='REGISTERED_BEFORE_NATIVE_DEV_REPLAY',
         outputs=outputs, count=len(outputs), source_sha256=sha256(__file__),
         plan_sha256=sha256(ROOT/'configs/perceptual_native_query_v2_20261123.json')))
    print(json.dumps(dict(configurations=len(outputs))))


if __name__ == '__main__':
    main()
