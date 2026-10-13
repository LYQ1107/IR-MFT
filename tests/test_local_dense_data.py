from collections import defaultdict

import numpy as np
import torch

from cfp.local_dense_data import LocalSceneBank


def synthetic_bank():
    bank = object.__new__(LocalSceneBank)
    bank.count = 4
    records = [dict(frame=f, identity=i, view=v) for f, i, v in
        [(1, 1, 'gt'), (2, 7, 'gt'), (5, 1, 'gt'), (9, 1, 'gt'),
         (12, 1, 'gt'), (15, 1, 'gt'), (5, 1, 'detector'), (12, -1, 'detector')]]
    frames = {v: defaultdict(list) for v in ('gt', 'detector')}
    ids = {v: defaultdict(lambda: defaultdict(list)) for v in frames}
    for index, row in enumerate(records):
        frames[row['view']][row['frame']].append(index)
        if row['identity'] >= 0:
            ids[row['view']][row['identity']][row['frame']].append(index)
    histories = {v: {i: sorted((f, rows[0]) for f, rows in m.items()) for i, m in mapping.items()}
                 for v, mapping in ids.items()}
    bank.videos = {'v': dict(records=records, frames=frames, identities=ids, histories=histories,
        arrays={'local4': np.broadcast_to(np.arange(8)[:, None, None], (8, 4, 384)).astype(np.float32),
                'visibility4': np.ones((8, 4), np.float32), 'cls': np.zeros((8, 384), np.float32),
                'valid': np.ones(8, bool)})}
    return bank


def test_target_current_future_excluded_unknown_preserved_absent_track_retained():
    bank = synthetic_bank()
    assert bank.evaluation_ids('v', 12) == [1, 7]
    scene = bank.scene('v', 12, [1, 7], length=4)
    assert scene['history_indices'] == [[0, 2, 3], [1]]
    assert scene['current_indices'] == [7] and scene['candidate_ids'].tolist() == [-1]
    x = scene['inputs']
    assert torch.all(x['times'][x['history_valid']] < 12)
    assert not any('target' in key or 'identity' in key or 'track_id' in key for key in x)
    target = bank.scene('v', 12, [1], length=4, include_targets=True)
    assert target['target_indices'] == [4] and 4 not in target['history_indices'][0]
    assert 5 not in target['history_indices'][0]


def test_verified_detector_history_has_no_gt_fallback_or_unknown_relabeling():
    bank = synthetic_bank()
    scene = bank.scene('v', 12, [1, 7], history_view='detector')
    assert scene['history_indices'] == [[6], []]
    assert scene['inputs']['history_valid'].sum().item() == 1
    mixed = bank.scene('v', 12, [1], history_view='mixed', rng=np.random.default_rng(5))
    assert all(bank.videos['v']['records'][i]['frame'] < 12 for i in mixed['history_indices'][0])
