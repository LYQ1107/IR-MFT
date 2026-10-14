"""Training-only verified old-set withholding; never infer newborn labels."""
import numpy as np
import torch
from .joint_onpolicy_data import query_batch
from .joint_set_supervision import joint_identity_target_sets


def augmented_query_batch(row, banks, codes_rng, subset_rng, config):
    inputs, targets = query_batch(row, banks, codes_rng, config['capacity'])
    q = banks[row['video']]['queries']; i = row['position']
    lo, hi = q['track_offsets'][i:i+2]; dl, dh = q['detection_offsets'][i:i+2]
    track_ids = q['track_identity'][lo:hi].copy()
    current_ids = q['current_identity'][dl:dh].copy()
    current_ids[~inputs['detection_valid'].numpy()] = -1
    metadata = dict(kind='actual_full_native_set', kept_slots=list(range(len(track_ids))))
    requested = subset_rng.random() < config['verified_subset_probability']
    known_ids = np.unique(track_ids[track_ids >= 0])
    eligible = [int(identity) for identity in known_ids if (current_ids == identity).sum() == 1]
    if requested and len(known_ids) >= 2 and eligible:
        withheld = int(subset_rng.choice(eligible))
        # All retained slots have verified consistent past identities distinct
        # from the withheld known current fish. Unknown slots cannot justify NULL.
        keep = np.flatnonzero((track_ids >= 0) & (track_ids != withheld))
        if not len(keep):
            raise ValueError('Verified subset must retain a real wrong-ID choice')
        inputs = dict(inputs)
        for key in ['history', 'times', 'valid', 'frame', 'track_state', 'codes', 'geometry', 'appearance']:
            inputs[key] = inputs[key][keep]
        targets = joint_identity_target_sets(torch.from_numpy(track_ids[keep]), torch.from_numpy(current_ids))
        withheld_rows = np.flatnonzero(current_ids == withheld)
        if not targets['positives'][withheld_rows, -1].all():
            raise ValueError('Only logically verified no-supplied-old-match targets allowed')
        metadata = dict(kind='verified_known_subset_withholding', kept_slots=keep.tolist(),
                        withheld_current_queries=len(withheld_rows))
    return inputs, targets, metadata
