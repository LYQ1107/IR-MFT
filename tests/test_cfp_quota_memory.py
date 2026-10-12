from types import SimpleNamespace

import numpy as np

from cfp.hypothesis_bank import isolated_state
from cfp.lifecycle_crossover import make_cfp_lifecycle
from cfp.online_recovery_write_quota import install_quota_risk_memory
from yolox.tracker import matching
from yolox.tracker.basetrack import BaseTrack


def test_quota_actual_matches_and_rng_and_branch_state_are_isolated():
    BaseTrack._count = 0
    original_assignment = matching.linear_assignment
    tracker = install_quota_risk_memory(make_cfp_lifecycle(SimpleNamespace(), 'gru', memory_mode='last'),
                                       None, None, fraction=.1, random_order=True)
    features = np.eye(10, 512, dtype=np.float32)
    detections = np.array([[i * 60, 0, i * 60 + 30, 20, .9] for i in range(10)])
    for frame in [1, 2]:
        tracker.update(detections, features, np.arange(10) + frame * 10)
    assert tracker.phase_vi_memory_counts['provisional_rejected'] == 1
    assert tracker.phase_vi_memory_counts['accepted'] == 9
    assert tracker.quota_counts['scored_matched'] == 10
    assert matching.linear_assignment is original_assignment
    branch = isolated_state(tracker)
    branch.update(detections, features, np.arange(10) + 30)
    assert branch.quota_counts['rejected_budget'] == 2
    assert tracker.quota_counts['rejected_budget'] == 1
    assert branch.quota_rng is not tracker.quota_rng
    assert matching.linear_assignment is original_assignment
