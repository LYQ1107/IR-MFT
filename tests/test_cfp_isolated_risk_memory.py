from types import SimpleNamespace

import numpy as np

from cfp.hypothesis_bank import isolated_state
from cfp.lifecycle_crossover import make_cfp_lifecycle
from cfp.online_recovery_write import install_risk_memory
from cfp.online_recovery_write_isolated import install_isolated_risk_memory
from yolox.tracker.basetrack import BaseTrack


def tracker(policy, risk):
    base = make_cfp_lifecycle(SimpleNamespace(), 'gru', memory_mode='last')
    return policy(base, None, None, constant_risk=risk)


def observations(frame):
    features = np.zeros((2, 512), np.float32)
    features[:, :2] = [[1., 0.], [0., 1.]]
    features[:, 2] = frame / 100
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    return np.array([[0, 0, 30, 20, .9], [80, 0, 110, 20, .9]]), features, np.array([2 * frame, 2 * frame + 1])


def test_isolated_writer_same_stream_as_original_provisional_writer():
    results = []
    for install in [install_risk_memory, install_isolated_risk_memory]:
        BaseTrack._count = 0
        state = tracker(install, 1.)
        rows = []
        for frame in range(1, 8):
            output = state.update(*observations(frame))
            rows.append([(t.track_id, t.current_proposal_index, list(t.identity_proposals)) for t in output])
        results.append((rows, state.phase_vi_memory_counts))
    assert results[0] == results[1]


def test_copied_branches_keep_evidence_counters_and_writes_separate():
    BaseTrack._count = 0
    original = tracker(install_isolated_risk_memory, 1.)
    for frame in range(1, 4):
        original.update(*observations(frame))
    branch = isolated_state(original)
    branch.risk_memory_constant = 0.
    branch.update(*observations(4))
    original.update(*observations(4))
    assert branch.phase_vi_memory_counts['accepted'] == 2
    assert original.phase_vi_memory_counts['accepted'] == 0
    for a, b in zip(original.tracked_stracks, branch.tracked_stracks):
        assert a is not b and a.risk_memory_evidence is not b.risk_memory_evidence
        assert len(a.identity_history) == 1 and len(b.identity_history) == 2
        assert a.risk_memory_counts['accepted'] == 0 and b.risk_memory_counts['accepted'] == 1
