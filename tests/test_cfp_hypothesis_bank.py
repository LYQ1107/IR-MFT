from types import SimpleNamespace

import numpy as np
import pytest

from cfp.hypothesis_bank import HypothesisBank, IdentityHypothesis, isolated_state
from cfp.ambiguity_association import alternative_pair, observed_assignment
from yolox.tracker import matching


def test_branches_do_not_share_mutable_history():
    original = SimpleNamespace(history=[np.array([1., 2.])])
    clone = isolated_state(original)
    clone.history[0][0] = 99
    assert original.history[0][0] == 1
    bank = HypothesisBank(original, 2)
    states = [IdentityHypothesis(isolated_state(original), float(i), 1, i) for i in range(4)]
    assert bank.advance(1, states).branch_id == 0
    assert len(bank.hypotheses) == 2
    with pytest.raises(ValueError):
        bank.advance(1, states)


def test_alternate_is_bijective_legal_and_bounded():
    cost = np.array([[.1, .12], [.12, .1]])
    result = alternative_pair(cost, [[0, 0], [1, 1]], .97, .05)
    assert result is not None and result[1].tolist() == [[0, 1], [1, 0]]
    assert alternative_pair(cost, [[0, 0], [1, 1]], .11, .05) is None
    assert alternative_pair(cost, [[0, 0]], .97, .05) is None


def test_assignment_transaction_restores_even_on_error():
    original = matching.linear_assignment
    with pytest.raises(RuntimeError):
        with observed_assignment() as evidence:
            matching.linear_assignment(np.array([[.1, .12], [.12, .1]]), .97)
            assert evidence['alternate'] is not None
            raise RuntimeError('deliberate owned transaction failure')
    assert matching.linear_assignment is original


def test_k1_exact_disabled_path_and_k2_live_branch_isolation():
    import torch
    from pathlib import Path
    from cfp.lifecycle_crossover import load_bot_reference, make_cfp_lifecycle
    from cfp.ambiguity_association import AmbiguityAssociation
    from cfp.online_recovery import install_risk_association
    from yolox.tracker.basetrack import BaseTrack
    reference = Path('/data3/liuyeqiang/CFP-FishMOT_references/BoT-SORT')
    if not reference.exists():
        pytest.skip('Pinned native reference unavailable')
    class Last(torch.nn.Module):
        def forward(self, x, times, valid, frame):
            return x[torch.arange(len(x)), (valid.sum(1) - 1).clamp_min(0)], valid.any(1)
    native = load_bot_reference(reference)
    frames = [np.array([[0, 0, 30, 20, .9], [2, 0, 32, 20, .9]], np.float32) for _ in range(4)]
    features = np.repeat(np.array([[1., 0]], np.float32), 2, axis=0)
    expected = []
    BaseTrack._count = 0
    original = make_cfp_lifecycle(native, 'gru', Last())
    for frame in frames:
        expected.append([(t.track_id, t.current_proposal_index, t.tlwh.copy()) for t in original.update(frame, features, np.arange(2))])
    BaseTrack._count = 0
    wrapped = AmbiguityAssociation(install_risk_association(make_cfp_lifecycle(native, 'gru', Last())), 1)
    for i, frame in enumerate(frames):
        actual = [(t.track_id, t.current_proposal_index, t.tlwh.copy()) for t in wrapped.update(frame, features, np.arange(2))]
        assert len(actual) == len(expected[i])
        for a, b in zip(actual, expected[i]):
            assert a[:2] == b[:2] and np.array_equal(a[2], b[2])
    BaseTrack._count = 0
    beam = AmbiguityAssociation(make_cfp_lifecycle(native, 'gru', Last()), 2, ambiguity_delta=1.)
    for frame in frames:
        output = beam.update(frame, features, np.arange(2))
        assert len({t.track_id for t in output}) == len(output)
        assert len(beam.bank.hypotheses) <= 2
    assert beam.statistics['branch_expansions'] > 0
    left, right = beam.bank.hypotheses
    assert left.state.tracked_stracks[0] is not right.state.tracked_stracks[0]
    assert left.state.tracked_stracks[0].identity_history is not right.state.tracked_stracks[0].identity_history

