import dataclasses

import pytest
import torch

from cfp.candidate_graph import CandidateGraph
from cfp.recoverability import observable_candidate_features, RecoverabilityEstimator, FEATURE_NAMES


def graph(n=2, m=3, t=3):
    return CandidateGraph(history=torch.randn(n, t, 8),
        history_frames=torch.arange(1, t + 1)[None].repeat(n, 1),
        history_valid=torch.ones(n, t, dtype=torch.bool), candidates=torch.randn(m, 8),
        track_boxes=torch.tensor([[0., 0, 10, 10]]).repeat(n, 1),
        candidate_boxes=torch.tensor([[1., 1, 11, 11]]).repeat(m, 1), confidence=torch.ones(m) * .9,
        baseline_cost=torch.ones(n, m) * .3, legal=torch.ones(n, m, dtype=torch.bool),
        age=torch.ones(n), gap=torch.ones(n), frame=5)


@pytest.mark.parametrize('n,m,t', [(2, 3, 3), (0, 3, 3), (2, 0, 3), (2, 1, 0), (2, 3, 0)])
def test_empty_single_and_normal_graph(n, m, t):
    g = graph(n, m, t)
    features = observable_candidate_features(g)
    assert features.shape == (n, m, len(FEATURE_NAMES)) and torch.isfinite(features).all()
    output = RecoverabilityEstimator().decide(g)
    assert output.scores.shape == (n, m + 1) and torch.isfinite(output.scores[:, -1]).all()


def test_reject_nonfinite_and_current_or_future_history():
    g = graph()
    with pytest.raises(ValueError, match='strictly past'):
        dataclasses.replace(g, frame=3).validate()
    g.candidates[0, 0] = torch.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        g.validate()


def test_candidate_permutation_equivariance_and_masked_null():
    g = graph()
    g.legal[0, :] = False
    model = RecoverabilityEstimator().eval()
    out = model.decide(g)
    assert not torch.isfinite(out.scores[0, :-1]).any()
    perm = torch.tensor([2, 0, 1])
    gp = dataclasses.replace(g, candidates=g.candidates[perm], candidate_boxes=g.candidate_boxes[perm],
        confidence=g.confidence[perm], baseline_cost=g.baseline_cost[:, perm], legal=g.legal[:, perm])
    output = model.decide(gp)
    torch.testing.assert_close(output.memory_write_risk, out.memory_write_risk[:, perm])

