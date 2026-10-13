import numpy as np
import pytest
import torch

from cfp.candidate_cue_supervision import known_cue_targets, cue_utility_loss


def test_actual_masked_times_map_labels_and_unknown_stays_unsupervised():
    rows = [dict(identity=5, sequence='v', frame=9, times=np.array([1, 2, 3]), indices=np.array([0, 1, 2]))]
    banks = {'v': dict(labels=np.array([5, -1, 6]))}
    labels = known_cue_targets(rows, banks, torch.tensor([[2, 3, 0]]), torch.tensor([[True, True, False]]))
    assert labels.tolist() == [[-1, 0, -1]]
    logits = torch.zeros(1, 2, 3, requires_grad=True)
    loss, count = cue_utility_loss(logits, labels, torch.tensor([[True, False]]))
    loss.backward()
    assert count == dict(known_correct=0, known_wrong=1)
    assert logits.grad[0, 0, 0] == 0 and logits.grad[0, 0, 1] > 0
    assert not logits.grad[0, 1].any()
    with pytest.raises(ValueError):
        known_cue_targets(rows, banks, torch.tensor([[9]]), torch.tensor([[True]]))
