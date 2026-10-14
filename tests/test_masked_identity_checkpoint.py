import copy
import numpy as np
import pytest
import torch
from torch import nn
from cfp.masked_identity_checkpoint import frozen_base, delta_state, load_delta_state, sample_patch_queries


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.student = nn.Module()
        self.student.backbone = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        self.student.backbone[0].requires_grad_(False)
        self.teacher = copy.deepcopy(self.student.backbone).requires_grad_(False)
        self.predictor = nn.Linear(2, 2)
        self.register_buffer('teacher_updates', torch.zeros((), dtype=torch.int64))


def test_delta_keeps_teacher_tail_and_buffers_and_rejects_base_drift():
    model = Tiny()
    base = frozen_base(model)
    fresh = copy.deepcopy(model)
    with torch.no_grad():
        model.teacher[1].weight.add_(3)
        model.predictor.bias.add_(4)
        model.teacher_updates.add_(9)
    delta = delta_state(model, base)
    load_delta_state(fresh, delta, base)
    assert all(torch.equal(a, fresh.state_dict()[k]) for k, a in model.state_dict().items())
    with torch.no_grad():
        fresh.teacher[0].weight.add_(1)
    with pytest.raises(ValueError, match='Immutable'):
        delta_state(fresh, base)
    with pytest.raises(ValueError, match='disjoint'):
        load_delta_state(model, {**delta, next(iter(base)): next(iter(base.values()))}, base)


def test_mask_queries_exclude_padding_preserve_context_and_resume_rng():
    fraction = torch.zeros(3, 162)
    fraction[0, :10] = 1
    fraction[1, 71] = .01
    rng = np.random.default_rng(2)
    state = copy.deepcopy(rng.bit_generator.state)
    queries = sample_patch_queries(fraction, rng, .5)
    assert queries.sum(1).tolist() == [5, 1, 0]
    assert not (queries & (fraction == 0)).any()
    rng.bit_generator.state = state
    assert torch.equal(queries, sample_patch_queries(fraction, rng, .5))
