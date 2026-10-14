"""Lossless frozen-base delta checkpoints and observable patch queries."""
import numpy as np
import torch


def frozen_base(model):
    """Only immutable pretrained prefix is omitted; heads always saved."""
    names = {n for n, p in model.student.backbone.named_parameters() if not p.requires_grad}
    state = model.state_dict()
    keys = {prefix+n for prefix in ('student.backbone.', 'teacher.') for n in names}
    return {k: state[k].detach().cpu().clone() for k in sorted(keys)}


def delta_state(model, base):
    state = model.state_dict()
    for key, original in base.items():
        if not torch.equal(state[key].detach().cpu(), original):
            raise ValueError('Immutable pretrained base changed: '+key)
    return {k: v.detach().cpu().clone() for k, v in state.items() if k not in base}


def load_delta_state(model, delta, base):
    expected = model.state_dict()
    if base.keys() & delta.keys() or base.keys() | delta.keys() != expected.keys():
        raise ValueError('Exact disjoint base and all model delta keys required')
    for key, value in base.items():
        if not torch.equal(expected[key].detach().cpu(), value):
            raise ValueError('Wrong official initialization: '+key)
    model.load_state_dict(dict(base, **delta), strict=True)


def sample_patch_queries(fraction, rng, ratio):
    if not 0 < ratio < 1 or fraction.ndim != 2 or fraction.shape[1] != 162:
        raise ValueError('Observable 162patches and declared nontrivial mask fraction required')
    available = fraction.detach().cpu().numpy() > 0
    selected = np.zeros_like(available)
    for row, valid in enumerate(available):
        indices = np.flatnonzero(valid)
        if len(indices):
            count = min(max(1, int(round(len(indices)*ratio))), max(1, len(indices)-1))
            selected[row, rng.choice(indices, count, replace=False)] = True
    return torch.from_numpy(selected).to(fraction.device)
