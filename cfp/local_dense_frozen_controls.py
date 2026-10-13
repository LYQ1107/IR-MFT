"""No learned parameters; exact same scene histories/current candidates."""
import torch
from torch import nn
from torch.nn import functional as F


class FrozenSceneControl(nn.Module):
    def __init__(self, mode='cls_fifo'):
        super().__init__()
        if mode not in ('cls_last', 'cls_fifo', 'local_corresponding', 'local_flexible'):
            raise ValueError('Registered frozen same-scene control required')
        self.mode = mode

    def forward(self, history, history_fraction, history_global, times, history_valid,
                frame, candidates, candidate_fraction, candidate_global):
        if (times[history_valid]>=frame).any() or (times[history_valid]<1).any():
            raise ValueError('Frozen control uses strictly earlier observations too')
        legal = candidate_fraction.sum(1)>0
        past_valid = history_valid & (history_fraction.sum(-1)>0)
        available = past_valid.any(1)
        n, m = len(history), len(candidates)
        if self.mode.startswith('cls_'):
            current = F.normalize(candidate_global, dim=-1)*legal[:, None]
            observed = F.normalize(history_global, dim=-1)*past_valid[..., None]
            predicted = observed.sum(1)
            if self.mode=='cls_last':
                predicted = predicted.new_zeros(predicted.shape)
                for j in range(n):
                    indices = torch.nonzero(past_valid[j], as_tuple=False).flatten()
                    if len(indices):
                        predicted[j] = observed[j, indices[-1]]
            predicted = F.normalize(predicted, dim=-1)
            scores = predicted @ current.T
            current_output = dict(tokens=current[:, None], valid=legal[:, None], weights=legal[:, None].float())
            predicted = predicted[:, None]
        else:
            support = history_fraction*history_valid[..., None]
            observed = F.normalize(history, dim=-1)*support[..., None]
            query = F.normalize(observed.sum(1), dim=-1)
            query_valid = support.sum(1)>0
            current = F.normalize(candidates, dim=-1)*(candidate_fraction>0)[..., None]
            similarity = torch.einsum('nkd,mjd->nmkj', query, current)
            if not n or not m:
                scores = history.new_zeros(n, m)
            elif self.mode=='local_corresponding':
                values = similarity.diagonal(dim1=-2, dim2=-1)
                weights = (support.sum(1)[:, None]*candidate_fraction[None]).sqrt()
                scores = (values*weights).sum(-1)/weights.sum(-1).clamp_min(1.e-6)
            else:
                current_safe = candidate_fraction>0
                current_safe[~current_safe.any(1), 0] = True
                query_safe = query_valid.clone()
                query_safe[~query_safe.any(1), 0] = True
                row = (similarity/.1).masked_fill(~current_safe[None, :, None], -torch.inf).softmax(-1)
                column = (similarity/.1).masked_fill(~query_safe[:, None, :, None], -torch.inf).softmax(-2)
                a = ((row*similarity).sum(-1)*query_valid[:, None]).sum(-1)/query_valid.sum(1).clamp_min(1)[:, None]
                weights = candidate_fraction/candidate_fraction.sum(1).clamp_min(1.e-6)[:, None]
                b = ((column*similarity).sum(-2)*weights[None]).sum(-1)
                scores = (a+b)/2
            predicted = query*available[:, None, None]
            current_output = dict(tokens=current, valid=candidate_fraction>0,
                weights=candidate_fraction/candidate_fraction.sum(1).clamp_min(1.e-6)[:, None])
        scores = (scores*available[:, None]).masked_fill(~legal[None], -torch.inf)
        return dict(scores=scores, current=current_output, predicted=predicted, history_available=available)
