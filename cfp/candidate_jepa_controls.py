"""Explicit NULL and recomputed-history-evidence ablations for Candidate-JEPA."""
import torch
from torch import nn
from torch.nn import functional as F

from .candidate_jepa import CandidateJEPAStudent
from .predictive import checked_history


class CandidateJEPAControl(CandidateJEPAStudent):
    def __init__(self, latent=64, architecture='cross_attention', candidate_context=True,
                 recoverability_evidence=False, use_null=True):
        super().__init__(latent=latent, architecture=architecture, candidate_context=candidate_context)
        self.use_null = use_null
        self.evidence_head = nn.Sequential(nn.Linear(19, 32), nn.GELU(), nn.Linear(32, 1)) if recoverability_evidence else None

    def forward(self, history, times, valid, frame, candidates, candidate_valid, evidence):
        output = super().forward(history, times, valid, frame, candidates, candidate_valid, evidence)
        b, m, _ = candidates.shape
        if self.evidence_head is not None and b and m:
            # Recompute from ONLY the actual visible/masked history. Cached full
            # history scalar channels and original prototype costs are ignored.
            source, _, length, available, last = checked_history(history, times, valid, frame, 'log1p_15000')
            current = F.normalize(candidates.float(), dim=-1)
            similarity = torch.einsum('btd,bmd->bmt', source, current)
            mean = F.normalize(source.sum(1), dim=-1)
            mean_similarity = torch.einsum('bd,bmd->bm', mean, current)
            last_similarity = torch.einsum('bd,bmd->bm', last, current)
            maximum = similarity.masked_fill(~valid[:, None], -1.).max(-1).values
            minimum = similarity.masked_fill(~valid[:, None], 1.).min(-1).values
            agreement = ((source * mean[:, None]).sum(-1) * valid).sum(1) / length.clamp_min(1)
            conflict = 1 - (mean * last).sum(-1)
            safe = torch.where(candidate_valid.any(1)[:, None], mean_similarity.masked_fill(~candidate_valid, -torch.inf), torch.zeros_like(mean_similarity))
            probability = safe.softmax(-1) * candidate_valid
            entropy = -(probability * probability.clamp_min(1e-8).log()).sum(-1)
            best = mean_similarity.masked_fill(~candidate_valid, -1.).max(-1).values
            history_evidence = torch.stack([mean_similarity, last_similarity, maximum, minimum,
                agreement[:, None].expand(b, m), conflict[:, None].expand(b, m),
                mean_similarity - best[:, None], entropy[:, None].expand(b, m),
                available[:, None].expand(b, m).float()], dim=-1)
            history_evidence = history_evidence * available[:, None, None]
            features = torch.cat([history_evidence, evidence[..., 9:19]], dim=-1)
            correction = self.evidence_head(features).squeeze(-1)
            output['scores'] = torch.cat([output['scores'][:, :m] + correction, output['scores'][:, -1:]], dim=1)
        if not self.use_null:
            output['scores'] = torch.cat([output['scores'][:, :m], output['scores'].new_full((b, 1), -1e4)], dim=1)
        return output
