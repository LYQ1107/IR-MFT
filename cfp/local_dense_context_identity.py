"""JEPA prediction regularizes representation; identity reads past context.

The v1 inference decoder bottleneck was diagnosed on innerBT001 only. This
separate class preserves v1 producers and leaves their live sources unchanged.
"""
import torch
from torch.nn import functional as F

from .local_dense_jepa import LocalDenseStudent


class ContextIdentityStudent(LocalDenseStudent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mode in ('no_prediction', 'local_contrastive'):
            # No latent decoder is used by these representation controls.
            # no_prediction is an explicit alias of local_temporal, rather
            # than a fictitious second independent baseline.
            for name, parameter in self.named_parameters():
                if name.startswith(('queries', 'horizon_encoder.', 'predictor.',
                                    'prediction_norm.', 'prediction_ffn.', 'output_norm.')):
                    parameter.requires_grad_(False)

    def forward(self, history, history_fraction, history_global, times, history_valid,
                frame, candidates, candidate_fraction, candidate_global):
        context = self.history_context(history, history_fraction, history_global, times, history_valid, frame)
        representation = F.normalize(context['pooled'], dim=-1)*context['available'][:, None, None]
        # A target decoder is used exclusively by the dense training objective.
        predicted = self.predict(context) if self.mode in ('full', 'no_ema', 'no_id') else representation
        current = self.current(candidates, candidate_fraction, candidate_global)
        age = torch.where(context['available'], context['age'][:, -1], torch.zeros_like(context['age'][:, -1]))
        scores = self.relation_scores(representation, current, context['available'], age)
        return dict(scores=scores, predicted=predicted, representation=representation,
                    context=context, current=current, history_available=context['available'])

    def online(self, history, history_fraction, history_global, times, history_valid,
               frame, candidates, candidate_fraction, candidate_global):
        """Actual deployment excludes both teacher and prediction decoder."""
        context = self.history_context(history, history_fraction, history_global, times, history_valid, frame)
        representation = F.normalize(context['pooled'], dim=-1)*context['available'][:, None, None]
        current = self.current(candidates, candidate_fraction, candidate_global)
        age = torch.where(context['available'], context['age'][:, -1], torch.zeros_like(context['age'][:, -1]))
        scores = self.relation_scores(representation, current, context['available'], age)
        return dict(scores=scores, predicted=representation, representation=representation,
                    context=context, current=current, history_available=context['available'])
