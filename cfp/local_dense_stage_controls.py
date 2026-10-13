"""Read-only coordinate/context/prediction decomposition of a fixed model."""
import torch
from torch import nn
from torch.nn import functional as F


class FixedLocalStage(nn.Module):
    def __init__(self, student, stage):
        super().__init__()
        if stage not in ('encoder_fifo', 'context_pool', 'prediction'):
            raise ValueError('Declared fixed-checkpoint evidence stage required')
        self.student, self.stage = student, stage

    def forward(self, history, history_fraction, history_global, times, history_valid,
                frame, candidates, candidate_fraction, candidate_global):
        student = self.student
        context = student.history_context(history, history_fraction, history_global, times, history_valid, frame)
        current = student.current(candidates, candidate_fraction, candidate_global)
        if self.stage=='prediction':
            predicted = student.predict(context)
        elif self.stage=='context_pool':
            predicted = F.normalize(context['pooled'], dim=-1)*context['available'][:, None, None]
        else:
            n, length, count, dimension = history.shape
            fraction = history_fraction*history_valid[..., None]
            encoded = student.current(history.reshape(-1, count, dimension), fraction.reshape(-1, count),
                                      history_global.reshape(-1, dimension))
            tokens = F.normalize(encoded['tokens'], dim=-1).reshape(n, length, count, dimension)
            valid = encoded['valid'].reshape(n, length, count)
            predicted = F.normalize((tokens*valid[..., None]).sum(1), dim=-1)*context['available'][:, None, None]
        age = torch.where(context['available'], context['age'][:, -1], torch.zeros_like(context['age'][:, -1]))
        scores = student.relation_scores(predicted, current, context['available'], age)
        return dict(scores=scores, predicted=predicted, current=current, history_available=context['available'])
