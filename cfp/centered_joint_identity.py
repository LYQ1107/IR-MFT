"""Same-input full-set controls and centered bounded identity corrections."""
import math
import torch
from torch import nn
from .joint_identity_decoder import JointFishIDDecoder


class TokenFeedForward(nn.Module):
    """Per-token FFN with the same parameter count as replaced set attention."""
    def __init__(self,hidden):
        super().__init__()
        self.input=nn.Linear(hidden,2*hidden)
        self.output=nn.Linear(2*hidden,hidden)
        self.gate=nn.Parameter(torch.ones(hidden))
    def forward(self,query,key,value,key_padding_mask=None,need_weights=False,**unused):
        output=self.output(torch.nn.functional.gelu(self.input(query)))*self.gate
        if key_padding_mask is not None:
            output=output.masked_fill(key_padding_mask[:,:,None],0.)
        return output,None


class CenteredJointIdentity(JointFishIDDecoder):
    def __init__(self, dimension=512, hidden=64, heads=4, capacity=128, architecture='joint'):
        if architecture not in ('joint', 'pair', 'cross_transformer'):
            raise ValueError('Registered same-input pair/cross-transformer/full-set controls required')
        super().__init__(dimension, hidden, heads, capacity,
            architecture='pair' if architecture=='pair' else 'joint')
        self.registered_architecture=architecture
        self.set_self_attention=architecture=='joint'
        if architecture=='cross_transformer':
            # Parameter budget stays exact, with genuinely gradient-active
            # per-token FFNs instead of allocated-but-unused self attention.
            self.track_attention=TokenFeedForward(hidden)
            self.detection_attention=TokenFeedForward(hidden)


def centered_joint_cost(baseline, residual, legal, maximum_delta=.03):
    """Zero-mean per-candidate legal-edge edits; cannot emulate uniform gating.

    Both before/after tanh centering are needed: raw common score offsets must
    not alter the nonlinear edit, and final bounded edits have zero legal mean.
    A single legal old track gets no identity-ranking edit. NULL is a separate
    N+1 classification/assignment route, never an impossible bounded objective.
    """
    if (baseline.ndim!=2 or baseline.shape!=residual.shape or legal.shape!=baseline.shape
            or legal.dtype!=torch.bool or not math.isfinite(maximum_delta) or not 0<=maximum_delta<=.1
            or not torch.isfinite(baseline).all() or not torch.isfinite(residual).all()):
        raise ValueError('Finite aligned costs/identity residual/boolean legal edges required')
    edits=torch.zeros_like(residual)
    for j in range(baseline.shape[1]):
        indices=legal[:,j].nonzero(as_tuple=True)[0]
        if len(indices)<2:
            continue
        # Subtract the first legal value before averaging; uniform values then
        # vanish exactly, rather than leaving a float reduction roundoff bias.
        relative=residual[indices,j]-residual[indices[0],j]
        centered=relative-relative.mean()
        bounded=centered.tanh()
        delta=.5*maximum_delta*(bounded-bounded.mean())
        edits[indices,j]=delta
    return baseline+edits
