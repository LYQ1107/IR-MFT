"""Light track-level student context and genuine latent prediction components.

No teacher instance, annotations, clean-target inputs or future pixels exist in
the inference module. Trainers own a separate EMA copy of the token encoder.
Decoded association prototypes remain in the frozen512dim FishReID space.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .predictive import checked_history


class FishJEPATokenEncoder(nn.Module):
    def __init__(self, dimension=512, latent=128):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimension, latent), nn.LayerNorm(latent), nn.SiLU(),
                                    nn.Linear(latent, latent), nn.LayerNorm(latent))

    def forward(self, features):
        return F.normalize(self.layers(F.normalize(features.float(), dim=-1)), dim=-1)


class FishJEPAStudent(nn.Module):
    def __init__(self, architecture='gru', latent=128, reliability=None, freeze_encoder=False):
        super().__init__()
        if architecture not in ('gru', 'transformer'):
            raise ValueError('Declared lightweight architecture required')
        self.architecture, self.latent = architecture, latent
        self.token_encoder = FishJEPATokenEncoder(512, latent)
        self.time_encoder = nn.Linear(2, latent)
        self.mask_token = nn.Parameter(torch.zeros(latent))
        if architecture == 'gru':
            self.context_encoder = nn.GRU(latent, latent, batch_first=True)
        else:
            layer = nn.TransformerEncoderLayer(latent, 4, dim_feedforward=144, dropout=0., batch_first=True)
            self.context_encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.latent_predictor = nn.Linear(latent, latent)
        self.masked_predictor = nn.Linear(latent, latent)
        self.raw_decoder = nn.Linear(latent, 512)
        self.reliability = reliability
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self.token_encoder.requires_grad_(False)
        nn.init.zeros_(self.latent_predictor.weight)
        nn.init.zeros_(self.latent_predictor.bias)
        nn.init.zeros_(self.raw_decoder.bias)

    def states(self, features, frames, valid, query_frame, context_mask=None):
        source, time, lengths, available, _ = checked_history(features, frames, valid, query_frame)
        if source.shape[-1] != 512:
            raise ValueError('Explicit512dim frozen FishReID coordinates required')
        context_mask = torch.zeros_like(valid) if context_mask is None else context_mask.bool()
        if context_mask.shape != valid.shape or torch.any(context_mask & ~valid):
            raise ValueError('Mask only actually present past observations')
        visible = valid & ~context_mask
        if torch.any(available & ~visible.any(1)):
            raise ValueError('At least one visible past cue required')
        # Masked token values cannot reach encoder, reliability, raw residual,
        # or any statistic. Teacher targets are separate training tensors.
        masked_source = source * visible[..., None]
        tokens = self.token_encoder(masked_source)
        tokens = torch.where(context_mask[..., None], self.mask_token[None, None], tokens)
        if self.reliability is not None:
            # Infer soft trust on visible, compacted strictly past cues. A
            # frozen/learned state is independent of any clean target encoder.
            compact = torch.zeros_like(source)
            compact_times = torch.zeros_like(frames)
            compact_valid = torch.zeros_like(valid)
            for row in range(len(source)):
                count = int(visible[row].sum())
                compact[row, :count] = source[row, visible[row]]
                compact_times[row, :count] = frames[row, visible[row]]
                compact_valid[row, :count] = True
            restored, _ = self.reliability(compact, compact_times, compact_valid, query_frame)
            restored_token = self.token_encoder(restored)
            tokens = .5 * tokens + .5 * restored_token[:, None]
        tokens = tokens + self.time_encoder(time)
        if self.architecture == 'gru':
            packed = nn.utils.rnn.pack_padded_sequence(tokens, lengths.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False)
            encoded, _ = self.context_encoder(packed)
            encoded, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=source.shape[1])
        else:
            safe_valid = valid.clone()
            safe_valid[~available, 0] = True
            future = torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool).triu(1)
            encoded = self.context_encoder(tokens, mask=future, src_key_padding_mask=~safe_valid)
        state = encoded[torch.arange(len(source), device=source.device), (lengths - 1).clamp_min(0)]
        steps = torch.arange(source.shape[1], device=source.device)[None].expand(len(source), -1)
        last_index = steps.masked_fill(~visible, -1).max(1).values.clamp_min(0)
        last_raw = source[torch.arange(len(source), device=source.device), last_index] * available[:, None]
        last_latent = self.token_encoder(last_raw)
        residual = self.latent_predictor(state)
        predicted_latent = F.normalize(last_latent + residual, dim=-1) * available[:, None]
        prototype = F.normalize(last_raw + self.raw_decoder(residual), dim=-1) * available[:, None]
        if not torch.isfinite(prototype).all() or not torch.isfinite(predicted_latent).all():
            raise ValueError('Nonfinite Fish-JEPA student')
        return prototype, available, predicted_latent, encoded

    def training_details(self, features, frames, valid, query_frame, context_mask=None):
        prototype, available, prediction, encoded = self.states(features, frames, valid, query_frame, context_mask)
        masked_prediction = F.normalize(self.masked_predictor(encoded), dim=-1) * valid[..., None]
        return prototype, available, prediction, masked_prediction

    def forward(self, features, frames, valid, query_frame):
        # Runtime computes only current identity forecast; the teacher and
        # masked reconstruction head are never evaluated by inference.
        return self.states(features, frames, valid, query_frame)[:2]


@torch.no_grad()
def update_ema_target(target_encoder, student_encoder, momentum):
    """Training-only exact EMA, with no optimizer/gradients for target state."""
    if not 0 <= momentum <= 1:
        raise ValueError('Bounded teacher momentum required')
    source = dict(student_encoder.named_parameters())
    destination = dict(target_encoder.named_parameters())
    if source.keys() != destination.keys():
        raise ValueError('Teacher/student encoders must match')
    for name in source:
        destination[name].mul_(momentum).add_(source[name].detach(), alpha=1 - momentum)
    for name, value in student_encoder.named_buffers():
        dict(target_encoder.named_buffers())[name].copy_(value.detach())
