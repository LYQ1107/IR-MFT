"""Current-image-only encoder for the audited visible-token pretraining.

The pretraining checkpoint contains only a DINOv2 backbone state.  This
adapter deliberately exposes the same 384-dimension/native K4 interface as
the frozen DINO producer so that a visible-token checkpoint can be replayed
through the unchanged V3 proposal, lifecycle and TrackEval protocol.  It
does not load annotations, identities, future frames, or an identity probe.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .dinov2_identity import load_pinned_small, resized_observations, spatial_observations
from .dinov2_native_encoder import current_proposal_crops
from .provenance import sha256


class CurrentVisibleTokenIdentityEncoder:
    """Run a completed fit-only visible-token backbone on current proposals."""

    def __init__(self, reference, weights, source_proof, training, deployment_video,
                 mode, seed, expected_fit, expected_inner, device='cuda', batch_size=128):
        if device not in ('cpu', 'cuda') or batch_size != 128:
            raise ValueError('Declared current-only visible-token device/padding required')
        training = Path(training)
        manifest = json.loads((training / 'manifest.json').read_text())
        guard = manifest.get('guard', {})
        training_config = manifest.get('config', {})
        declared_fit = training_config.get('fit_videos', [])
        declared_excluded = training_config.get('excluded_videos', [])
        if (manifest.get('status') != 'COMPLETE_UNLABELED_VISIBLE_TOKEN_PRETRAINING'
                or guard.get('mode') != mode or guard.get('seed') != seed
                or declared_fit != list(expected_fit)
                or set(declared_excluded) != {deployment_video, expected_inner}
                or deployment_video not in declared_excluded
                or deployment_video in declared_fit
                or expected_inner in declared_fit
                or manifest.get('physical_ID_values_used') is not False
                or manifest.get('GT_crop_values_used') is not False
                or sha256(training / 'encoder_final.pt') != manifest.get('encoder_checkpoint_sha256')):
            raise ValueError('Visible-token checkpoint does not match declared excluded-video fold')
        proofs = json.loads(Path(source_proof).read_text())['imported_source_proofs']
        self.backbone = load_pinned_small(reference, weights, proofs)
        saved = torch.load(training / 'encoder_final.pt', map_location='cpu', weights_only=False)
        if saved.get('guard') != guard or not isinstance(saved.get('backbone'), dict):
            raise ValueError('Visible-token checkpoint guard/state mismatch')
        self.backbone.load_state_dict(saved['backbone'], strict=True)
        self.backbone.eval().requires_grad_(False).to(device)
        self.device, self.batch_size = device, batch_size
        self.metadata = dict(
            backbone_pretrained_sha256=sha256(weights),
            visible_token_checkpoint_sha256=manifest['encoder_checkpoint_sha256'],
            visible_token_manifest_sha256=sha256(training / 'manifest.json'),
            visible_token_mode=mode,
            visible_token_seed=seed,
            dimension=384,
            crop_hw=[128, 256], model_hw=[126, 252], crop_margin=.05,
            pixel_quantization='NormalizedcropFP32->FP16->FP32; current proposal path',
            encoder_precision='FP32; TF32disabled', encoder_batch_padding=128,
            coordinate_kind='fit_only_visible_token_backbone_CLS384',
            local_kind='current real content-weighted geometricK4; no GT/teacher/future',
            trained_encoder=dict(path=str(training), manifest_sha256=sha256(training / 'manifest.json'),
                                 checkpoint_sha256=manifest['encoder_checkpoint_sha256'],
                                 fit=list(expected_fit), inner=expected_inner,
                                 outer=deployment_video, seed=seed, mode=mode),
            fitting_scope='Unlabeled four-fit-video visible-token pretraining; deployment outer/inner excluded',
        )
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def encode(self, rgb, boxes):
        started = time.perf_counter()
        crops, masks, valid = current_proposal_crops(rgb, boxes)
        timing = dict(current_crop_and_mask_seconds=time.perf_counter() - started,
                      resize_transfer_seconds=0., encoder_seconds=0., feature_transfer_seconds=0.)
        output = dict(embedding=np.zeros((len(boxes), 384), np.float32),
                      raw_cls=np.zeros((len(boxes), 384), np.float32),
                      local4=np.zeros((len(boxes), 4, 384), np.float32),
                      visibility4=np.zeros((len(boxes), 4), np.float32), valid=valid)
        with torch.inference_mode():
            for offset in range(0, len(crops), self.batch_size):
                count = min(len(crops) - offset, self.batch_size)
                begin = time.perf_counter()
                padded = torch.zeros(self.batch_size, 3, 128, 256)
                supported = torch.zeros(self.batch_size, 1, 128, 256)
                padded[:count] = crops[offset:offset + count]
                supported[:count] = masks[offset:offset + count]
                images, fractions = resized_observations(padded.to(self.device), supported.to(self.device))
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                timing['resize_transfer_seconds'] += time.perf_counter() - begin
                begin = time.perf_counter()
                observed = self.backbone.forward_features(images)
                cls, tokens = observed['x_norm_clstoken'], observed['x_norm_patchtokens']
                raw_cls = F.normalize(cls.float(), dim=-1)
                spatial = spatial_observations(tokens, fractions)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                timing['encoder_seconds'] += time.perf_counter() - begin
                begin = time.perf_counter()
                values = dict(embedding=raw_cls, raw_cls=raw_cls,
                              local4=spatial['local4'], visibility4=spatial['visibility4'])
                for name, value in values.items():
                    current = value[:count].float().cpu().numpy()
                    current[~valid[offset:offset + count]] = 0
                    if not np.isfinite(current).all():
                        raise ValueError('Nonfinite visible-token current-only feature output')
                    output[name][offset:offset + count] = current
                timing['feature_transfer_seconds'] += time.perf_counter() - begin
        output['stage_seconds'] = timing
        return output
