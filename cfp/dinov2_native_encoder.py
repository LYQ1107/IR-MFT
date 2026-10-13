"""Current RGB/proposal-only DINO identity features; no annotations/teacher.

Model fitting provenance and feature coordinates are checked before loading.
Each call may read only the supplied current image and actual proposal boxes.
"""
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .dinov2_fish_encoder import FineGrainedDinoFishEncoder
from .dinov2_identity import load_pinned_small, resized_observations, spatial_observations
from .identity_data import crop_to_tensor
from .local_identity_encoder import letterbox_visibility
from .provenance import sha256


def current_proposal_crops(rgb, boxes):
    if (rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3
            or boxes.shape!=(len(boxes), 4) or not np.isfinite(boxes).all()):
        raise ValueError('One current BGRuint8 image and finite actual XYXYproposal rows required')
    images, masks, valid = [], [], []
    for x1, y1, x2, y2 in boxes:
        xywh = [float(x1), float(y1), float(x2-x1), float(y2-y1)]
        try:
            crop = crop_to_tensor(rgb, xywh)
            mask = letterbox_visibility(rgb.shape[:2], xywh)
        except ValueError:
            crop, mask = torch.zeros(3, 128, 256), torch.zeros(1, 128, 256)
            available = False
        else:
            available = bool(mask.sum()>0)
        # Exact same declared pixel quantization as the registered fitting bank.
        images.append(crop.half().float())
        masks.append(mask)
        valid.append(available)
    return (torch.stack(images) if images else torch.zeros(0, 3, 128, 256),
            torch.stack(masks) if masks else torch.zeros(0, 1, 128, 256), np.array(valid, dtype=np.bool_))


class CurrentDinoIdentityEncoder:
    def __init__(self, reference, weights, source_proof, training=None, deployment_video=None,
                 device='cuda', batch_size=128):
        if device not in ('cpu', 'cuda') or batch_size!=128:
            raise ValueError('Declared device and current-frame encoder padding128 required')
        self.device, self.batch_size = device, batch_size
        proofs = json.loads(Path(source_proof).read_text())['imported_source_proofs']
        pretrained = load_pinned_small(reference, weights, proofs)
        self.model, self.backbone = None, pretrained
        self.metadata = dict(backbone_pretrained_sha256=sha256(weights), source_reference=pretrained.__class__.__module__,
            source_commit='7764ea0f912e53c92e82eb78a2a1631e92725fc8',
            imported_source_proofs=proofs, trained_encoder=None, dimension=384,
            crop_hw=[128, 256], model_hw=[126, 252], crop_margin=.05,
            pixel_quantization='NormalizedcropFP32->FP16->FP32, exact registered pilot input bytes',
            encoder_precision='FP32; TF32disabled', encoder_batch_padding=128,
            coordinate_kind='frozen_DINO_CLS384', local_kind='current real content-weighted geometricK4; no anatomy/GTocclusion',
            fitting_scope='No fish fitting for frozen source; privatepretrain/video/biological overlap unverified')
        if training is not None:
            training = Path(training)
            manifest = json.loads((training/'manifest.json').read_text())
            guard = manifest['guard']
            if (manifest['status']!='COMPLETE' or deployment_video is None
                    or deployment_video not in (guard['outer'], guard['inner'])
                    or deployment_video in guard['fit'] or guard['inner']==guard['outer']
                    or sha256(training/'final.pt')!=manifest['checkpoint_sha256']
                    or guard['pretrained_sha256']!=sha256(weights)
                    or any(sha256(p)!=value for p, value in guard['source_sha256'].items())):
                raise ValueError('Exact completed fit-only model and encoder-level excluded deployment video required')
            checkpoint = torch.load(training/'final.pt', map_location='cpu', weights_only=False)
            state = checkpoint['model']
            self.model = FineGrainedDinoFishEncoder(pretrained, state['head.classifier.weight'].shape[0],
                guard['mode'], guard['local_count'], manifest['config']['train_backbone_suffix_blocks'])
            self.model.load_state_dict(state, strict=True)
            self.model.eval().requires_grad_(False)
            self.backbone = self.model.backbone
            self.metadata.update(coordinate_kind='supervised_DINO_'+guard['mode']+'_embedding384',
                trained_encoder=dict(path=str(training), checkpoint_sha256=manifest['checkpoint_sha256'],
                    manifest_sha256=sha256(training/'manifest.json'), fit=guard['fit'], inner=guard['inner'],
                    outer=guard['outer'], seed=guard['seed'], count=guard['local_count'], mode=guard['mode']),
                fitting_scope='Onlyfourfitting videos; distinctinner/outer excluded from visualgradients/state. Frozen inheriteddetector exposure remains explicit.')
        self.metadata['coordinate_sha256'] = sha256(weights) if self.model is None else self.metadata['trained_encoder']['checkpoint_sha256']
        self.backbone.eval().requires_grad_(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if self.model is None:
            self.backbone.to(device)
        else:
            self.model.to(device)

    def encode(self, rgb, boxes):
        start = time.perf_counter()
        crops, masks, valid = current_proposal_crops(rgb, boxes)
        timings = dict(current_crop_and_mask_seconds=time.perf_counter()-start,
                       resize_transfer_seconds=0., encoder_seconds=0., feature_transfer_seconds=0.)
        outputs = dict(embedding=np.zeros((len(boxes), 384), np.float32),
            raw_cls=np.zeros((len(boxes), 384), np.float32),
            local4=np.zeros((len(boxes), 4, 384), np.float32),
            visibility4=np.zeros((len(boxes), 4), np.float32), valid=valid)
        with torch.inference_mode():
            for offset in range(0, len(crops), self.batch_size):
                count = min(self.batch_size, len(crops)-offset)
                start = time.perf_counter()
                padded = torch.zeros(self.batch_size, 3, 128, 256)
                supported = torch.zeros(self.batch_size, 1, 128, 256)
                padded[:count], supported[:count] = crops[offset:offset+count], masks[offset:offset+count]
                images, fractions = resized_observations(padded.to(self.device), supported.to(self.device))
                if self.device=='cuda':
                    torch.cuda.synchronize()
                timings['resize_transfer_seconds'] += time.perf_counter()-start
                start = time.perf_counter()
                observed = self.backbone.forward_features(images)
                cls, tokens = observed['x_norm_clstoken'], observed['x_norm_patchtokens']
                raw_cls = F.normalize(cls.float(), dim=-1)
                embedding = raw_cls if self.model is None else self.model.head(cls, tokens, fractions)['embedding']
                spatial = spatial_observations(tokens, fractions)
                if self.device=='cuda':
                    torch.cuda.synchronize()
                timings['encoder_seconds'] += time.perf_counter()-start
                start = time.perf_counter()
                values = dict(embedding=embedding, raw_cls=raw_cls,
                    local4=spatial['local4'], visibility4=spatial['visibility4'])
                for name, value in values.items():
                    current = value[:count].float().cpu().numpy()
                    current[~valid[offset:offset+count]] = 0
                    if not np.isfinite(current).all():
                        raise ValueError('Nonfinite current-only DINO feature output')
                    outputs[name][offset:offset+count] = current
                timings['feature_transfer_seconds'] += time.perf_counter()-start
        outputs['stage_seconds'] = timings
        return outputs
