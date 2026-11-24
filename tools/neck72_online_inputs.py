"""Uncached current-image inputs for the registered neck72 query backends.

Only consumed fields are produced. The RGB point scorer uses bbox/RGB modes;
neither point mode depends on IR. The additional native vector and its scalar
quality need actual IR only when enabled, while the fixed V3 frontend always
needs its actual paired image. No proposal, embedding or point bank is an
inference input to this module.
"""
import copy
from pathlib import Path
from types import FunctionType

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from tools.native_neck_io import load_native_encoder
from tools.neck72_frontend_serialization import load_verified_frontend_model
from tools.neck72_runtime import read
from tools.run_pseudo_ir_comparison import sha256
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.tracker import foreground_transport as original_points
from yolox.tracker.native_cue import native_crop, native_input
from yolox.tracker.perceptual_features import (clipped_crop, foreground, hog,
    color_histogram, letterbox_tensor_array)
from yolox.utils import postprocess

ROOT = Path(__file__).resolve().parents[1]
RGB_MEAN = (.485, .456, .406)
RGB_STD = (.229, .224, .225)
LAGS = (1, 2, 4, 8)
POINT_MODES = ('bbox', 'rgb')
FEATURE_FIELDS = ('embedding', 'rgb_mask_embedding', 'rgb_hog', 'color', 'quality',
                  'native_reference', 'native_auxiliary')


def needs_ir(config):
    return (config['detection_arm'] == 'new' or
            (config['backend'] != 'byte' and config['native_cue'] == 'ir'
             and not config.get('disable_native_increment', False)))


def rgb_masks(bgr, unused_ir, box):
    """The literal RGB part of region_masks, without accessing IR pixels."""
    h, w = bgr.shape[:2]
    x1, y1 = np.maximum(np.floor(box[:2]).astype(int), 0)
    x2, y2 = np.minimum(np.ceil(box[2:]).astype(int), [w, h])
    x1, y1 = min(x1, w-1), min(y1, h-1)
    x2, y2 = max(x1+1, x2), max(y1+1, y2)
    rgb = cv2.resize(bgr[y1:y2, x1:x2], (64, 32))
    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[:3].reshape(-1, 3), lab[-3:].reshape(-1, 3),
        lab[:, :3].reshape(-1, 3), lab[:, -3:].reshape(-1, 3)])
    distance = np.linalg.norm(lab-np.median(border, axis=0), axis=-1).clip(0, 255).astype(np.uint8)
    mask, quality = foreground(distance)
    if quality < .15:
        mask = np.ones_like(mask)
    return dict(bbox=np.ones_like(mask), rgb=mask), [quality, np.nan]


def _point_function(original, **overrides):
    namespace = dict(original.__globals__, MODES=POINT_MODES, **overrides)
    return FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)


# Keep the original point/LK/voting calculation bodies. Pruning unused modes
# changes the pooled point list, so exact real-image comparisons are required.
rgb_point_groups = _point_function(original_points.point_groups, region_masks=rgb_masks)
rgb_frame_correspondences = _point_function(original_points.frame_correspondences)


def appearance_inputs(rgb, ir, boxes, ir_quality):
    """Prepare original two consumed FP16 appearance views and descriptors."""
    arrays = dict(rgb_hog=np.empty((len(boxes), 144), np.float16),
        color=np.empty((len(boxes), 128), np.float16),
        quality=np.full((len(boxes), 2), np.nan, np.float16))
    views = []
    for index, box in enumerate(boxes):
        patch = clipped_crop(rgb, box, .05)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        mask, quality = foreground(gray)
        arrays['rgb_hog'][index] = hog(gray)
        arrays['color'][index] = color_histogram(patch)
        arrays['quality'][index, 0] = quality
        if ir_quality:
            if ir is None:
                raise ValueError('Enabled native IR quality requires the actual current image')
            arrays['quality'][index, 1] = foreground(clipped_crop(ir, box, .05))[1]
        views.extend([letterbox_tensor_array(rgb, box), letterbox_tensor_array(rgb, box, mask)])
    return arrays, views


def detector_input(rgb, ir, metadata):
    size = tuple(metadata['input_size'])
    assert size == (800, 1440)
    value, ratio = preproc(rgb, size, RGB_MEAN, RGB_STD)
    if metadata['variant'] != 'rgb':
        if ir is None or ir.ndim != 2 or ir.dtype != np.uint8 or ir.shape != rgb.shape[:2]:
            raise ValueError('The V3 frontend requires actual paired November scalar IR')
        auxiliary, second = preproc(np.repeat(ir[:, :, None], 3, axis=2), size,
                                    metadata['ir_means'], metadata['ir_std'])
        assert ratio == second
        value = np.concatenate([value, auxiliary], axis=0)
    return np.ascontiguousarray(value, dtype=np.float32), ratio


class FrozenFrontend:
    def __init__(self, plan, arm, split, device):
        prefix = 'new_diffv2ir' if arm == 'new' else 'rgb'
        source = Path(plan['frozen_detection_sources'][prefix+'_'+split])
        self.manifest_path = source / 'metrics.json'
        self.metadata = read(self.manifest_path)['protocol']
        expected_variant = 'diffv2ir' if arm == 'new' else 'rgb'
        assert self.metadata['variant'] == expected_variant and self.metadata['seed'] == 0
        assert self.metadata['conf'] == .1 and self.metadata['nms'] == .7
        state, serialization = load_verified_frontend_model(plan, arm, split, self.metadata, plan['work_root'])
        experiment = 'yolox_mft26ir_pseudo_ir_comparison.py' if arm == 'new' else 'yolox_mft26ir_visible.py'
        exp = get_exp(str(ROOT / 'exps' / experiment), None)
        self.model = exp.get_model()
        self.model.load_state_dict(state, strict=True)
        del state
        self.model = self.model.to(device).half().eval().requires_grad_(False)
        self.device = device
        self.provenance = dict(**serialization,
            source_manifest=str(self.manifest_path), source_manifest_sha256=sha256(self.manifest_path),
            precision='FP16, original unfused model and postprocess; independent current frame')

    @torch.inference_mode()
    def predict(self, rgb, ir):
        array, ratio = detector_input(rgb, ir, self.metadata)
        tensor = torch.from_numpy(array[None]).to(device=self.device, dtype=torch.float16)
        output = postprocess(self.model(tensor), 1, .1, .7)[0]
        if output is None:
            return np.empty((0, 5), np.float32)
        value = output.float().cpu().numpy()
        return np.column_stack([value[:, :4]/ratio, value[:, 4]*value[:, 5]]).astype(np.float32)


class CurrentFeatures:
    def __init__(self, config, work, protocol, device='cuda'):
        self.config, self.device = config, device
        reference, metadata = load_native_encoder('reference', 0, work, protocol)
        self.appearance = copy.deepcopy(reference).to(device).half().eval().requires_grad_(False)
        self.reference = reference.to(device).float().eval().requires_grad_(False)
        self.native, self.rotation = None, None
        self.provenance = dict(reference=metadata, appearance=metadata,
            appearance_precision='FP16 pre-neck', reference_precision='FP32 post-neck')
        if not config.get('disable_native_increment', False):
            self.native, endpoint = load_native_encoder(config['native_cue'], 0, work, protocol)
            self.native = self.native.to(device).float().eval().requires_grad_(False)
            self.provenance['native'] = endpoint
            alignment_path = Path(work) / 'native_neck_alignment/complete.json'
            alignment = read(alignment_path)
            assert alignment['Dev_rows'] == alignment['Test_rows'] == 0
            info = alignment['outputs'][config['native_cue']+'_s0']
            assert sha256(info['path']) == info['sha256']
            with np.load(info['path']) as arrays:
                self.rotation = arrays['rotation'].copy()
            assert self.rotation.shape == (512, 512)
            self.provenance['alignment'] = dict(path=str(alignment_path), sha256=sha256(alignment_path),
                rotation_sha256=info['sha256'], applied=not config.get('disable_native_alignment', False))

    @torch.inference_mode()
    def _native(self, model, image, boxes):
        result = np.empty((len(boxes), 512), np.float32)
        for start in range(0, len(boxes), 64):
            crops = [native_crop(image, box) for box in boxes[start:start+64]]
            pixels = torch.from_numpy(np.stack([c[0] for c in crops]))
            rectangles = torch.from_numpy(np.stack([c[1] for c in crops]))
            value = native_input(pixels, rectangles, self.device)
            raw = model.projection(model.backbone(value))
            result[start:start+len(crops)] = F.normalize(model.neck(raw), dim=-1).cpu().numpy()
        return result

    @torch.inference_mode()
    def predict(self, rgb, ir, boxes):
        use_ir = self.native is not None and self.config['native_cue'] == 'ir'
        arrays, views = appearance_inputs(rgb, ir, boxes, use_ir)
        encoded = np.empty((len(views), 512), np.float16)
        for start in range(0, len(views), 64):
            batch = torch.from_numpy(np.stack(views[start:start+64])).to(device=self.device, dtype=torch.float16)
            encoded[start:start+len(batch)] = self.appearance(batch).cpu().numpy()
        arrays.update(embedding=encoded[::2], rgb_mask_embedding=encoded[1::2],
                      native_reference=self._native(self.reference, rgb, boxes))
        if self.native is not None:
            image = ir if use_ir else rgb
            value = self._native(self.native, image, boxes)
            if not self.config.get('disable_native_alignment', False):
                value = value @ self.rotation
            arrays['native_auxiliary'] = np.asarray(value, np.float32)
        return arrays

    def models(self):
        return {k: v for k, v in dict(appearance=self.appearance, reference=self.reference,
                                    native=self.native).items() if v is not None}


class CurrentPointStream:
    """At most eight earlier resized RGB images, with current original modes."""
    def __init__(self):
        self.history = {}

    def step(self, frame, rgb, boxes):
        scale = np.array([960/rgb.shape[1], 540/rgb.shape[0]]*2, np.float32)
        small = cv2.resize(rgb, (960, 540), interpolation=cv2.INTER_AREA)
        regions = rgb_point_groups(small, None, boxes*scale, maximum=20)
        result = dict(quality=regions['quality'].astype(np.float16))
        for lag in LAGS:
            if frame-lag in self.history:
                old = self.history[frame-lag]
                values = rgb_frame_correspondences(old['rgb'], small,
                    {'current': old['regions']}, {'current': regions})
                for source in POINT_MODES:
                    result[f'total_{lag}_{source}'] = old['regions']['groups'][source]['total']
                    for target in POINT_MODES:
                        matrix, good = values[('current', source, target)]
                        result[f'vote_{lag}_{source}_{target}'] = matrix.ravel().astype(np.float16)
                        result[f'good_{lag}_{source}'] = good
            else:
                for source in POINT_MODES:
                    result[f'total_{lag}_{source}'] = np.empty(0, np.uint8)
                    result[f'good_{lag}_{source}'] = np.empty(0, np.uint8)
                    for target in POINT_MODES:
                        result[f'vote_{lag}_{source}_{target}'] = np.empty(0, np.float16)
        self.history[frame] = dict(rgb=small, regions=regions)
        self.history = {f: v for f, v in self.history.items() if frame-f < max(LAGS)}
        assert len(self.history) <= 8
        return result


class PrefixPackets:
    """Append only already-computed values, retaining at most a declared prefix.

    This bounded buffer is for the preregistered runtime sample, not a claim
    of an unbounded video-serving implementation or a copied feature bank.
    """
    def __init__(self, frames):
        if not 1 <= frames <= 64:
            raise ValueError('Runtime prefix buffer must be explicitly bounded to at most64frames')
        self.limit, self.data = frames, dict(frame_ids=np.empty(0, int), offsets=np.zeros(1, int))
        self.data['foreground_transport'] = dict(frame_ids=np.empty(0, int), detection_offsets=np.zeros(1, int))
        for lag in LAGS:
            self.data['foreground_transport'].update({f'vote_offsets_{lag}': np.zeros(1, int),
                                                     f'count_offsets_{lag}': np.zeros(1, int)})

    @staticmethod
    def extend(target, key, value):
        value = np.asarray(value)
        target[key] = np.concatenate([target[key], value]) if key in target else value.copy()

    def append(self, frame, detections, features, points):
        if frame != len(self.data['frame_ids'])+1 or frame > self.limit:
            raise ValueError('Only sequential current frames within the registered prefix')
        data = self.data; transport = data['foreground_transport']; count = len(detections)
        self.extend(data, 'frame_ids', [frame]); self.extend(data, 'offsets', [int(data['offsets'][-1])+count])
        self.extend(data, 'boxes', detections[:, :4]); self.extend(data, 'scores', detections[:, 4])
        for key, value in features.items():
            assert len(value) == count
            self.extend(data, key, value)
        self.extend(transport, 'frame_ids', [frame])
        self.extend(transport, 'detection_offsets', [int(data['offsets'][-1])])
        for lag in LAGS:
            previous = int(data['offsets'][frame-lag]-data['offsets'][frame-lag-1]) if frame > lag else 0
            for name, increment in [('vote', previous*count), ('count', previous)]:
                key = f'{name}_offsets_{lag}'
                self.extend(transport, key, [int(transport[key][-1])+increment])
        for key, value in points.items():
            self.extend(transport, key, value)
        return data
