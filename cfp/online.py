"""Current-frame RGB -> causal response -> V3 proposals -> fish embeddings.

This module never loads annotation labels or future images. Detector and encoder
batch padding is explicit because CUDA kernel choices can depend on batch shape.
"""
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from .identity_data import crop_to_tensor
from .pseudo_ir import CausalPseudoIR
from .provenance import sha256
from .reid import FishReID
from yolox.data import RGBTValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess


V3_CHECKPOINT_SHA256 = 'c448238beaa8540860d17f6a022d99b29b1407649823398a8553fa0890b6d786'


class CurrentFrameRGBTTransform:
    """Exact uint8 resize/normalization using per-channel lookup tables.

    Original preprocessing resizes uint8, stores integer pixels in float64,
    then divides/subtracts/divides and casts to float32. These tables retain
    that operation order without full float64 image buffers.
    """
    def __init__(self, rgb_means, rgb_std, ir_means, ir_std):
        self.tables = []
        for means, std in ((rgb_means, rgb_std), (ir_means, ir_std)):
            values = np.tile(np.arange(256, dtype=np.float64)[:, None], (1, 3))
            values /= 255.
            values -= means
            values /= std
            self.tables.append(values.astype(np.float32))

    def __call__(self, rgb, ir, targets, input_dim):
        if rgb.dtype != np.uint8 or ir.dtype != np.uint8 or rgb.shape != ir.shape:
            raise ValueError('Exact lookup preprocessing requires aligned uint8 images')
        h, w = input_dim
        ratio = min(h / rgb.shape[0], w / rgb.shape[1])
        rh, rw = int(rgb.shape[0] * ratio), int(rgb.shape[1] * ratio)
        result = np.empty((6, h, w), dtype=np.float32)
        for branch, (pixels, table) in enumerate(zip((rgb, ir), self.tables)):
            resized = cv2.resize(pixels, (rw, rh), interpolation=cv2.INTER_LINEAR)
            for channel in range(3):
                plane = result[branch * 3 + channel]
                plane.fill(table[114, channel])
                plane[:rh, :rw] = table[resized[:, :, 2 - channel], channel]
        return result, np.zeros((1, 5), np.float32)


def jpeg_response(image, quality=95):
    """Match the established causal JPEG input, with no disk write/lookahead."""
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError('Response JPEG encoding failed')
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape != image.shape:
        raise ValueError('Response JPEG roundtrip changed dimensions')
    return decoded


def proposal_crop_arrays(rgb, ir, boxes, rgb_only=False):
    arrays, valid = [], []
    for image in ((rgb,) if rgb_only else (rgb, ir)):
        rows, flags = [], []
        for box in boxes:
            xywh = [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]
            try:
                crop = crop_to_tensor(image, xywh)
            except ValueError:
                crop = torch.zeros(3, 128, 256)
                flags.append(False)
            else:
                flags.append(True)
            rows.append(crop)
        arrays.append(torch.stack(rows) if rows else torch.empty(0, 3, 128, 256))
        valid.append(np.asarray(flags, bool))
    if rgb_only:
        arrays.append(torch.empty(0, 3, 128, 256))
        valid.append(np.empty(0, bool))
    return arrays, valid


def identity_ir_view(rgb, ir, intervention='none'):
    """Current-frame pixel interventions; detector/generator pixels stay intact."""
    if intervention == 'none':
        return ir
    if intervention == 'zero':
        return np.zeros_like(ir)
    if intervention == 'repeat_rgb':
        return rgb
    if intervention == 'spatial_shuffle':
        # Fixed deterministic permutation, no future frame or identity labels.
        order = np.random.default_rng(20261012).permutation(ir.shape[0] * ir.shape[1])
        return ir.reshape(-1, 3)[order].reshape(ir.shape)
    raise ValueError('Unknown current-frame IR pixel intervention')


class CurrentFrameFrontend:
    def __init__(self, detector_checkpoint, scripts_root, rgb_checkpoint=None,
                 ir_checkpoint=None, detector_batch=8, encoder_batch=128,
                 input_regime='causal', protocol_sha256=None, preprocessing='original', detector_runtime_python=None,
                 detector_runtime_factory=None, skip_unused_ir_crops=False):
        if detector_batch not in (1, 8) or encoder_batch != 128:
            raise ValueError('Declared current-frame padding requires detector1/8 and encoder128')
        if input_regime not in ('causal', 'causal_precomputed', 'legacy_reproduction'):
            raise ValueError('Explicit input provenance required')
        if sha256(detector_checkpoint) != V3_CHECKPOINT_SHA256:
            raise ValueError('Frozen V3-E3 checkpoint changed')
        self.input_regime = input_regime
        self.scripts_root = Path(scripts_root)
        self.generator = CausalPseudoIR(scripts_root, mode='temporal_ema', alpha=.01)
        self.detector_batch = detector_batch
        self.encoder_batch = encoder_batch
        self.skip_unused_ir_crops = skip_unused_ir_crops
        self.exp = get_exp('exps/yolox_mft26ir_rgbt_v3.py', None)
        self.transform = RGBTValTransform(rgb_means=self.exp.rgb_means, std=self.exp.rgb_std,
                                         ir_means=self.exp.ir_means, ir_std=self.exp.ir_std)
        if preprocessing not in ('original', 'lookup_uint8'):
            raise ValueError('Explicit original or exact lookup preprocessing required')
        if preprocessing == 'lookup_uint8':
            self.transform = CurrentFrameRGBTTransform(self.exp.rgb_means, self.exp.rgb_std,
                                                       self.exp.ir_means, self.exp.ir_std)
        self.detector_worker = None
        self.detector = None
        if detector_runtime_python:
            from .detector_runtime import DetectorRuntime
            runtime_type = detector_runtime_factory or DetectorRuntime
            self.detector_worker = runtime_type(detector_runtime_python, detector_checkpoint, scripts_root, detector_batch, preprocessing)
            self.detector_parameters = self.detector_worker.metadata['parameters']
        else:
            self.detector = self.exp.get_model().cuda()
            checkpoint = torch.load(detector_checkpoint, map_location='cpu', weights_only=False)
            self.detector.load_state_dict(checkpoint['model'], strict=True)
            del checkpoint
            self.detector.half().eval().requires_grad_(False)
            self.detector_parameters = sum(v.numel() for v in self.detector.parameters())
        self.rgb_encoder, self.rgb_metadata = self._load_encoder(rgb_checkpoint, protocol_sha256)
        self.ir_encoder, self.ir_metadata = self._load_encoder(ir_checkpoint, protocol_sha256)
        if ir_checkpoint and not rgb_checkpoint:
            raise ValueError('Paired feature extraction needs a fixed RGB branch')

    @staticmethod
    def _load_encoder(path, protocol_sha256):
        if path is None:
            return None, None
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        metadata = checkpoint['metadata']
        if checkpoint['epoch'] != 12 or metadata.get('crop_variant', 'original') != 'original':
            raise ValueError('Online trial requires original-crop fixed epoch12 fish encoder')
        if protocol_sha256 is not None and metadata['protocol_sha256'] != protocol_sha256:
            raise ValueError('Online encoder split differs')
        state = checkpoint['model']
        model = FishReID(state['classifier.weight'].shape[0], state['projection.weight'].shape[0])
        model.load_state_dict(state, strict=True)
        model.cuda().eval().requires_grad_(False)
        return model, dict(checkpoint_sha256=sha256(path), seed=checkpoint['seed'],
                           embedding_stage='post_neck', training_metadata=metadata)

    def reset(self):
        self.generator = CausalPseudoIR(self.scripts_root, mode='temporal_ema', alpha=.01)

    def close(self):
        return self.detector_worker.close() if self.detector_worker else None

    def encode_crops(self, crops, valid, model):
        if model is None:
            return None
        result = np.zeros((len(crops), 512), np.float32)
        with torch.inference_mode():
            for lo in range(0, len(crops), self.encoder_batch):
                current = crops[lo:lo + self.encoder_batch]
                padded = torch.zeros(self.encoder_batch, 3, 128, 256)
                padded[:len(current)] = current
                with torch.autocast('cuda', dtype=torch.float16):
                    features, _ = model(padded.cuda(), embedding_stage='post_neck')
                values = features[:len(current)].float().cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite online identity features')
                values[~valid[lo:lo + len(current)]] = 0.
                result[lo:lo + len(current)] = values
        return result

    def process(self, rgb, provided_ir=None, fixed_proposals=None, ir_intervention='none'):
        """Process exactly this frame; optional fixed proposals are Protocol A."""
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Current uint8 BGR frame required')
        timings = {}
        start = time.perf_counter()
        if self.input_regime == 'causal':
            if provided_ir is not None:
                raise ValueError('Causal mode generates its own current/past response')
            ir = jpeg_response(self.generator(rgb))
        else:
            if provided_ir is None or provided_ir.shape != rgb.shape:
                raise ValueError('Legacy reproduction requires the actual paired image')
            ir = provided_ir
        timings['generation_codec' if self.input_regime == 'causal' else 'provided_ir_validation'] = time.perf_counter() - start
        raw = None
        if fixed_proposals is None and self.detector_worker is not None:
            start = time.perf_counter()
            values, child_stages = self.detector_worker.process(rgb, ir)
            boxes, scores, raw = values['boxes'], values['scores'], values['raw_post_nms']
            timings.update({'worker_' + k: v for k, v in child_stages.items()})
            timings['detector_worker_roundtrip'] = time.perf_counter() - start
        elif fixed_proposals is None:
            start = time.perf_counter()
            image, _ = self.transform(rgb, ir, None, self.exp.test_size)
            tensor = torch.from_numpy(image)[None].repeat(self.detector_batch, 1, 1, 1).cuda().half()
            torch.cuda.synchronize()
            timings['detector_preprocess_transfer'] = time.perf_counter() - start
            start = time.perf_counter()
            with torch.inference_mode():
                outputs = postprocess(self.detector(tensor), 1, .1, .7)
            torch.cuda.synchronize()
            timings['detector_nms'] = time.perf_counter() - start
            start = time.perf_counter()
            raw = np.empty((0, 7), np.float32) if outputs[0] is None else outputs[0].float().cpu().numpy()
            scale = min(self.exp.test_size[0] / rgb.shape[0], self.exp.test_size[1] / rgb.shape[1])
            boxes, scores = (raw[:, :4] / scale).astype(np.float32), (raw[:, 4] * raw[:, 5]).astype(np.float32)
            if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
                raise ValueError('Nonfinite online detector output')
            timings['proposal_transfer'] = time.perf_counter() - start
        else:
            boxes, scores = fixed_proposals
            if boxes.shape != (len(scores), 4) or not np.isfinite(boxes).all() or not np.isfinite(scores).all():
                raise ValueError('Finite fixed current-frame proposals required')
        rgb_features = ir_features = None
        if self.rgb_encoder is not None:
            start = time.perf_counter()
            crops, valid = proposal_crop_arrays(rgb, identity_ir_view(rgb, ir, ir_intervention), boxes,
                rgb_only=self.skip_unused_ir_crops and self.ir_encoder is None)
            timings['crop_preparation'] = time.perf_counter() - start
            start = time.perf_counter()
            rgb_features = self.encode_crops(crops[0], valid[0], self.rgb_encoder)
            ir_features = self.encode_crops(crops[1], valid[1], self.ir_encoder)
            torch.cuda.synchronize()
            timings['encoders_transfer'] = time.perf_counter() - start
        return dict(boxes=boxes, scores=scores, raw_post_nms=raw, rgb_features=rgb_features,
                    ir_features=ir_features, ir_image=ir, stage_seconds=timings)
