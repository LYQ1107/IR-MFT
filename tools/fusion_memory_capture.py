"""Read-only current P3 capture for a future Dev-qualified F4 workflow.

No checkpoint loader, parent selection, image reader, tracker call or GPU launch.
The engineering probe uses synthetic inputs/untrained modules only. A separate
formal workflow must establish visual eligibility before learned fusion is used.
"""
import hashlib
import math

import numpy as np
import torch

from tools.fusion_memory_features import pool_p3


class FusionFeatureCapture:
    def __init__(self, module, *, feature_source, quality_policy):
        if feature_source not in ['rgb', 'fused'] or quality_policy not in ['uniform', 'attention_proxy']:
            raise ValueError('Explicit registered feature source and quality policy required')
        self.feature_source, self.quality_policy = feature_source, quality_policy
        self.module, self.handles = module, []
        self.sequence, self.last_frame, self.current = None, 0, None
        self.closed = False
        self._clear()
        if module is not None:
            self.handles.append(module.register_forward_pre_hook(self._before, with_kwargs=True))
            if feature_source == 'fused':
                self.handles.append(module.encoder.register_forward_hook(self._encoded))
            self.handles.append(module.register_forward_hook(self._after, with_kwargs=True))

    def _clear(self):
        self.seen, self.ready = False, False
        self.rgb = self.auxiliary = self.fused = self.gate = self.raw_auxiliary = None
        self.image_available, self.auxiliary_source = None, None

    def begin_frame(self, sequence, frame, original_hw, input_hw, resize_ratio):
        if self.closed or self.current is not None:
            raise ValueError('Close/consume/abort the outstanding frame before starting another')
        if not sequence or not isinstance(frame, int) or frame < 1:
            raise ValueError('Explicit causal sequence and frame required')
        if len(original_hw) != 2 or len(input_hw) != 2 or any(not isinstance(v, int) or v < 1 for v in [*original_hw, *input_hw]):
            raise ValueError('Explicit positive source and network geometry required')
        expected = min(input_hw[0]/original_hw[0], input_hw[1]/original_hw[1])
        if not math.isfinite(resize_ratio) or not math.isclose(resize_ratio, expected, rel_tol=1e-7, abs_tol=1e-8):
            raise ValueError('The actual aspect-preserving resize ratio is required')
        if self.sequence != sequence:
            if frame != 1:
                raise ValueError('A new video begins at frame1')
            self.sequence, self.last_frame = sequence, 0
        if frame != self.last_frame+1:
            raise ValueError('Deliver every consecutive frame exactly once')
        self._clear()
        self.current = dict(sequence=sequence, frame=frame, original_hw=tuple(original_hw),
            input_hw=tuple(input_hw), resize_ratio=float(resize_ratio))

    def _p3(self, value):
        if (not torch.is_tensor(value) or value.ndim != 4 or value.shape[0] != 1
                or value.shape[1] < 1 or not value.is_floating_point()
                or tuple(8*v for v in value.shape[-2:]) != self.current['input_hw']):
            raise ValueError('One actual stride8 P3 map from the declared input is required')
        return value.detach()

    def _before(self, module, args, kwargs):
        if self.current is None or self.seen:
            raise ValueError('Exactly one visual forward per armed current frame')
        if kwargs.get('sequence') != self.current['sequence'] or kwargs.get('frame') != self.current['frame']:
            raise ValueError('The visual forward must carry the same explicit sequence/frame')
        self.seen = True

    def _encoded(self, module, args, output):
        if self.current is None or not self.seen or self.raw_auxiliary is not None:
            raise ValueError('One current native encoder delivery, never a stale feature')
        if len(output) != 3 or len(output[0]) != 3:
            raise ValueError('Current native pyramid/valid/statistics interface required')
        self.raw_auxiliary = self._p3(output[0][0])

    def _after(self, module, args, kwargs, output):
        rgb = args[0] if args else kwargs['rgb']
        if len(rgb) != 3:
            raise ValueError('The visual RGB pyramid must contain all three levels')
        self.rgb = self._p3(rgb[0])
        # The strong ordinary RGB memory control reads no auxiliary output or diagnostics.
        if self.feature_source == 'rgb' and self.quality_policy == 'uniform':
            self.ready = True
            return
        information = output[2]
        self.image_available = not bool(information['missing_exact_RGB'])
        if self.image_available:
            if self.quality_policy == 'attention_proxy':
                gate = information['levels'][0]['gate']
                if gate.shape == (1, 1):
                    gate = gate[:, :, None, None].expand(1, 1, *self.rgb.shape[-2:])
                if gate.shape != self.rgb[:, :1].shape:
                    raise ValueError('Actual scalar/per-location P3 attention weights required')
                self.gate = gate.detach()
            if self.feature_source == 'fused':
                if self.raw_auxiliary is None:
                    raise ValueError('An available visual observation must have encoded its current cue')
                state = output[1]
                if state is not None:
                    if state.sequence != self.current['sequence'] or state.frame != self.current['frame']:
                        raise ValueError('Filtered auxiliary state is from another video/frame')
                    self.auxiliary = self._p3(state.latent[0])
                    self.auxiliary_source = 'current_filtered_native_state'
                else:
                    self.auxiliary = self.raw_auxiliary
                    self.auxiliary_source = 'current_native_encoder'
                self.raw_auxiliary = None
                self.fused = self._p3(output[0][0])
                if self.fused.shape != self.rgb.shape:
                    raise ValueError('RGB and actual fused P3 must share dimensions')
        else:
            if self.raw_auxiliary is not None:
                raise ValueError('A missing observation unexpectedly encoded auxiliary pixels')
        self.ready = True

    def capture_rgb_only(self, sequence, frame, rgb_p3):
        if self.module is not None or self.current is None or self.seen:
            raise ValueError('Explicit one-time source capture is only for a direct RGB source')
        if (sequence, frame) != (self.current['sequence'], self.current['frame']):
            raise ValueError('Wrong source video/frame')
        self.rgb, self.seen, self.ready, self.image_available = self._p3(rgb_p3), True, True, False

    def _pool(self, feature, boxes, normalize=True):
        value, valid = pool_p3(feature, boxes, self.current['original_hw'], self.current['input_hw'],
            self.current['resize_ratio'], normalize=normalize)
        return value.cpu().numpy().copy(), valid.cpu().numpy().copy()

    def take_packet(self, sequence, frame, detections):
        if self.current is None or not self.ready or (sequence, frame) != (self.current['sequence'], self.current['frame']):
            raise ValueError('One completed matching current visual frame is required')
        detections = np.asarray(detections)
        if (detections.ndim != 2 or detections.shape[1] != 5 or not np.isfinite(detections).all()
                or ((detections[:, 4] < 0) | (detections[:, 4] > 1)).any()
                or (detections[:, 2:4] <= detections[:, :2]).any()):
            raise ValueError('Finite original-coordinate xyxy/score observations required')
        boxes = np.asarray(detections[:, :4], dtype=np.float32).copy()
        rgb, rgb_valid = self._pool(self.rgb, boxes)
        scaled = boxes*np.float32(self.current['resize_ratio'])
        height, width = (int(v*self.current['resize_ratio']) for v in self.current['original_hw'])
        scaled[:, 0::2] = scaled[:, 0::2].clip(0, width)
        scaled[:, 1::2] = scaled[:, 1::2].clip(0, height)
        geometry_valid = (scaled[:, 2] > scaled[:, 0]) & (scaled[:, 3] > scaled[:, 1])
        available = geometry_valid & bool(self.image_available)
        packet = dict(rgb=rgb, auxiliary_available=available, quality=np.ones(len(boxes), dtype=np.float32))
        descriptor_validity = dict(rgb=rgb_valid.tolist())
        if bool(available.any()):
            if self.quality_policy == 'attention_proxy':
                if not bool(torch.isfinite(self.gate).all()) or not bool(((self.gate >= 0) & (self.gate <= 1)).all()):
                    raise ValueError('Attention proxy must be finite and bounded, not inferred uncertainty')
                proxy, _ = self._pool(self.gate, boxes, normalize=False)
                # Zero is a valid attention value; zero-vector descriptor validity does not define q.
                packet['quality'][available] = proxy[available, 0].clip(0, 1)
            if self.feature_source == 'fused':
                for key, feature in [('auxiliary', self.auxiliary), ('fused', self.fused)]:
                    packet[key], valid = self._pool(feature, boxes)
                    descriptor_validity[key] = valid.tolist()
        packet['metadata'] = dict(self.current, feature_source=self.feature_source, quality_policy=self.quality_policy,
            auxiliary_image_available=self.image_available, auxiliary_source=self.auxiliary_source,
            ROI_geometry_valid=geometry_valid.tolist(), descriptor_validity=descriptor_validity,
            detector_packet_sha256=hashlib.sha256(np.asarray(detections, dtype='<f4').tobytes()).hexdigest(),
            quality_is_calibrated_reliability=False, formal_visual_eligibility_established_by_capture=False)
        self.last_frame = frame
        self.current = None
        self._clear()
        return packet

    def abort_frame(self):
        self.current = None
        self._clear()

    def close(self):
        self.abort_frame()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.closed = True
