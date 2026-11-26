"""Frozen native-IR detector plus Train-fitted paired appearance; online only."""
import numpy as np
import torch
from tools.build_correlation_cache import auxiliary_input
from tools.fusion_memory_features import pool_p3
from tools.mft_identity_projection import paired_features, transform
from tools.native_pyramid_online_inference import OnlineNativeDetector
from yolox.tracker.fisher_identity_tracker import FisherIdentityTracker
from yolox.tracker.foveated_regions import exact_half_preproc


def fixed_cases(spec):
    return [dict(id='FF_native_zero', appearance_weight=0., lost_buffer=25)] + [
        dict(id=f'FF_native_w{int(w*100):03d}_b{b:03d}', appearance_weight=w, lost_buffer=b)
        for b in spec['association']['lost_buffers_frames'] for w in spec['association']['cost_weights']]


class FisherOnline:
    def __init__(self, teacher, module, visual_model, cases, calibration, parameters, association):
        self.teacher, self.parameters, self.calibration = teacher, parameters, calibration
        self.cases, self.sequence = list(cases), None
        self.visual_id = visual_model['id']
        self.detector = OnlineNativeDetector(teacher, [visual_model], {self.visual_id: module}, (800, 1440), calibration)
        self.trackers = {c['id']: FisherIdentityTracker(c['appearance_weight'], c['lost_buffer'], association) for c in cases}
        self.closed = False

    @property
    def needs_actual_IR_image(self):
        return True

    @torch.no_grad()
    def encode(self, rgb, ir, boxes):
        rgb_input, ratio = exact_half_preproc(rgb, (480, 864), (.485, .456, .406), (.229, .224, .225))
        norm = self.calibration['statistics']['ir']
        ir_input, ir_ratio = auxiliary_input(ir, (480, 864), norm['mean'], norm['std'])
        assert ratio == ir_ratio
        pooled = []
        for value in [rgb_input, ir_input]:
            feature = self.teacher.backbone(torch.from_numpy(value[None].copy()).to(
                device=next(self.teacher.parameters()).device, dtype=torch.float16))[0]
            result, valid = pool_p3(feature, boxes, tuple(rgb.shape[:2]), (480, 864), ratio)
            pooled.append(result.cpu().numpy())
        return transform(paired_features(*pooled), self.parameters)

    def step(self, rgb, ir, sequence, frame):
        if self.closed:
            raise ValueError('Closed engine')
        packets, info = self.detector.step(rgb, ir, sequence, frame)
        packet = packets[self.visual_id]
        vectors = self.encode(rgb, ir, packet[:, :4])
        rows, current = {}, {}
        for case in self.cases:
            name = case['id']
            tracks = self.trackers[name].update(sequence, frame, packet, vectors if case['appearance_weight'] else None)
            rows[name] = [(frame, t.tlwh.copy(), t.track_id, t.score) for t in tracks if t.tlwh[2]*t.tlwh[3] > 100]
            current[name] = packet
        self.sequence = sequence
        return rows, current, dict(info, Train_fitted_projection=True,
            appearance_input_hw=[480, 864], video_or_GT_identity_input=False,
            feature_or_detection_disk_cache=False)

    def close(self):
        self.closed = True
