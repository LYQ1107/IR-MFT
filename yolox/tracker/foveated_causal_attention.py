"""Coarse actual-IR evidence conditioned on causal physical-scene novelty."""
import cv2
import numpy as np

from yolox.tracker.foveated_regions import coarse_attention, working_region


class CausalAttention:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_frame = 0
        self.models = {cue: cv2.createBackgroundSubtractorMOG2(
            history=256, varThreshold=16, detectShadows=False) for cue in ['rgb', 'ir']}

    def update(self, frame, rgb, ir, boxes, scores):
        if frame != self.last_frame+1:
            raise ValueError('Consume exactly one current frame at a time; reset between sequences')
        if ir.ndim != 2 or rgb.shape[:2] != ir.shape or ir.dtype != np.uint8:
            raise ValueError('Require an actual aligned-size RGB/new-IR pair')
        self.last_frame = frame
        rgb_input = cv2.cvtColor(cv2.resize(rgb, (320,180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB)
        ir_input = cv2.cvtColor(cv2.resize(ir, (320,180), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        masks = {cue: cv2.GaussianBlur(self.models[cue].apply(value, learningRate=-1).astype(np.float32)/255,
                                      (5,5), 1.) for cue,value in [('rgb',rgb_input), ('ir',ir_input)]}
        support = working_region(rgb, boxes, scores)
        rgb_heat = coarse_attention(rgb, boxes, scores, rgb.shape, support=support)
        reference = rgb_heat*masks['rgb']
        if float(ir.std()) < 1e-6:
            return {key:reference.copy() for key in ['rgb','ir_self','ir_physical','ir_consensus']}
        ir_heat = coarse_attention(ir, boxes, scores, rgb.shape, support=support)
        return dict(rgb=reference, ir_self=ir_heat*masks['ir'],
                    ir_physical=ir_heat*masks['rgb'],
                    ir_consensus=ir_heat*np.sqrt(masks['rgb']*masks['ir']))
