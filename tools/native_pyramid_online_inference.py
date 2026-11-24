"""Sequential native fusion inference; no GT, checkpoint selection or GPU launch.

The caller owns source/checkpoint verification, physical-device release, complete
sequence coverage, tracking, official evaluation and performance measurement.
"""
import numpy as np
import torch

from tools.extract_structural_relative import detections
from tools.native_pyramid_train_data import current_to_previous_flow, detach_state, preprocess_observation
from yolox.utils import postprocess


class OnlineNativeDetector:
    def __init__(self, model, rows, modules, input_hw, calibration):
        self.model, self.rows, self.modules = model, list(rows), dict(modules)
        self.input_hw, self.calibration = tuple(input_hw), calibration
        self.device, self.dtype = next(model.parameters()).device, next(model.parameters()).dtype
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('The original RGB encoder/head must be frozen in evaluation mode')
        ids = [r['id'] for r in self.rows]
        if len(ids) != len(set(ids)) or set(ids) != set(self.modules):
            raise ValueError('Every unique registered observation needs one explicit module entry')
        for row in self.rows:
            if row.get('seed', 0) != 0 or row['cue'] not in ['rgb', 'gray', 'ir']:
                raise ValueError('Only the registered seed0 matched cues are supported')
            intervention = row.get('intervention', 'normal')
            if intervention not in ['normal', 'unavailable', 'IR_image_spatial_roll']:
                raise ValueError('Unknown image intervention')
            if intervention != 'normal' and row['cue'] != 'ir':
                raise ValueError('The registered interventions apply to actual November IR')
            module = self.modules[row['id']]
            if module is None:
                if not row.get('baseline') or row['cue'] != 'rgb' or intervention != 'normal':
                    raise ValueError('A missing module is allowed only for the explicit RGB source')
            elif module.training or any(p.requires_grad for p in module.parameters()):
                raise ValueError('Loaded native modules must be frozen in evaluation mode')
        self.sequence, self.last_frame, self.previous_gray, self.states = None, 0, None, {}

    @property
    def needs_actual_IR_image(self):
        return any(r['cue'] == 'ir' and r.get('intervention') != 'unavailable'
                   and not r.get('baseline') for r in self.rows)

    def reset(self, sequence):
        self.sequence, self.last_frame, self.previous_gray = sequence, 0, None
        self.states = {r['id']: None for r in self.rows}

    @torch.no_grad()
    def step(self, rgb, ir, sequence, frame):
        if not sequence or not isinstance(frame, int) or frame < 1:
            raise ValueError('One current frame with an explicit video identity is required')
        if self.sequence != sequence:
            if frame != 1:
                raise ValueError('A new sequence must start at frame1')
            self.reset(sequence)
        if frame != self.last_frame + 1:
            raise ValueError('Only consecutive online frames are allowed')
        if self.needs_actual_IR_image and (ir is None or ir.shape != rgb.shape[:2] or
                ir.ndim != 2 or ir.dtype != np.uint8):
            raise ValueError('The registered actual November IR image is required')
        source = preprocess_observation(rgb, None, 'rgb', self.input_hw, self.calibration)
        pixels, _, valid_array, gray, ratio, _ = source
        tensor = lambda value: torch.from_numpy(value[None].copy()).to(self.device)
        valid = tensor(valid_array)
        features = self.model.backbone(tensor(pixels).to(dtype=self.dtype))
        teacher = self.model.head(features)
        if teacher.shape[0] != 1 or teacher.shape[-1] != 6:
            raise ValueError('The original single-class YOLOX decoded head is required')
        score_maps, offset = [], 0
        for value in features:
            count = value.shape[-2] * value.shape[-1]
            probability = teacher[:, offset:offset+count, 4] * teacher[:, offset:offset+count, 5]
            score_maps.append(probability.reshape(1, 1, *value.shape[-2:]))
            offset += count
        if offset != teacher.shape[1]:
            raise ValueError('Original decoded anchors do not cover the RGB pyramid')
        flow = None
        if any(module is not None and module.mode == 'temporal' and module.use_memory and module.use_motion
               and self.states[row['id']] is not None for row in self.rows
               for module in [self.modules[row['id']]]):
            flow = current_to_previous_flow(gray, self.previous_gray)
        cache, outputs, exactness = {('rgb', 'normal'): (tensor(source[1]), True)}, {}, {}
        for row in self.rows:
            name, module = row['id'], self.modules[row['id']]
            if module is None:
                prediction, available = teacher, True
            else:
                cue, intervention = row['cue'], row.get('intervention', 'normal')
                key = cue, intervention
                if intervention == 'unavailable':
                    auxiliary, available = None, False
                else:
                    if key not in cache:
                        image = ir
                        if intervention == 'IR_image_spatial_roll':
                            image = np.roll(ir, (ir.shape[0]//4, ir.shape[1]//4), (0, 1))
                        values = preprocess_observation(rgb, image, cue, self.input_hw, self.calibration)
                        cache[key] = tensor(values[1]), values[-1]
                    auxiliary, available = cache[key]
                repaired, state, _ = module(features, auxiliary, score_maps, valid=valid,
                    available=available, state=self.states[name], sequence=sequence, frame=frame,
                    flow_to_previous=flow)
                self.states[name] = detach_state(state)
                prediction = self.model.head(repaired)
                if row.get('geometry_fixed', False):
                    prediction = torch.cat((teacher[:, :, :4], prediction[:, :, 4:]), -1)
                if not available or row.get('expect_RGB_exact', False):
                    if not torch.equal(prediction, teacher):
                        raise RuntimeError('Unavailable or zero native fusion changed the full B1 RGB detector')
                exactness[name] = bool(torch.equal(prediction, teacher))
            if not bool(torch.isfinite(prediction).all()):
                raise RuntimeError('Native inference produced nonfinite decoded predictions')
            selected = postprocess(prediction.clone(), 1, .1, .7)[0]
            outputs[name] = detections(selected, ratio)
        self.previous_gray, self.last_frame = gray, frame
        return outputs, dict(sequence=sequence, frame=frame, independent_input_batch_size=1,
            full_head_exact_to_fresh_RGB=exactness, IR_image_required=self.needs_actual_IR_image,
            GT_read=False, future_frame_input=False, previous_outputs_rewritten=False)
