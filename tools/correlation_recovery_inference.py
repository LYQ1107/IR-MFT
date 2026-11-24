"""Original CGR B4 inference and legacy-compatible causal track numbering.

No GPU launch, model selection, GT access or whole-video output buffers.
Callers must supply the frozen cases/checkpoints and current canonical packets.
"""
from pathlib import Path

import cv2
import numpy as np
import torch

from tools.build_correlation_cache import auxiliary_input
from tools.native_pyramid_stream_tracks import StreamTrackWriter
from tools.run_pseudo_ir_comparison import sha256
from yolox.models.local_correlation_repair import CorrelationRepair
from yolox.tracker.foveated_regions import exact_half_preproc
from yolox.tracker.priority_observations import priority_union
from yolox.utils import postprocess


def detections(value, ratio):
    if value is None:
        return np.empty((0, 5), np.float32)
    array = value.detach().float().cpu().numpy()
    return np.c_[array[:, :4] / ratio, array[:, 4] * array[:, 5]].astype(np.float32)


def load_adapters(cases, repair_sha256, source_RGB_sha256, device):
    """Strict trained seed0 provenance; no reinitialization or precision change."""
    loaded, information = {}, {}
    for row in cases:
        name = row['id']
        if name in loaded:
            raise ValueError('Unique registered cases required')
        if row.get('baseline'):
            loaded[name], information[name] = None, dict(row)
            continue
        source = row['source']
        if source['seed'] != 0 or source['variant'] not in {
                'full', 'aligned', 'blanket_counterfactual', 'no_curriculum'}:
            raise ValueError('Only the frozen seed0 CGR variants are supported')
        if sha256(source['checkpoint']) != source['checkpoint_sha256']:
            raise ValueError('Frozen auxiliary checkpoint changed')
        saved = torch.load(source['checkpoint'], map_location='cpu', weights_only=False)
        meta = saved['metadata']
        assert saved['config_sha256'] == repair_sha256
        assert saved['completed_epochs'] == source['epoch']
        assert (meta['cue'], meta['variant'], meta['seed']) == (source['cue'], source['variant'], 0)
        assert meta['source_RGB_sha256'] == source_RGB_sha256
        module = CorrelationRepair(radius=0 if source['variant'] == 'aligned' else 2)
        module.load_state_dict(saved['adapter'], strict=True)
        assert sum(p.numel() for p in module.parameters()) == 217619
        loaded[name] = module.to(device).float().eval().requires_grad_(False)
        information[name] = dict(row, training_metadata=meta)
    return loaded, information


def view_name(row):
    if row.get('baseline') or row.get('intervention') == 'unavailable':
        return None
    if row.get('intervention') == 'IR_image_spatial_roll':
        if row['source']['cue'] != 'ir':
            raise ValueError('IR-image intervention requires an IR case')
        return 'ir_roll'
    if row.get('intervention') != 'normal':
        raise ValueError('Unregistered CGR intervention')
    return row['source']['cue']


class CorrelationCaseDetector:
    """Independent B1--B4 samples with exactly the original CGR forward math."""
    def __init__(self, model, cases, adapters, input_hw, calibration):
        self.model, self.cases, self.adapters = model, list(cases), adapters
        self.input_hw, self.calibration = tuple(input_hw), calibration
        if not self.cases or len({r['id'] for r in self.cases}) != len(self.cases):
            raise ValueError('Nonempty unique frozen cases required')
        if set(adapters) != {r['id'] for r in self.cases}:
            raise ValueError('One loaded adapter entry per case required')
        self.views = {'rgb'} | {v for r in self.cases if (v := view_name(r)) is not None}
        self.needs_IR = bool(self.views & {'ir', 'ir_roll'})

    @torch.inference_mode()
    def batch(self, rgb_images, ir_images, canonical):
        count = len(rgb_images)
        if count not in [1, 2, 3, 4] or len(canonical) != count:
            raise ValueError('One original independent B4 batch including its short tail required')
        if self.needs_IR and (ir_images is None or len(ir_images) != count):
            raise ValueError('Actual current November IR images required')
        arrays, available, ratios = {v: [] for v in self.views}, {v: [] for v in self.views}, []
        for index, rgb in enumerate(rgb_images):
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError('Original decoded uint8 BGR image required')
            value, ratio = exact_half_preproc(rgb, self.input_hw, (.485, .456, .406), (.229, .224, .225))
            arrays['rgb'].append(value)
            available['rgb'].append(True)
            ratios.append(ratio)
            ir = ir_images[index] if self.needs_IR else None
            if self.needs_IR and (ir.dtype != np.uint8 or ir.ndim != 2 or ir.shape != rgb.shape[:2]):
                raise ValueError('Matched scalar November IR required')
            for view in self.views - {'rgb'}:
                image = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if view == 'gray' else ir
                if view == 'ir_roll':
                    image = np.roll(image, (image.shape[0] // 4, image.shape[1] // 4), (0, 1))
                norm = self.calibration['statistics']['ir' if view.startswith('ir') else 'gray']
                value, _ = auxiliary_input(image, self.input_hw, norm['mean'], norm['std'])
                arrays[view].append(value)
                available[view].append(bool(image.max() != image.min()))
        parameter = next(self.model.parameters())
        features = {view: self.model.backbone(torch.from_numpy(np.stack(values)).to(parameter))
                    for view, values in arrays.items()}
        teacher = self.model.head(features['rgb'])
        scores, offset = [], 0
        for value in features['rgb']:
            cells = value.shape[-2] * value.shape[-1]
            score = teacher[:, offset:offset+cells, 4] * teacher[:, offset:offset+cells, 5]
            scores.append(score.reshape(count, 1, *value.shape[-2:]))
            offset += cells
        fresh = [detections(v, ratio) for v, ratio in zip(postprocess(teacher.clone(), 1, .1, .7), ratios)]
        outputs = {}
        for row in self.cases:
            name = row['id']
            if row.get('baseline'):
                if row['convention'] not in {'fresh_B4', 'canonical_B4'}:
                    raise ValueError('Keep distinct registered RGB baseline conventions')
                values = fresh if row['convention'] == 'fresh_B4' else canonical
            else:
                view = view_name(row)
                flags = available[view] if view is not None else [False] * count
                valid = torch.tensor(flags, dtype=torch.bool, device=parameter.device)
                repaired = self.adapters[name](features['rgb'], features.get(view), scores, valid)
                prediction = self.model.head(repaired)
                values = [detections(v, ratio) for v, ratio in zip(postprocess(prediction, 1, .1, .7), ratios)]
                if view is None:
                    for current, baseline in zip(values, fresh):
                        np.testing.assert_array_equal(current, baseline)
                if row['source']['output_policy'] == 'priority':
                    values = [original.copy() if not ok else priority_union(original, [(value, 1.)])[0]
                              for original, value, ok in zip(canonical, values, flags)]
                elif row['source']['output_policy'] != 'raw':
                    raise ValueError('Only original raw/priority CGR policies allowed')
            outputs[name] = [np.asarray(v, dtype=np.float32).copy() for v in values]
        return outputs, fresh


class LegacyCaseTrackWriter(StreamTrackWriter):
    """Fresh video tracker state, continuing only each legacy case's ID counter.

    The old replay had one process per condition and a global BaseTrack counter
    across videos. Private caller-owned counters reproduce those literal IDs
    when several conditions share a producer. No emitted row is renumbered.
    """
    def __init__(self, directory, sequence, model_ids, case_counters):
        ids = list(model_ids)
        if set(ids) != set(case_counters) or any(
                not isinstance(v, list) or len(v) != 1 or type(v[0]) is not int or v[0] < 0
                for v in case_counters.values()):
            raise ValueError('One explicit nonnegative mutable counter per frozen case required')
        super().__init__(directory, sequence, ids)
        for name, tracker in self.trackers.items():
            tracker.counter = case_counters[name]


def input_paths(plan, split, sequence, frames, needs_IR):
    """No labels or future frames; caller supplies one ordered current batch."""
    if split not in {'dev', 'test'}:
        raise ValueError('Registered Dev/Test pixels only')
    rgb_paths = [Path(plan['data_root']) / ('train' if split == 'dev' else 'test') /
                 sequence / 'visible' / f'{frame:06d}.jpg' for frame in frames]
    ir_paths = ([Path(plan['generated_ir_root']) / split / sequence /
                 'infrared_diffv2ir' / f'{frame:06d}.png' for frame in frames] if needs_IR else [])
    return rgb_paths, ir_paths
