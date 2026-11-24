"""In-memory online F4 composition; the caller must qualify/release real models.

No file/checkpoint/image loader, fit, selection, evaluation or GPU authorization.
Formal producers must supply verified Dev-frozen models and source bytes. The
engineering tests use an artificial teacher and untrained visual modules.
"""
import copy
import hashlib

import numpy as np
import torch

from tools.fusion_memory_capture import FusionFeatureCapture
from tools.fusion_memory_design import make_case
from tools.native_pyramid_online_inference import OnlineNativeDetector
from yolox.tracker.fusion_memory_tracker import FusionMemoryConfig, FusionMemoryTracker


def visual_routes(cases):
    """Deduplicate exact visual IDs/metadata, never scores or network tensors."""
    routes, identities, case_routes = {}, {}, {}
    for case in cases:
        expected = make_case(case['visual'], case['profile'], case['appearance_weight'],
            floor=case['minimum_reliability'], intervention=case['intervention'])
        if case != expected or case['id'] in case_routes:
            raise ValueError('Unique complete registered memory cases required')
        visual = case['visual']
        if visual['model']['cue'] != visual['cue'] or visual['model']['id'] != visual['id']:
            raise ValueError('Visual object and original model metadata disagree')
        if visual.get('baseline') and case['profile'] not in ['byte', 'rgb_uniform']:
            raise ValueError('Direct RGB has no fused or attention-proxy memory')
        key = case['visual_key']
        if key in identities and identities[key] != visual:
            raise ValueError('One visual identity cannot name different snapshots')
        identities[key] = copy.deepcopy(visual)
        route_key = key+':'+case['intervention']
        identifier = 'FV_'+hashlib.sha256(route_key.encode()).hexdigest()[:24]
        row = copy.deepcopy(visual['model'])
        row.update(id=identifier, intervention={'IR_unavailable': 'unavailable'}.get(
            case['intervention'], case['intervention']))
        route = dict(id=identifier, visual_key=key, visual=copy.deepcopy(visual), model=row,
                     intervention=case['intervention'])
        if identifier in routes and routes[identifier] != route:
            raise ValueError('Route identity collision or incompatible model reuse')
        routes[identifier], case_routes[case['id']] = route, identifier
    if not routes:
        raise ValueError('At least one frozen memory case required')
    return routes, case_routes


def discard_consumed_observations(tracker, cleared_removed_prefix=0):
    """The current candidate was consumed by activate/update/re_activate.

    Future costs read fresh detection.observation and retained EMA memories.
    The old track.observation is never used again, including on reactivation.
    ByteTrack retains removed objects; do not retain their initial ROI vectors.
    """
    removed = tracker.core.removed_stracks
    if not 0 <= cleared_removed_prefix <= len(removed):
        raise ValueError('Consumed removed prefix belongs to another tracker state')
    for group in [tracker.core.tracked_stracks, tracker.core.lost_stracks, removed[cleared_removed_prefix:]]:
        for track in group:
            track.observation = None
    return len(removed)


class OnlineFusionMemory:
    def __init__(self, teacher, cases, modules, input_hw, calibration):
        self.cases = copy.deepcopy(list(cases))
        self.routes, self.case_routes = visual_routes(self.cases)
        if set(modules) != set(self.routes):
            raise ValueError('Each visual/intervention route needs an explicit isolated module entry')
        actual = [module for module in modules.values() if module is not None]
        if len({id(module) for module in actual}) != len(actual):
            raise ValueError('Different visual/intervention routes cannot alias a module with capture hooks')
        self.detector = OnlineNativeDetector(teacher, [r['model'] for r in self.routes.values()],
            modules, input_hw, calibration)
        self.input_hw = tuple(input_hw)
        self.sequence, self.last_frame, self.inflight = None, 0, None
        self.closed = self.failed = False
        self.captures, self.case_captures, self.trackers, self.handles = {}, {}, {}, []
        self.cleared_removed = {c['id']: 0 for c in self.cases}
        try:
            for case in self.cases:
                route = self.case_routes[case['id']]
                if case['appearance_weight']:
                    key = route, case['feature_source'], case['quality_policy']
                    if key not in self.captures:
                        self.captures[key] = FusionFeatureCapture(modules[route],
                            feature_source=case['feature_source'], quality_policy=case['quality_policy'])
                    self.case_captures[case['id']] = key
                self.trackers[case['id']] = FusionMemoryTracker(FusionMemoryConfig(
                    appearance_weight=case['appearance_weight'], feature_source=case['feature_source'] or 'rgb',
                    quality_mode='provided' if case['quality_policy'] == 'attention_proxy' else 'uniform',
                    minimum_reliability=case['minimum_reliability']))
            direct = [cap for (route, _, _), cap in self.captures.items() if modules[route] is None]
            if direct:
                def source_hook(module, args, output):
                    if self.inflight is None or len(output) != 3:
                        raise ValueError('One armed current RGB backbone pyramid required')
                    for capture in direct:
                        capture.capture_rgb_only(*self.inflight, output[0])
                self.handles.append(teacher.backbone.register_forward_hook(source_hook))
        except Exception:
            self.close()
            raise

    @property
    def needs_actual_IR_image(self):
        return self.detector.needs_actual_IR_image

    @torch.no_grad()
    def step(self, rgb, ir, sequence, frame):
        if self.closed or self.failed:
            raise ValueError('Closed/failed online state cannot be reused')
        if (not sequence or not isinstance(frame, int) or frame < 1 or
                (sequence != self.sequence and frame != 1) or
                (sequence == self.sequence and frame != self.last_frame+1)):
            raise ValueError('Every consecutive frame from1 with explicit sequence identity required')
        if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError('One decoded current uint8 RGB frame required')
        original_hw = tuple(int(v) for v in rgb.shape[:2])
        ratio = min(self.input_hw[0]/original_hw[0], self.input_hw[1]/original_hw[1])
        try:
            for capture in self.captures.values():
                capture.begin_frame(sequence, frame, original_hw, self.input_hw, ratio)
            self.inflight = sequence, frame
            detections, visual_info = self.detector.step(rgb, ir, sequence, frame)
            self.inflight = None
            packets = {key: cap.take_packet(sequence, frame, detections[key[0]])
                       for key, cap in self.captures.items()}
            rows, current = {}, {}
            for case in self.cases:
                name = case['id']
                observed = detections[self.case_routes[name]]
                key = self.case_captures.get(name)
                packet = None if key is None else packets[key]
                tracker = self.trackers[name]
                tracks = tracker.update(sequence, frame, observed, packet)
                rows[name] = [(frame, track.tlwh.copy(), track.track_id, track.score) for track in tracks
                              if track.tlwh[2]*track.tlwh[3] > 100]
                current[name] = observed  # Source packet unchanged; caller may hash/write it.
                prefix = self.cleared_removed[name] if self.sequence == sequence else 0
                self.cleared_removed[name] = discard_consumed_observations(tracker, prefix)
            self.sequence, self.last_frame = sequence, frame
            return rows, current, dict(visual_info, memory_cases=len(self.cases),
                unique_visual_routes=len(self.routes), unique_capture_profiles=len(self.captures),
                zero_weight_feature_reads=False, consumed_observations_released=True,
                feature_or_detection_disk_cache=False, visual_parent_qualification_established_here=False)
        except Exception:
            self.inflight, self.failed = None, True
            for capture in self.captures.values():
                capture.abort_frame()
            raise

    def close(self):
        for capture in self.captures.values():
            capture.close()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.inflight, self.closed = None, True
