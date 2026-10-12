"""Explicit feature, identity-memory and lifecycle adapters for a 2x2 experiment.

Motion/output state remains owned by each lifecycle. Association owns its cost
formula, memory representation and calibrated first-stage acceptance threshold.
Native reference files are verified and never modified.
"""
from collections import deque
from contextlib import contextmanager
import gzip
import importlib
import json
from pathlib import Path
import sys
import subprocess
from types import ModuleType, SimpleNamespace

import numpy as np
from scipy.spatial.distance import cdist

from .association import IdentityTrack, optional_feature
from .bot_lifecycle import bot_lifecycle_update
from .predictive_association import PredictiveAssociationTracker
from .provenance import sha256
from .reliable_writes import ReliableWritesMixin
from .history import early_recent_history
from yolox.tracker import matching as cfp_matching


BOT_COMMIT = '251985436d6712aaf682aaaf5f71edb4987224bd'
BOT_SOURCE_SHA = 'ba6dd32ad4b96ee49d6f70654159269bd45ce15fd0fe5007f3a1b81bc257edf0'


class FrameFeatures:
    def __init__(self, *_):
        self.boxes = self.features = None
        self.entries = {}

    def set_frame(self, detections, features, indices, high=.58, low=.1):
        selected = (detections[:, 4] > high) & (detections[:, 4] > low)
        self.boxes, self.features = detections[selected, :4].copy(), features[selected].copy()
        self.entries = {}
        for box, feature, index in zip(detections, features, indices):
            tlwh = box[:4].copy()
            tlwh[2:] -= tlwh[:2]
            key = tuple(tlwh.tolist()) + (float(box[4]),)
            if key in self.entries:
                raise ValueError('Ambiguous duplicate proposal mapping')
            self.entries[key] = (optional_feature(feature), int(index))

    def lookup(self, tlwh, score):
        return self.entries[tuple(np.asarray(tlwh).tolist()) + (float(score),)]

    def inference(self, image, boxes):
        if not np.array_equal(np.asarray(boxes).reshape(-1, 4), self.boxes):
            raise ValueError('Native feature request lost exact proposal order')
        return self.features.copy()


def load_bot_reference(root):
    root = Path(root).resolve()
    if subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() != BOT_COMMIT:
        raise ValueError('Pinned BoT reference commit changed')
    expected = {
        'tracker/basetrack.py': 'a0f717a5059ebfa626a63cea5b040a649acd7720072f503cabfa24d749f082f6',
        'tracker/gmc.py': '508c3a4ea0e9a72478ce2d1d29aa1eb8892b3804c76b524a16a34dca23dba9ba',
        'tracker/kalman_filter.py': 'c1a4c0acc71b9a91c0cdbe7b689f88712d68e7e8be4d7e54fb437f33fdbc9e5a',
        'tracker/matching.py': '020b2215c20467b31f7d8a977a2f557b8d1838f8baa5a5be54c0b8d14bbee4e1',
    }
    if any(sha256(root / name) != digest for name, digest in expected.items()):
        raise ValueError('Pinned motion/assignment/lifecycle dependency changed')
    if sha256(root / 'tracker/bot_sort.py') != BOT_SOURCE_SHA:
        raise ValueError('Pinned BoT source changed')
    for alias, value in [('float', float), ('int', int), ('bool', bool)]:
        if alias not in np.__dict__:
            setattr(np, alias, value)
    shim = ModuleType('fast_reid.fast_reid_interfece')
    shim.FastReIDInterface = FrameFeatures
    sys.modules['fast_reid.fast_reid_interfece'] = shim
    sys.path.insert(0, str(root))
    module = importlib.import_module('tracker.bot_sort')
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise ValueError('Wrong tracker module imported')
    return module


def bot_cost(tracker, tracks, detections, native, appearance_threshold):
    raw = native.matching.iou_distance(tracks, detections)
    geometry = native.matching.fuse_score(raw, detections)
    if not raw.size:
        return geometry
    memories = [getattr(track, 'smooth_feat', getattr(track, 'bot_memory', None)) for track in tracks]
    candidates = [getattr(det, 'curr_feat', det.current_rgb) for det in detections]
    if any(value is None for value in memories + candidates):
        raise ValueError('BoT high-stage memory mapping incomplete')
    appearance = np.maximum(0., cdist(np.asarray(memories, dtype=float), np.asarray(candidates, dtype=float), 'cosine')) / 2.
    appearance[appearance > appearance_threshold] = 1.
    appearance[raw > .5] = 1.
    return np.minimum(geometry, appearance)


def new_identity_fields(track, feature, index):
    track.current_rgb, track.current_ir = feature, None
    track.current_proposal_index = index
    track.memory_rgb = track.memory_ir = None
    track.pending = True
    track.last_identity_frame = 0
    track.identity_history, track.identity_proposals = deque(maxlen=8), deque(maxlen=8)
    track.initial_identity_frame = 0
    track.initial_identity_rgb = track.initial_identity_ir = None
    track.initial_proposal_index = None


def make_bot_lifecycle(native, parameters, association, predictor=None, weight=.3, appearance_threshold=.15):
    class MappedTrack(native.STrack):
        def __init__(self, tlwh, score, feat=None, feat_history=50):
            super().__init__(tlwh, score, feat, feat_history)
            feature, index = provider.lookup(tlwh, score)
            new_identity_fields(self, feature, index)

        write_identity = IdentityTrack.write_identity

        def update(self, new_track, frame_id):
            super().update(new_track, frame_id)
            self.current_rgb, self.current_proposal_index = new_track.current_rgb, new_track.current_proposal_index
            self.pending = True

        def re_activate(self, new_track, frame_id, new_id=False):
            super().re_activate(new_track, frame_id, new_id)
            self.current_rgb, self.current_proposal_index = new_track.current_rgb, new_track.current_proposal_index
            self.pending = True

    class Tracker(native.BoTSORT):
        def __init__(self):
            values = dict(parameters, appearance_thresh=appearance_threshold,
                          match_thresh=.8 if association == 'bot' else .97)
            super().__init__(SimpleNamespace(**values), frame_rate=25)
            self.track_type, self.native_helpers, self.native_matching, self.track_states = MappedTrack, native, native.matching, native.TrackState
            self.appearance_weight, self.proximity_iou, self.ema = weight, .1, .9
            self.predictor, self.memory_mode, self.history_view = predictor, 'predictor', 'anchor_recent'
            self.prediction_counts = dict(predicted_rows=0, missing_history_rows=0, modified_edges=0)
            self.last_output = []

        def association_cost(self, tracks, detections):
            if association == 'bot':
                return bot_cost(self, tracks, detections, native, appearance_threshold)
            return PredictiveAssociationTracker.association_cost(self, tracks, detections)

        def finalize_identity(self):
            for track in self.tracked_stracks:
                if track.pending:
                    track.write_identity(self.frame_id, .9)
            # Same identity retirement boundary as the original CFP adapter;
            # native removed/lost object overlap remains untouched.
            for track in self.removed_stracks:
                if track.state == self.track_states.Removed:
                    track.identity_history.clear()
                    track.identity_proposals.clear()
                    track.memory_rgb = None
                if all(track is not live for live in self.tracked_stracks + self.lost_stracks):
                    track.initial_identity_rgb = None
                    track.initial_proposal_index = None

        def update(self, detections, features, indices):
            provider.set_frame(detections, features, indices)
            self.encoder = provider
            self.last_output = bot_lifecycle_update(self, detections.copy(), None)
            return self.last_output

    provider = FrameFeatures()
    return Tracker()


def make_cfp_lifecycle(native, association, predictor=None, weight=.3, appearance_threshold=.15, write_policy=None, soft_write=False, memory_mode=None, read_view='anchor_recent'):
    if read_view not in ('anchor_recent', 'early3_recent5'):
        raise ValueError('Explicit frozen read view required')
    class Tracker(ReliableWritesMixin, PredictiveAssociationTracker):
        def association_cost(self, tracks, detections):
            if soft_write:
                for track in tracks:
                    if getattr(track, '_phase_v_soft_write', False):
                        continue
                    original = track.write_identity
                    def write(frame, ema, original=original, track=track):
                        # Observed current matched confidence, after scoring;
                        # every accepted cue still writes, with bounded weight.
                        return original(frame, .9 + .09 * (1 - float(track.score)))
                    track.write_identity = write
                    track._phase_v_soft_write = True
            if association == 'bot':
                return bot_cost(self, tracks, detections, native, appearance_threshold)
            if read_view == 'anchor_recent':
                return super().association_cost(tracks, detections)
            saved = []
            for track in tracks:
                saved.append((track, track.identity_history, track.initial_identity_rgb))
                selected = early_recent_history(track.identity_history, getattr(track, 'early_identity_history', []))
                track.identity_history = deque(selected, maxlen=8)
                track.initial_identity_rgb = None  # Input already contains its independent early cues.
            try:
                return super().association_cost(tracks, detections)
            finally:
                for track, history, initial in saved:
                    track.identity_history, track.initial_identity_rgb = history, initial

        def update(self, detections, features, indices):
            output = super().update(detections.copy(), (1, 1), (1, 1), features, proposal_indices=indices)
            if read_view == 'early3_recent5':
                for track in self.tracked_stracks:
                    if not hasattr(track, 'early_identity_history'):
                        track.early_identity_history = []
                    times = {row[0] for row in track.early_identity_history}
                    for row in track.identity_history:
                        if row[0] not in times and len(track.early_identity_history) < 3:
                            track.early_identity_history.append(row)
                            times.add(row[0])
                live = {id(track) for track in self.tracked_stracks + self.lost_stracks}
                for track in self.removed_stracks:
                    if id(track) not in live:
                        track.early_identity_history = []
            if association == 'bot':
                for track in self.tracked_stracks:
                    # Native BoT updates its appearance EMA on high-confidence
                    # matches only; low-confidence geometry updates do not write.
                    if track.frame_id != self.frame_id or track.score <= .58:
                        continue
                    value = track.current_rgb.copy()
                    if hasattr(track, 'bot_memory'):
                        value /= np.linalg.norm(value)  # Native existing-track second normalization.
                        track.bot_memory = .9 * track.bot_memory + .1 * value
                        track.bot_memory /= np.linalg.norm(track.bot_memory)
                    else:
                        track.bot_memory = value
            return output

    args = SimpleNamespace(track_thresh=.58, track_buffer=45, match_thresh=.8 if association == 'bot' else .97, mot20=False)
    return Tracker(args, predictor=predictor, memory=memory_mode or ('ema' if association == 'bot' else 'predictor'),
                   history_view='anchor_recent', appearance_weight=weight, proximity_iou=.1, ema=.9, write_policy=write_policy)


class DecisionRecorder:
    """Observe actual online matrices/assignments; no labels or counterfactual writes."""
    def __init__(self, path, tracker):
        self.stream = gzip.open(path, 'wt', compresslevel=3)
        self.tracker = tracker
        self.pair = None
        self.stage = 0

    def close(self):
        self.stream.close()

    @contextmanager
    def capture(self, modules):
        self.stage = 0
        originals = []
        for module in dict.fromkeys(modules):
            original_iou, original_assignment = module.iou_distance, module.linear_assignment

            def iou(tracks, detections, original=original_iou):
                result = original(tracks, detections)
                self.pair = (tracks, detections, result.copy())
                return result

            def assignment(cost, thresh, original=original_assignment):
                result = original(cost, thresh)
                self.stage += 1
                tracks, detections, raw = self.pair
                if cost.shape != (len(tracks), len(detections)):
                    raise ValueError('Matrix observation lost current decision correspondence')
                histories = []
                for track in tracks:
                    histories.append(dict(track_id=int(track.track_id), state=int(track.state), confirmed=bool(track.is_activated),
                                          start_frame=int(track.start_frame), end_frame=int(track.end_frame),
                                          last_identity_frame=int(track.last_identity_frame), initial_frame=int(track.initial_identity_frame),
                                          initial_proposal=track.initial_proposal_index, history=list(track.identity_proposals), box=track.tlbr.tolist()))
                matches = np.asarray(result[0]).reshape(-1, 2)
                row = dict(frame=int(self.tracker.frame_id), stage=['high', 'low', 'unconfirmed'][self.stage - 1],
                           tracks=histories, proposals=[int(det.current_proposal_index) for det in detections],
                           boxes=[det.tlbr.tolist() for det in detections], scores=[float(det.score) for det in detections],
                           geometry=raw.tolist(), costs=cost.tolist(), threshold=float(thresh), matches=matches.tolist(),
                           unmatched_tracks=list(map(int, result[1])), unmatched_detections=list(map(int, result[2])))
                self.stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
                return result

            module.iou_distance, module.linear_assignment = iou, assignment
            originals.append((module, original_iou, original_assignment))
        try:
            yield
        finally:
            for module, original_iou, original_assignment in originals:
                module.iou_distance, module.linear_assignment = original_iou, original_assignment

