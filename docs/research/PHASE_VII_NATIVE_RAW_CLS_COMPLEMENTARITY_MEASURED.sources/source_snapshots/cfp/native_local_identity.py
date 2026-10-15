"""Actual native identity memory, never clean GT groups or offline teacher."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .predictive import checked_history
from yolox.tracker import matching


class RecentIdentityControl(nn.Module):
    def __init__(self, mode='fifo', length=4, maximum_age=1500):
        super().__init__()
        if mode not in ('last', 'fifo', 'ema') or not 1<=length<=8:
            raise ValueError('Registered same-history untrained current-coordinate control required')
        self.mode, self.length, self.maximum_age = mode, length, maximum_age

    def forward(self, features, times, valid, frame):
        source, _, _, available, _ = checked_history(features, times, valid, frame)
        output = source.new_zeros(len(source), source.shape[-1])
        observed = torch.zeros_like(available)
        for row in range(len(source)):
            indices = torch.nonzero(valid[row] & (times[row]>=frame[row]-self.maximum_age), as_tuple=False).flatten()[-self.length:]
            if not len(indices):
                continue
            values = source[row, indices]
            if self.mode=='last':
                result = values[-1]
            elif self.mode=='fifo':
                result = values.sum(0)
            else:
                result = values[0]
                for value in values[1:]:
                    result = F.normalize(.9*result+.1*value, dim=-1)
            output[row] = F.normalize(result, dim=-1)
            observed[row] = True
        return output, observed


def actual_model_inputs(tracks, detections, frame, bank, length=4, maximum_age=1500):
    """Only native assigned proposal references, with exact past/source guards."""
    n, m = len(tracks), len(detections)
    count = bank['local4'].shape[1]
    dimension = bank['raw_cls'].shape[1]
    history = np.zeros((n, length, count, dimension), np.float32)
    fractions = np.zeros((n, length, count), np.float32)
    global_values = np.zeros((n, length, dimension), np.float32)
    times, valid = np.zeros((n, length), np.int64), np.zeros((n, length), bool)
    for i, track in enumerate(tracks):
        rows = list(track.identity_proposals)
        if any(time>=frame for time, index in rows):
            raise ValueError('Current/future identity writes must not enter native scoring')
        rows = [(t, index) for t,index in rows if frame-maximum_age<=t][-length:]
        for j, (time, index) in enumerate(rows, start=length-len(rows)):
            if not 0<=index<len(bank['raw_cls']) or bank['frame_of_proposal'][index]!=time:
                raise ValueError('Native assigned history reference lost exact timestamp/source row')
            if not bank['valid'][index]:
                raise ValueError('An unobservable proposal must never be written into native identity memory')
            history[i,j], fractions[i,j], global_values[i,j] = bank['local4'][index], bank['visibility4'][index], bank['raw_cls'][index]
            times[i,j], valid[i,j] = time, True
    current_indices = np.array([d.current_proposal_index for d in detections], dtype=np.int64)
    if np.any(current_indices<0) or np.any(current_indices>=len(bank['raw_cls'])) or (m and not np.all(bank['frame_of_proposal'][current_indices]==frame)):
        raise ValueError('Current actual proposals must match the current frame exactly')
    result = dict(history=history, history_fraction=fractions, history_global=global_values,
        times=times, history_valid=valid, candidates=np.array(bank['local4'][current_indices], np.float32),
        candidate_fraction=np.array(bank['visibility4'][current_indices], np.float32),
        candidate_global=np.array(bank['raw_cls'][current_indices], np.float32))
    return dict({k:torch.from_numpy(v) for k,v in result.items()}, frame=int(frame))


def install_native_local_identity(tracker, model, bank, length=4, maximum_age=1500, context_identity=False):
    """Keep the native lifecycle/graph; change only confirmed high-stage cues."""
    model.eval()
    base = type(tracker)

    class LocalIdentityTracker(base):
        def association_cost(self, tracks, detections):
            baseline = super().association_cost(tracks, detections)
            if not baseline.size or any(not t.is_activated for t in tracks):
                return baseline
            inputs = actual_model_inputs(tracks, detections, self.frame_id, self.local_feature_bank,
                                         self.local_history_length, self.local_maximum_age)
            with torch.inference_mode():
                output = self.local_model.online(**inputs) if self.context_identity else self.local_model(**inputs)
                scale = float(self.local_model.logit_scale.clamp(np.log(1.), np.log(50.)).exp())
                # Explicit inverse of the model's common pair-logit scale,
                # preserving its small learned residual in cosine coordinates.
                similarities = (output['scores']/scale).clamp(-1,1).numpy()
            raw = matching.iou_distance(tracks, detections)
            geometry = matching.fuse_score(raw, detections) if not self.args.mot20 else raw
            result = baseline.copy()
            candidate_available = output['current']['valid'].any(1).numpy()
            for row in range(len(tracks)):
                if not output['history_available'][row]:
                    continue
                legal = candidate_available & (raw[row]<=1-self.proximity_iou)
                if not np.isfinite(similarities[row, legal]).all():
                    raise ValueError('Nonfinite legal native identity score')
                appearance = np.clip((1-similarities[row])/2,0,1)
                result[row,legal] = (1-self.appearance_weight)*geometry[row,legal]+self.appearance_weight*appearance[legal]
                self.local_counts['scored_rows'] += 1
                self.local_counts['eligible_edges'] += int(legal.sum())
            self.local_counts['scored_frames'] += 1
            return result

    tracker.__class__ = LocalIdentityTracker
    tracker.local_model, tracker.local_feature_bank = model, bank
    tracker.local_history_length, tracker.local_maximum_age = length, maximum_age
    tracker.context_identity = context_identity
    tracker.local_counts = dict(scored_frames=0,scored_rows=0,eligible_edges=0)
    return tracker
