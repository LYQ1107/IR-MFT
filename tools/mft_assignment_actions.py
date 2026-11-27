"""GT-free local alternatives to a completed stage assignment.

Closed swaps and replacements of one unmatched endpoint retain cardinality.
The census observes these actions only; it never applies them to its teacher.
All features use current observations or earlier reliable templates.
"""
from dataclasses import dataclass
import numpy as np

from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker import matching
from tools.mft_candidate_risk_teacher import CandidateTeacher, CandidateTrack
from tools.mft_tracklet_permanence import unit

STAGES = ('primary', 'secondary', 'pending')
KINDS = ('closed_swap', 'detection_replacement', 'track_replacement')
THRESHOLDS = dict(primary=.9, secondary=.5, pending=.7)
FEATURE_NAMES = tuple(
    f'{view}_{field}' for view in ['RGB', 'IR', 'anchor_RGB', 'anchor_IR']
    for field in ['before', 'after', 'minimum_delta']
) + ('geometry_before', 'geometry_after', 'geometry_regret',
     'score_before', 'score_after', 'past_quality_before', 'past_quality_after',
     'current_quality_before', 'current_quality_after', 'log_gap_before', 'log_gap_after',
     'log_age_before', 'log_age_after', 'IR_available', 'anchor_IR_available',
     'RGB_variance_before', 'RGB_variance_after', 'IR_variance_before', 'IR_variance_after',
     'RGB_anchor_consistency_before', 'RGB_anchor_consistency_after',
     'IR_anchor_consistency_before', 'IR_anchor_consistency_after',
     *('kind_' + k for k in KINDS), *('stage_' + s for s in STAGES))
assert len(FEATURE_NAMES) == 41


@dataclass(frozen=True)
class Action:
    kind: str
    before: tuple
    after: tuple
    regret: float


def trusted_template(t):
    return (t.accepted >= 3 and t.rgb is not None and t.last_box is not None
            and getattr(t, 'anchor_rgb', None) is not None)


def alternatives(tracks, detections, base, stage, maximum_regret=.1):
    base = np.asarray(base, np.float64)
    assert base.shape == (len(tracks), len(detections)) and np.isfinite(base).all()
    threshold = THRESHOLDS[stage]
    pairs, free_rows, free_columns = matching.linear_assignment(base, thresh=threshold)
    pairs = tuple(sorted((int(i), int(j)) for i, j in pairs))
    actions = []
    for left, (i, a) in enumerate(pairs):
        if not trusted_template(tracks[i]):
            continue
        for j, b in pairs[left + 1:]:
            if not trusted_template(tracks[j]) or base[i, b] > threshold or base[j, a] > threshold:
                continue
            regret = float(base[i, b] + base[j, a] - base[i, a] - base[j, b])
            if regret <= maximum_regret + 1e-12:
                actions.append(Action(KINDS[0], ((i, a), (j, b)), ((i, b), (j, a)), regret))
        for b in free_columns:
            b = int(b); regret = float(base[i, b] - base[i, a])
            if base[i, b] <= threshold and regret <= maximum_regret + 1e-12:
                actions.append(Action(KINDS[1], ((i, a),), ((i, b),), regret))
        for j in free_rows:
            j = int(j); regret = float(base[j, a] - base[i, a])
            if trusted_template(tracks[j]) and base[j, a] <= threshold and regret <= maximum_regret + 1e-12:
                actions.append(Action(KINDS[2], ((i, a),), ((j, a),), regret))
    return pairs, tuple(actions)


def force_actions(base, stage, original, selected):
    """Apply disjoint accepted edits, keeping exactly the original cardinality."""
    if not selected:
        return base
    planned = set(original); used_rows = set(); used_columns = set()
    for action in selected:
        assert len(action.before) == len(action.after) and set(action.before) <= planned
        rows = {i for i, j in action.before + action.after}
        columns = {j for i, j in action.before + action.after}
        assert not rows & used_rows and not columns & used_columns
        assert all(base[i, j] <= THRESHOLDS[stage] for i, j in action.after)
        used_rows |= rows; used_columns |= columns
        planned.difference_update(action.before); planned.update(action.after)
    assert len(planned) == len(original)
    assert len({i for i, j in planned}) == len({j for i, j in planned}) == len(planned)
    result = np.full(np.shape(base), 2., np.float64)
    for i, j in planned:
        result[i, j] = base[i, j]
    observed = matching.linear_assignment(result, thresh=THRESHOLDS[stage])[0]
    assert set(map(tuple, observed.tolist())) == planned
    return result


class AnchorTrack(CandidateTrack):
    def __init__(self, *args):
        super().__init__(*args)
        self.anchor_rgb = self.anchor_ir = None
        self.anchor_count = 0; self.anchor_ir_complete = True
        self.anchor_rgb_sum = self.anchor_ir_sum = None
        self.rgb_variance = self.ir_variance = 0.

    def accept(self, obs):
        old_rgb = None if self.rgb is None else self.rgb.copy()
        old_ir = None if self.ir is None else self.ir.copy()
        super().accept(obs)
        if not obs['reliable']:
            return
        if old_rgb is not None:
            self.rgb_variance = .95*self.rgb_variance + .05*(1-float(np.clip(old_rgb@obs['rgb'], -1, 1)))
        if old_ir is not None and obs['ir_available']:
            self.ir_variance = .95*self.ir_variance + .05*(1-float(np.clip(old_ir@obs['ir'], -1, 1)))
        if self.anchor_count < 3:
            self.anchor_rgb_sum = obs['rgb'].copy() if self.anchor_rgb_sum is None else self.anchor_rgb_sum + obs['rgb']
            if obs['ir_available']:
                self.anchor_ir_sum = obs['ir'].copy() if self.anchor_ir_sum is None else self.anchor_ir_sum + obs['ir']
            else:
                self.anchor_ir_complete = False
            self.anchor_count += 1
            if self.anchor_count == 3:
                self.anchor_rgb = unit(self.anchor_rgb_sum)
                self.anchor_ir = unit(self.anchor_ir_sum) if self.anchor_ir_complete else None
                self.anchor_rgb_sum = self.anchor_ir_sum = None


def action_features(tracks, detections, base, frame, stage, action):
    assert action.kind in KINDS and stage in STAGES
    before, after = action.before, action.after
    all_edges = before + after
    ir_ok = all(tracks[i].ir is not None and detections[j].observation['ir_available'] for i, j in all_edges)
    air_ok = all(tracks[i].anchor_ir is not None and detections[j].observation['ir_available'] for i, j in all_edges)
    values = []
    for view in ['RGB', 'IR', 'anchor_RGB', 'anchor_IR']:
        effective = 'RGB' if view == 'IR' and not ir_ok else 'anchor_RGB' if view == 'anchor_IR' and not air_ok else view
        def scores(edges):
            result = []
            for i, j in edges:
                t, obs = tracks[i], detections[j].observation
                vector = t.rgb if effective == 'RGB' else t.ir if effective == 'IR' else t.anchor_rgb if effective == 'anchor_RGB' else t.anchor_ir
                current = obs['ir'] if effective in ['IR', 'anchor_IR'] else obs['rgb']
                assert vector is not None and current is not None
                result.append(float(np.clip(vector@current, -1, 1)))
            return np.asarray(result, np.float64)
        old, new = scores(before), scores(after)
        values.extend([old.mean(), new.mean(), (new-old).min()])
    mean = lambda edges, fn: float(np.mean([fn(tracks[i], detections[j]) for i, j in edges]))
    values.extend([float(np.mean([base[i, j] for i, j in before])), float(np.mean([base[i, j] for i, j in after])), action.regret])
    for fn in [lambda t,d:d.score, lambda t,d:t.memory_quality, lambda t,d:d.observation['quality'],
               lambda t,d:np.log1p(frame-t.frame_id), lambda t,d:np.log1p(frame-t.start_frame)]:
        values.extend([mean(before,fn),mean(after,fn)])
    values.extend([float(ir_ok), float(air_ok)])
    for fn in [lambda t,d:t.rgb_variance, lambda t,d:t.ir_variance if ir_ok else 0.,
               lambda t,d:float(np.clip(t.rgb@t.anchor_rgb,-1,1)),
               lambda t,d:float(np.clip(t.ir@t.anchor_ir,-1,1)) if ir_ok and air_ok else 0.]:
        values.extend([mean(before,fn),mean(after,fn)])
    values.extend(float(action.kind == k) for k in KINDS)
    values.extend(float(stage == s) for s in STAGES)
    result = np.asarray(values, np.float32)
    assert result.shape == (len(FEATURE_NAMES),) and np.isfinite(result).all()
    return result


def view_features(x, kind):
    x = np.asarray(x).copy()
    assert x.shape[-1] == len(FEATURE_NAMES)
    if kind == 'paired':
        return x
    assert kind in ['RGB', 'duplicatedRGB']
    for prefix in ['IR', 'anchor_IR']:
        rgb = prefix.replace('IR', 'RGB')
        for field in ['before', 'after', 'minimum_delta']:
            x[..., FEATURE_NAMES.index(prefix+'_'+field)] = 0 if kind == 'RGB' else x[..., FEATURE_NAMES.index(rgb+'_'+field)]
    for suffix in ['variance_before','variance_after','anchor_consistency_before','anchor_consistency_after']:
        x[..., FEATURE_NAMES.index('IR_'+suffix)] = 0 if kind == 'RGB' else x[..., FEATURE_NAMES.index('RGB_'+suffix)]
    for name in ['IR_available','anchor_IR_available']:
        x[..., FEATURE_NAMES.index(name)] = float(kind == 'duplicatedRGB')
    return x


class ActionTeacher(CandidateTeacher):
    def __init__(self, speed_upper, sink, maximum_regret=.1):
        super().__init__(speed_upper)
        self.sink = sink; self.maximum_regret = maximum_regret

    def reset(self, sequence):
        super().reset(sequence); counter = [0]
        self.core = BYTETracker(self.core.args, frame_rate=25,
            track_factory=lambda b,s,o:AnchorTrack(b,s,o,counter), association_hook=self.observe)

    def observe(self, tracks, detections, base, frame, stage):
        saved = np.asarray(base).copy()
        pairs, actions = alternatives(tracks,detections,base,stage,self.maximum_regret)
        matches = [(int(tracks[i].track_id),int(detections[j].observation['proposal_index']),int(trusted_template(tracks[i]))) for i,j in pairs]
        records = []
        for action in actions:
            assert all(tracks[i].anchor_rgb is not None for i,j in action.before+action.after)
            remap = lambda edges:tuple((int(tracks[i].track_id),int(detections[j].observation['proposal_index']))for i,j in edges)
            records.append(dict(kind=action.kind,before=remap(action.before),after=remap(action.after),
                regret=action.regret,features=action_features(tracks,detections,base,frame,stage,action)))
        self.sink(frame,stage,matches,records)
        np.testing.assert_array_equal(base,saved)
        return base

    def update(self, sequence, frame, detections, rgb, ir=None):
        result = super().update(sequence,frame,detections,rgb,ir)
        live = {id(t)for t in self.core.tracked_stracks+self.core.lost_stracks}
        for t in self.core.removed_stracks:
            if id(t) not in live:
                t.anchor_rgb=t.anchor_ir=t.anchor_rgb_sum=t.anchor_ir_sum=None
        return result
