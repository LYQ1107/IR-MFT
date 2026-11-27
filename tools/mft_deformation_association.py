"""Fixed-factorial confidence and measured-scale association hypotheses.

The current detection supplies the comparison scale while the track supplies
its predicted center. Only matching costs change. Original detections, birth
thresholds, measurement updates and template reliability rules stay in force.
These are geometric tracking hypotheses, not new IR information.
"""
import numpy as np
from yolox.tracker import matching
from tools.mft_birth_only import BirthOnlyTeacher
from tools.mft_protected_peripheral import ProtectedStreams


def measured_scale_cost(tracks, detections):
    if not len(tracks) or not len(detections):
        return np.empty((len(tracks), len(detections)), np.float64)
    boxes = np.asarray([t.tlbr for t in tracks], np.float64)
    current = np.asarray([d.tlbr for d in detections], np.float64)
    sizes = current[:, 2:] - current[:, :2]
    assert (sizes > 0).all()
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    observed = (current[:, :2] + current[:, 2:]) / 2
    displacement = np.abs(centers[:, None] - observed[None])
    intersection = np.maximum(sizes[None] - displacement, 0).prod(axis=2)
    area = sizes.prod(axis=1)
    cost = 1 - intersection / (2 * area[None] - intersection)
    assert np.isfinite(cost).all() and ((cost >= 0) & (cost <= 1)).all()
    return cost


class AssociationTeacher(BirthOnlyTeacher):
    def __init__(self, speed_upper, case):
        super().__init__(speed_upper)
        self.case = dict(case)

    def reset(self, sequence):
        super().reset(sequence)
        self.core.args.mot20 = bool(self.case['without_score_fusion'])
        self.core.association_hook = self.condition
        self.statistics.update(measured_scale_cost_entries_changed=0, measured_scale_cost_stages=0)

    def condition(self, tracks, detections, base, frame, stage):
        original = matching.fuse_score(base, detections) if (
            self.case['without_score_fusion'] and stage in ['primary', 'pending']) else base
        if self.case['measured_scale']:
            cost = measured_scale_cost(tracks, detections)
            if not self.case['without_score_fusion'] and stage in ['primary', 'pending']:
                cost = matching.fuse_score(cost, detections)
            self.statistics['measured_scale_cost_entries_changed'] += int(np.count_nonzero(cost != base))
            self.statistics['measured_scale_cost_stages'] += 1
        else:
            cost = self.observe(tracks, detections, base, frame, stage)
        assert cost.shape == base.shape
        self.statistics['original_association_cost_entries_changed'] += int(np.count_nonzero(cost != original))
        return cost


class AssociationStreams(ProtectedStreams):
    def __init__(self, association_case, views, heads, thresholds, speed,
                 archive_models=None, archive_thresholds=None, *, standalone=False):
        super().__init__(views, heads, thresholds, speed, archive_models, archive_thresholds,
                         standalone=standalone)
        teacher = AssociationTeacher(speed, association_case)
        self.main.teacher = teacher
        if not standalone:
            self.main.parent.linker.inner = teacher


def policies():
    return [dict(id='DA_reference', measured_scale=False, without_score_fusion=False),
            dict(id='DA_without_score', measured_scale=False, without_score_fusion=True),
            dict(id='DA_measured_scale', measured_scale=True, without_score_fusion=False),
            dict(id='DA_both', measured_scale=True, without_score_fusion=True)]
