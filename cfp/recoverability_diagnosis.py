"""Retrospective candidate/past-cue diagnosis. Never import in online inference.

Labels are independent same-frame physical annotation matches, not CLEAR's
continuity-adjusted matches. An early multi-observation reference is a diagnostic
identity convention, not an inference input or unconditional identity truth.
"""
from collections import Counter

import numpy as np

from .recovery_supervision import history_view, modal_identity


class EarlyReference:
    """Freeze a consensus from >=3 distinct known observations among first five.

    Only strictly past accepted observations are consulted. Unknown observations
    remain unknown; the first accepted token is not privileged as ground truth.
    """
    def __init__(self):
        self.observations = {}
        self.references = {}

    def update(self, track, labels):
        key = int(track['track_id'])
        observed = self.observations.setdefault(key, {})
        for frame, index in history_view(track):
            if labels[index] >= 0 and len(observed) < 5:
                observed.setdefault(index, frame)
        if key not in self.references and len(observed) >= 3:
            indices = sorted(observed, key=lambda index: observed[index])
            identity, consensus = modal_identity(indices, labels)
            if identity is not None:
                self.references[key] = (int(identity), float(consensus))
        return self.references.get(key, (None, None))[0]


def assess_candidate_row(row, position, bank, features, target, association='gru', topks=(1, 2, 3, 5)):
    """Return local opportunities, never assert global/HOTA oracle improvements.

    Geometry eligibility means appearance blending, not a hard association mask:
    outside it the original GRU may still match by geometry alone. Cost threshold
    feasibility is recorded separately, as is competition for a correct column.
    """
    if target is None:
        return dict(reference_known=False, state='unknown_reference')
    track = row['tracks'][position]
    history = history_view(track)
    frame = int(row['frame'])
    if any(t >= frame or bank['frame_of_proposal'][index] != t for t, index in history):
        raise ValueError('Diagnostic history violates actual past timestamp')
    labels = bank['labels']
    proposals = np.asarray(row['proposals'], dtype=np.int64)
    n, m = len(row['tracks']), len(proposals)
    cost = np.asarray(row['costs'], float).reshape(n, m)[position]
    geometry = np.asarray(row['geometry'], float).reshape(n, m)[position]
    matching = {int(a): int(b) for a, b in row['matches']}
    chosen = matching.get(position)
    current_labels = labels[proposals]
    correct = np.flatnonzero(current_labels == target)
    lo, hi = bank['offsets'][frame - 1:frame + 1]
    full = np.arange(lo, hi, dtype=np.int64)
    full_correct = np.flatnonzero(labels[full] == target)
    visible = target in {identity for identity, _ in bank['gt'][frame]}
    past_labels = labels[[index for _, index in history]]
    known = past_labels >= 0
    cues = past_labels == target
    prototype = None
    if history:
        prototype = features[[index for _, index in history]].mean(0)
        prototype = prototype / max(float(np.linalg.norm(prototype)), 1e-8)
    similarities = features[full] @ prototype if prototype is not None else np.zeros(len(full))
    ranking = np.argsort(-similarities, kind='stable')
    topk_recall = {str(k): bool(np.any(labels[full[ranking[:k]]] == target)) for k in topks}
    gate = .9 if association == 'gru' else .5
    full_boxes = bank['boxes'][full]
    predicted = np.asarray(track['box'], dtype=float)
    extent = np.maximum(np.minimum(predicted[2:], full_boxes[:, 2:]) - np.maximum(predicted[:2], full_boxes[:, :2]), 0.)
    overlap = extent.prod(1)
    union = np.prod(predicted[2:] - predicted[:2]) + (full_boxes[:, 2:] - full_boxes[:, :2]).prod(1) - overlap
    full_geometry = 1 - overlap / np.maximum(union, 1e-8)
    union_indices = set(np.flatnonzero(full_geometry <= gate).tolist()) | set(ranking[:3].tolist())
    geom_present = bool(np.any(geometry[correct] <= gate))
    feasible_present = bool(np.any(cost[correct] <= row['threshold']))
    assigned_identity = int(current_labels[chosen]) if chosen is not None else None
    assignment_known = chosen is not None and assigned_identity >= 0
    wrong = assignment_known and assigned_identity != target
    good = assignment_known and assigned_identity == target
    unknown_write = chosen is not None and not assignment_known
    correct_taken = any(j in correct and i != position for i, j in matching.items())
    cue_state = 'known_correct_cue' if cues.any() else ('all_known_no_correct_cue' if len(known) and known.all() else 'unverified_cue_availability')
    margin = None
    if chosen is not None:
        alternative = np.delete(cost, chosen)
        margin = min(float(alternative.min()) if len(alternative) else float(row['threshold']), float(row['threshold'])) - float(cost[chosen])
    ordered_cost = np.sort(cost)
    top12 = float(ordered_cost[1] - ordered_cost[0]) if len(ordered_cost) > 1 else None
    separability = None
    if prototype is not None and full_correct.size:
        negative = similarities[labels[full] != target]
        separability = float(similarities[full_correct].max() - negative.max()) if len(negative) else None
    if not wrong:
        opportunity = 'correct_assignment' if good else ('unknown_assignment' if unknown_write else 'unmatched')
    elif correct.size and feasible_present:
        opportunity = 'local_reranking_cost_feasible'
    elif correct.size:
        opportunity = 'local_rescoring_or_acceptance_change'
    elif full_correct.size:
        opportunity = 'current_stage_coverage_change'
    elif visible:
        opportunity = 'detector_missing_or_label_unverified'
    else:
        opportunity = 'reference_not_visible'
    return dict(reference_known=True, target=int(target), visible=bool(visible),
        detection_coverage=bool(full_correct.size), stage_coverage=bool(correct.size),
        geometry_appearance_coverage=geom_present, cost_feasible_coverage=feasible_present,
        full_geometry_appearance_coverage=bool(np.any(full_geometry[full_correct] <= gate)),
        appearance_topk_recall=topk_recall,
        union_geometry_top3_coverage=bool(set(full_correct.tolist()) & union_indices),
        correct_columns=correct.tolist(), assigned_column=chosen,
        assigned_identity=assigned_identity, wrong=bool(wrong), correct=bool(good),
        unknown_write=bool(unknown_write), unmatched=chosen is None,
        known_history=int(known.sum()), unknown_history=int((~known).sum()),
        correct_history=int(cues.sum()), known_wrong_history=int((known & ~cues).sum()),
        cue_state=cue_state, opportunity=opportunity,
        correct_column_taken_by_other_track=bool(correct_taken),
        chosen_margin=float(margin) if margin is not None else None,
        top1_top2_cost_margin=top12, mean_history_candidate_separability=separability,
        high_confidence_wrong=bool(wrong and row['scores'][chosen] >= .8 and margin >= .05),
        wrong_reactivation=bool(wrong and track['state'] != 1),
        accepted_memory_write_wrong=bool(wrong),
        two_axis=f'{int(bool(correct.size))}x{int(bool(cues.any()))}')


def accumulate(counter, assessment):
    counter['rows'] += 1
    if not assessment['reference_known']:
        counter['unknown_reference'] += 1
        return
    counter['known_reference'] += 1
    for key in ('visible', 'detection_coverage', 'stage_coverage', 'geometry_appearance_coverage',
                'cost_feasible_coverage', 'full_geometry_appearance_coverage',
                'union_geometry_top3_coverage', 'wrong', 'correct', 'unknown_write', 'unmatched',
                'high_confidence_wrong', 'wrong_reactivation', 'correct_column_taken_by_other_track'):
        counter[key] += int(assessment[key])
    if assessment['visible']:
        for key in ('detection_coverage', 'stage_coverage', 'geometry_appearance_coverage',
                    'union_geometry_top3_coverage'):
            counter['visible_' + key] += int(assessment[key])
    for k, present in assessment['appearance_topk_recall'].items():
        counter['appearance_top' + k + '_coverage'] += int(present)
    counter['two_axis_' + assessment['two_axis']] += 1
    counter['cue_' + assessment['cue_state']] += 1
    if assessment['wrong']:
        counter['wrong_' + assessment['opportunity']] += 1
        counter['wrong_two_axis_' + assessment['two_axis']] += 1
        counter['wrong_cue_' + assessment['cue_state']] += 1
    if assessment['known_wrong_history']:
        counter['contaminated_history'] += 1
        counter['contaminated_history_wrong'] += int(assessment['wrong'])
    if assessment['unmatched'] and assessment['visible']:
        counter['unmatched_visible'] += 1
        counter['unmatched_visible_stage_covered'] += int(assessment['stage_coverage'])

