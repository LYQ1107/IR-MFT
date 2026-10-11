"""Candidate-set counterfactual utility with explicit, exact RGB fallback."""
import numpy as np
import torch
from torch import nn


EVIDENCE_FIELDS = (
    'candidate_count_log', 'rgb_min', 'rgb_mean', 'rgb_std', 'rgb_margin_with_null', 'rgb_entropy',
    'ir_min', 'ir_mean', 'ir_std', 'ir_margin_with_null', 'ir_entropy',
    'branch_best_disagreement', 'mean_branch_distance_difference', 'ir_available_fraction',
    'geometry_min', 'geometry_mean', 'geometry_std', 'confidence_max', 'confidence_mean',
    'age_fraction', 'gap_fraction', 'history_rgb_dispersion',
)
QUALITY_FIELDS = ('candidate_count_log', 'ir_available_fraction', 'geometry_min', 'geometry_mean',
                  'geometry_std', 'confidence_max', 'confidence_mean', 'age_fraction',
                  'gap_fraction', 'history_rgb_dispersion')


def _checked_distances(rgb, ir):
    rgb = np.asarray(rgb, dtype=np.float32)
    ir = rgb if ir is None else np.asarray(ir, dtype=np.float32)
    if rgb.ndim != 1 or ir.shape != rgb.shape or not np.isfinite(rgb).all() or not np.isfinite(ir).all():
        raise ValueError('Finite aligned one-dimensional candidate distances required')
    return rgb, ir


def candidate_ce(cost, target, temperature=.1, null_distance=.35):
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 1 or not np.isfinite(cost).all() or not np.isfinite([temperature, null_distance, target]).all() or target != int(target) or not 0 <= target <= len(cost) or temperature <= 0:
        raise ValueError('Invalid candidate/NULL supervised decision')
    logits = -np.append(cost, null_distance) / temperature
    peak = logits.max()
    return float(peak + np.log(np.exp(logits - peak).sum()) - logits[int(target)])


def counterfactual_utility(rgb, ir, target, temperature=.1, ir_weight=.5, null_distance=.35):
    rgb, ir = _checked_distances(rgb, ir)
    if not 0 <= ir_weight <= 1:
        raise ValueError('Invalid IR mixture')
    if np.array_equal(rgb, ir) or ir_weight == 0:
        return 0.
    mixed = (1 - ir_weight) * rgb + ir_weight * ir
    return candidate_ce(rgb, target, temperature, null_distance) - candidate_ce(mixed, target, temperature, null_distance)


def utility_mix(rgb, ir, utility, threshold=0., maximum_weight=.5, full_weight_at=.2):
    if not np.isfinite([utility, threshold, maximum_weight, full_weight_at]).all() or full_weight_at <= 0 or not 0 <= maximum_weight <= 1:
        raise ValueError('Invalid calibrated utility or mixing policy')
    if ir is None or utility <= threshold:
        # Preserve the exact RGB object/values; no rounding or normalization.
        return rgb, 0.
    rgb, ir = _checked_distances(rgb, ir)
    weight = float(maximum_weight * np.clip(utility / full_weight_at, 0, 1))
    return (1 - weight) * rgb + weight * ir, weight


def decision_evidence(rgb, ir, geometry, confidence, available, age, gap, history_rgb,
                      temperature=.1, null_distance=.35):
    rgb, ir = _checked_distances(rgb, ir)
    geometry, confidence, available = [np.asarray(x) for x in (geometry, confidence, available)]
    if not len(rgb) or any(x.shape != rgb.shape for x in (geometry, confidence, available)):
        raise ValueError('Evidence requires a nonempty, unchanged candidate set')
    def stats(cost):
        all_cost = np.append(cost, null_distance)
        sorted_cost = np.sort(all_cost)
        logits = -all_cost.astype(np.float64) / temperature
        probability = np.exp(logits - logits.max()); probability /= probability.sum()
        entropy = -np.sum(probability * np.log(np.maximum(probability, 1e-12))) / np.log(len(probability))
        return [float(cost.min()), float(cost.mean()), float(cost.std()), float(sorted_cost[1] - sorted_cost[0]), float(entropy)]
    history = np.asarray(history_rgb, dtype=np.float32)
    dispersion = 0.
    if history.size:
        if history.ndim != 2 or not np.isfinite(history).all():
            raise ValueError('Invalid accepted history features')
        prototype = history.mean(axis=0)
        norm = np.linalg.norm(prototype)
        if norm > 1e-8:
            dispersion = float(np.mean(1 - history @ (prototype / norm)))
    r_all, i_all = np.append(rgb, null_distance), np.append(ir, null_distance)
    disagreement = not np.any((r_all == r_all.min()) & (i_all == i_all.min()))
    result = np.array([np.log1p(len(rgb)) / np.log(65), *stats(rgb), *stats(ir),
                       float(disagreement),
                       float(np.mean(np.abs(rgb - ir))), float(available.mean()),
                       float(geometry.min()), float(geometry.mean()), float(geometry.std()),
                       float(confidence.max()), float(confidence.mean()),
                       min(float(age) / 10000, 1.), min(float(gap) / 37, 1.), dispersion], np.float32)
    if result.shape != (len(EVIDENCE_FIELDS),) or not np.isfinite(result).all():
        raise ValueError('Nonfinite/misaligned utility evidence')
    return result


class UtilityMLP(nn.Module):
    def __init__(self, hidden=64, quality_only=False):
        super().__init__()
        mask = torch.ones(len(EVIDENCE_FIELDS))
        if quality_only:
            mask.zero_()
            for name in QUALITY_FIELDS:
                mask[EVIDENCE_FIELDS.index(name)] = 1
        self.register_buffer('evidence_mask', mask)
        self.network = nn.Sequential(nn.Linear(len(EVIDENCE_FIELDS), hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, evidence):
        if evidence.shape[-1] != len(EVIDENCE_FIELDS) or not torch.isfinite(evidence).all():
            raise ValueError('Invalid estimator evidence')
        return self.network(evidence * self.evidence_mask).squeeze(-1)
