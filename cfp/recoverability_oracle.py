"""Offline-only joint candidate intervention on frozen logged states.

GT-derived references are passed by an analysis tool. Nothing here is an online
scorer. A maximum feasible correct assignment is not an information-theoretic or
whole-tracker performance upper bound: histories/output identities remain fixed.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def joint_candidate_oracle(rows, proposals, labels, expand=False):
    """Keep unknown-reference existing assignments, maximize known correct edges.

    Each row has its own NULL column. Ties preserve an already correct decision.
    The expanded graph adds existing current proposals only, never GT detections.
    """
    proposals = np.asarray(proposals, np.int64)
    position = {int(index): j for j, index in enumerate(proposals)}
    n, m = len(rows), len(proposals)
    cost = np.full((n, m + n), 1e6, dtype=np.float64)
    for i, row in enumerate(rows):
        cost[i, m + i] = 0
        target = row['target']
        chosen = row['chosen']
        if target is None:
            if chosen is not None and chosen in position:
                cost[i, position[chosen]] = -2
                cost[i, m + i] = 1e6
            continue
        allowed = proposals if expand else np.asarray(row['candidates'], np.int64)
        for index in allowed:
            if index in position and labels[index] == target:
                cost[i, position[int(index)]] = -1 - (1e-6 if index == chosen else 0)
    if n == 0:
        return [], dict(rows=0, N01=0, N10=0)
    row_ids, column_ids = linear_sum_assignment(cost)
    assigned = [None] * n
    for i, j in zip(row_ids, column_ids):
        if cost[i, j] >= 1e6:
            raise ValueError('Oracle graph lacks feasible dedicated NULL/preserved assignment')
        if j < m:
            assigned[i] = int(proposals[j])
    counts = dict(rows=n, known_reference=0, original_correct=0, original_wrong=0,
        oracle_correct=0, N01=0, N10=0, wrong_to_abstention=0, previously_unmatched_or_unknown_to_correct=0)
    for row, chosen in zip(rows, assigned):
        target, original = row['target'], row['chosen']
        if target is None:
            continue
        counts['known_reference'] += 1
        original_known = original is not None and labels[original] >= 0
        old_correct = original_known and labels[original] == target
        old_wrong = original_known and labels[original] != target
        now_correct = chosen is not None and labels[chosen] == target
        counts['original_correct'] += int(old_correct); counts['original_wrong'] += int(old_wrong)
        counts['oracle_correct'] += int(now_correct)
        counts['N01'] += int(old_wrong and now_correct)
        counts['N10'] += int(old_correct and not now_correct)
        counts['wrong_to_abstention'] += int(old_wrong and chosen is None)
        counts['previously_unmatched_or_unknown_to_correct'] += int(not original_known and now_correct)
    return assigned, counts

