"""Preregistered Dev-only budget, checkpoint and removal decisions.

No file access or Test argument: callers must supply complete official Dev
curves and attest their coverage before applying these decisions.
"""
import math


NORMAL = tuple((mode, cue) for mode in ['quality', 'temporal'] for cue in ['rgb', 'gray', 'ir'])


def complete_curves(rows, through_epoch):
    """Reject missing, duplicated or nonfinite Dev cases before any selection."""
    expected = {(m, c, e) for m, c in NORMAL for e in range(1, through_epoch + 1)}
    found = {}
    for row in rows:
        key = row['mode'], row['cue'], row['epoch']
        if (key not in expected or key in found or row['split'] != 'dev' or
                row['sequences'] != ['BT-005', 'SN-013'] or row['frames'] != 4834 or
                row['seed'] != 0 or row['intervention'] != 'normal'):
            raise ValueError('Require unique, normal, seed0, complete two-video Dev cases')
        if any(not math.isfinite(float(row['metrics'][k])) for k in ['HOTA', 'AssA', 'IDF1']):
            raise ValueError('Nonfinite Dev selection metrics')
        found[key] = row
    if set(found) != expected:
        raise ValueError('All six matched complete Dev curves are required')
    return found


def best_dev(rows):
    if not rows:
        raise ValueError('Cannot select an empty Dev curve')
    # Smaller tuple wins: higher metrics, then earlier snapshot, then quality.
    return min(rows, key=lambda r: (-r['metrics']['HOTA'], -r['metrics']['AssA'],
        -r['metrics']['IDF1'], r['epoch'], 0 if r['mode'] == 'quality' else 1))


def budget_decision(rows, stage, intervention_coverage_complete):
    if stage not in [6, 9, 12] or not intervention_coverage_complete:
        raise ValueError('Registered stage and every required Dev intervention must be complete')
    curves = complete_curves(rows, stage)
    comparisons = []
    earlier = stage - 3
    for mode, cue in NORMAL:
        values = [curves[mode, cue, e] for e in range(1, stage + 1)]
        before = best_dev([r for r in values if r['epoch'] <= earlier])
        now = best_dev(values)
        comparisons.append(dict(mode=mode, cue=cue, earlier_best_HOTA=before['metrics']['HOTA'],
            current_best_HOTA=now['metrics']['HOTA'],
            improvement_pp=now['metrics']['HOTA'] - before['metrics']['HOTA']))
    extend = stage < 12 and any(r['improvement_pp'] > .2 + 1e-10 for r in comparisons)
    return dict(completed_stage=stage, next_stage=stage+3 if extend else None,
                extend_all_six=extend, comparisons=comparisons,
                numerical_convergence_claim=False, selection_input_split='dev')


def select_models(rows, through_epoch, direct_rgb_dev_HOTA):
    complete_curves(rows, through_epoch)
    normal = {(m, c): best_dev([r for r in rows if (r['mode'], r['cue']) == (m, c)])
              for m, c in NORMAL}
    primary = best_dev([r for r in rows if r['cue'] == 'ir'])
    rgb, gray = normal[primary['mode'], 'rgb'], normal[primary['mode'], 'gray']
    threshold = max(float(direct_rgb_dev_HOTA), rgb['metrics']['HOTA'], gray['metrics']['HOTA'])
    return dict(primary=primary, all_six_own_Dev_best=[normal[key] for key in NORMAL],
                primary_mode_RGB_control=rgb, primary_mode_gray_control=gray,
                fit_registered_removals=primary['metrics']['HOTA'] > threshold,
                removal_gate_threshold_Dev_HOTA=threshold,
                all_six_still_require_full_Test=True, selection_input_split='dev',
                goal_met=False, supported_innovations=0)
