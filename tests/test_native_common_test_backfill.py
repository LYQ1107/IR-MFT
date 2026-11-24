"""Reject incomplete or changed original Dev choices; never inherit seed replicas."""
import copy

import pytest

from tools.native_common_test_backfill import choices


def decision():
    rows = []
    for seed in [0, 1, 2]:
        for algorithm in ['robust', 'consensus']:
            for frontend in ['new', 'rgb']:
                for cue in ['ir', 'rgb']:
                    curve = [dict(epoch=e, HOTA=42 + seed * 100 + e / 100) for e in [6, 12, 18, 24, 30, 36]]
                    rows.append(dict(seed=seed, algorithm=algorithm, frontend=frontend, cue=cue,
                                     curve=curve, best=copy.deepcopy(curve[-1])))
    return dict(status='STOP_ALL_MATCHED_HEADS', epoch=36, stopped_at_cap=True, rows=rows)


def test_old_multiple_seed_results_do_not_add_new_seed_work():
    selected = choices(decision())
    assert len(selected) == 8 and {row['seed'] for row in selected} == {0}


def test_incomplete_curve_cannot_be_called_complete_common36():
    data = decision()
    data['rows'][0]['curve'].pop(0)
    with pytest.raises(AssertionError):
        choices(data)


def test_original_dev_best_cannot_be_replaced():
    data = decision()
    data['rows'][0]['best'] = copy.deepcopy(data['rows'][0]['curve'][0])
    with pytest.raises(AssertionError):
        choices(data)


def test_dev_tie_keeps_earlier_checkpoint():
    data = decision()
    data['rows'][0]['curve'][-1]['HOTA'] = data['rows'][0]['curve'][-2]['HOTA']
    data['rows'][0]['best'] = copy.deepcopy(data['rows'][0]['curve'][-2])
    assert choices(data)[4]['best']['epoch'] == 30
