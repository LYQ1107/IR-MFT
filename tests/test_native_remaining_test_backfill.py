"""Behavior grouping must retain ablations and the queue must prove parent exit."""
import copy

import pytest

from tools.native_remaining_test_backfill import signature, selected_row, predecessor_ready


def config():
    return dict(id='old', registration='old prose', backend='native_query', native_seed=0,
        native_cue='ir', detection_arm='rgb', native_query_checkpoint='/fit/native_query/rgb_ir_s0/epoch06.pt',
        association_checkpoint='/fit/association_metric/rgb_s0/epoch30.pt', query_null=True,
        prebirth_recovery=True)


def test_only_snapshot_epoch_can_be_grouped_within_a_behavior():
    first, later = config(), config()
    later['id'] = 'later'
    later['registration'] = 'different provenance prose'
    later['native_query_checkpoint'] = '/fit/native_query/rgb_ir_s0/epoch36.pt'
    assert signature(first, curve=True) == signature(later, curve=True)
    assert signature(first) != signature(later)


@pytest.mark.parametrize('key,value', [('query_null', False), ('prebirth_recovery', False),
    ('native_cue', 'rgb'), ('backend', 'ordered_query'),
    ('association_checkpoint', '/fit/association_metric/rgb_s0/epoch06.pt')])
def test_route_modality_backend_and_rgb_core_remain_distinct(key, value):
    first, altered = config(), config()
    altered[key] = value
    assert signature(first, curve=True) != signature(altered, curve=True)


def test_known_false_flag_default_does_not_duplicate_the_same_case():
    first, altered = config(), config()
    altered['disable_native_increment'] = False
    assert signature(first) == signature(altered)
    altered['disable_native_increment'] = True
    assert signature(first) != signature(altered)


def test_selection_uses_original_dev_and_earlier_tie():
    rows = [dict(HOTA=50, epoch=18, config=dict(id='e18')),
            dict(HOTA=50, epoch=6, config=dict(id='e06')),
            dict(HOTA=49, epoch=36, config=dict(id='e36'))]
    assert selected_row(rows)['epoch'] == 6


def parent():
    return dict(status='COMPLETE_ALL26_FROZEN_NATIVE36_TEST', controller_pid=123,
        dev_jobs={str(i): dict(status='COMPLETE', source_proof=dict(
            all39metrics_tracks_official_CSV_exact=True)) for i in range(8)},
        test_jobs={str(i): dict(status='COMPLETE') for i in range(26)})


def test_complete_label_report_and_actual_exit_are_all_required():
    state = parent()
    assert predecessor_ready(state, True, process_finished=lambda pid: True)
    assert not predecessor_ready(state, False, process_finished=lambda pid: True)
    assert not predecessor_ready(state, True, process_finished=lambda pid: False)
    incomplete = copy.deepcopy(state)
    incomplete['test_jobs'].pop('0')
    assert not predecessor_ready(incomplete, True, process_finished=lambda pid: True)


def test_failed_source_proof_does_not_open_more_test_jobs():
    state = parent()
    state['dev_jobs']['0']['source_proof']['all39metrics_tracks_official_CSV_exact'] = False
    assert not predecessor_ready(state, True, process_finished=lambda pid: True)
