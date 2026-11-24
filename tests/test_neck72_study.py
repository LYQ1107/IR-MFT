import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import launch_neck72_study as launch
from tools.check_neck72_train_crops import compare_crops
from tools.neck72_experiment import (STUDY, ALGORITHMS, read, config_for,
                                    selected_cases, common_budget_decision)
from tools.neck72_smoke_binding import make_smoke
from tools.run_neck72_replay import QueryIncrementObserver, tracker_for
from tools.run_pseudo_ir_comparison import dump, sha256
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.native_cue import native_crop


def curves(study, epochs=(6, 12, 18), values=(40., 40.2, 40.2)):
    return {h['id']+'_'+a: [dict(epoch=e, HOTA=v) for e, v in zip(epochs, values)]
            for h in study['heads'] for a in ALGORITHMS}


def test_common_budget_boundary_and_all_matched_heads_extend_together():
    study = read(STUDY); data = curves(study)
    assert not common_budget_decision(data, 18)['extend_all8_heads']
    data[next(iter(data))][-1]['HOTA'] = 40.201
    decision = common_budget_decision(data, 18)
    assert decision['extend_all8_heads'] and decision['next_epochs'] == 24
    capped = curves(study, (6, 12, 18, 24, 30, 36), (40., 41., 42., 43., 44., 45.))
    assert not common_budget_decision(capped, 36)['extend_all8_heads']


def test_incomplete_dev_curve_cannot_stop_or_select_test():
    study = read(STUDY); data = curves(study)
    data[next(iter(data))].pop()
    with pytest.raises(ValueError, match='partial/missing'):
        common_budget_decision(data, 18)
    with pytest.raises(ValueError, match='All16'):
        common_budget_decision({}, 18)


def test_all_negative_conditions_retained_and_paired_removals_use_primary_epoch():
    study = read(STUDY); data = curves(study, values=(40., 39., 38.))
    primary = study['primary']['head']+'_'+study['primary']['algorithm']
    data[primary][1]['HOTA'] = 41.
    frozen = selected_cases(study, data)
    assert frozen['primary_snapshot'] == 12 and not frozen['Test_accuracy_read_for_selection']
    normals = [c for c in frozen['cases'] if c.get('intervention') == 'normal']
    assert {c['head'] for c in normals} == {h['id'] for h in study['heads']}
    cases = {c['id']: c for c in frozen['cases']}
    for role in ['primary_paired_no_KL', 'primary_paired_extra_RGB', 'primary_no_consensus']:
        assert cases[frozen['roles'][role]]['epoch'] == 12
    assert cases[frozen['roles']['primary_paired_no_KL']]['kind'] == 'clean'
    for case in frozen['cases']:
        if not case.get('baseline'):
            assert case['config']['native_seed'] == 0 and case['config']['native_feature_family'] == 'neck_metric72'


def test_every_cached_crop_must_match_original_pixels_and_rectangle():
    image = np.arange(32*48*3, dtype=np.uint8).reshape(32, 48, 3)
    frames = np.array([1, 1, 3]); boxes = np.array([[1, 2, 16, 20], [4, 5, 22, 27], [2, 3, 21, 23]])
    generated = [native_crop(image, box) for box in boxes]
    crops = np.stack([p[0] for p in generated]); rectangles = np.stack([p[1] for p in generated])
    reader = SimpleNamespace(read=lambda sequence, frame, cue: image)
    result = compare_crops('BT-003', 'rgb', frames, boxes, crops, rectangles, reader)
    assert result['observations'] == 3 and result['actual_source_frames'] == 2
    crops[1, 0, 0, 0] ^= 1
    with pytest.raises(RuntimeError, match='observation1'):
        compare_crops('BT-003', 'rgb', frames, boxes, crops, rectangles, reader)


def test_query_observer_counts_cached_packet_once_and_never_changes_it():
    packet = dict(delta=np.array([[0., -1.5]], np.float32), available=np.array([[True, True]]))
    original = {k: v.copy() for k, v in packet.items()}
    tracker = SimpleNamespace(config={}, query_posterior=lambda frame, context: packet)
    observer = QueryIncrementObserver(tracker)
    assert observer(2, {}) is packet and observer(2, {}) is packet
    assert observer.values['packets'] == 1 and observer.values['pair_cells'] == 2
    assert observer.values['sum_abs_delta'] == 1.5
    for key in packet:
        np.testing.assert_array_equal(packet[key], original[key])


def test_each_sequence_has_a_fresh_track_id_counter():
    for _ in range(2):
        BaseTrack._count = 999
        tracker = tracker_for(dict(backend='byte'))
        values = tracker.update(np.array([[20., 20., 40., 40., .9]], np.float32), (1, 1), (1, 1))
        assert len(values) == 1 and values[0].track_id == 1


@pytest.mark.parametrize('kind', ['clean', 'robust'])
def test_engineering_budget_changes_only_private_smoke_not_original_step(kind):
    assert not torch.cuda.is_initialized()
    entry, proof, captured = make_smoke(kind, '4')
    assert callable(entry) and not captured and proof['original_optimizer_step_AST_exact']
    assert proof['changes'] == dict(output=1, steps=1, epoch_range=1, epochs_metadata=1, status=1)
    assert proof['output_family'].startswith('neck72_engineering_fits/gpu4/')
    assert proof['formal_output_untouched'] and not torch.cuda.is_initialized()


def test_native50_release_requires_all_jobs_and_actual_process_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'ROOT', tmp_path)
    monkeypatch.setattr(launch, 'gpu_predecessor_ready', lambda work: True)
    monkeypatch.setattr(launch, 'finished', lambda pid: True)
    specification = tmp_path / 'configs/perceptual_native_remaining_test_backfill_20261124.json'
    dump(specification, {'fixed_seed': 0})
    status = dict(status='COMPLETE_ALL50_REMAINING_NATIVE_TEST', controller_pid=5,
        specification_sha256=sha256(specification),
        dev_jobs={str(i): dict(status='COMPLETE', pid=i+100) for i in range(50)},
        test_jobs={str(i): dict(status='COMPLETE', pid=i+200) for i in range(50)})
    dump(tmp_path / 'native_remaining_test_backfill_state.json', status)
    report = tmp_path / 'docs/fusion_research/PERCEPTUAL_MFT_20261124_NATIVE_REMAINING_TEST_BACKFILL.json'
    dump(report, dict(status='COMPLETE_ORIGINAL58_NATIVE_BEHAVIORS_SEED0_ALL7_TEST',
                      specification_sha256=sha256(specification)))
    assert launch.predecessors_ready(tmp_path)
    monkeypatch.setattr(launch, 'finished', lambda pid: pid != 210)
    assert not launch.predecessors_ready(tmp_path)
    status['test_jobs'] = {}; dump(tmp_path / 'native_remaining_test_backfill_state.json', status)
    assert not launch.predecessors_ready(tmp_path)


def test_zero_child_exit_does_not_replace_required_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'registered', lambda: None)
    monkeypatch.setattr(launch, 'require_storage', lambda *args: None)
    reservations = []
    monkeypatch.setattr(launch, 'require_host_memory',
                        lambda incoming, maximum_workers: reservations.append((incoming, maximum_workers)))
    monkeypatch.setattr(launch.time, 'sleep', lambda seconds: None)
    class Done:
        pid = 101
        def poll(self):
            return 0
    monkeypatch.setattr(launch.subprocess, 'Popen', lambda *args, **kwargs: Done())
    state = dict(phases={}); queue = launch.Queue(tmp_path, state)
    with pytest.raises(RuntimeError, match='evidence is false'):
        queue.jobs('CHECK', [dict(id='x', command=['unused'])], False, lambda row: False)
    assert state['phases']['CHECK']['x']['status'] == 'FAILED_EVIDENCE'
    assert reservations == [(8*2**30, 2)]
