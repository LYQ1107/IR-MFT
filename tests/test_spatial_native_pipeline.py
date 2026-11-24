"""CPU fixtures for orchestration/provenance; no fitted model or new accuracy."""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from tools import launch_spatial_native as launch
from tools import benchmark_spatial_native as timing
from tools.native_pyramid_stream_tracks import StreamTrackWriter
from tools.observe_spatial_native import CANONICAL, FRESH, full_precision_selection, require_test_freeze
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.spatial_native_design import removal_plan
from tools.spatial_native_study import STUDY, arm_path, make_arm, read

ROOT = Path(__file__).resolve().parents[1]


def arms(base):
    study = read(STUDY)
    for fit in study['normal_fits']:
        dump(arm_path(base, fit['id']), make_arm(study, fit['id'], fit['variant'], fit['cue']))
        for epoch in range(1, 7):
            p = Path(base)/'fits'/fit['id']/f'epoch_{epoch:02d}.pt'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b'SYNTHETIC_UNIT_FIXTURE_NOT_WEIGHTS')
    return study


def test_complete_primary_only_adds_two_new_removal_fits_and_reuses_exact_no_rho(tmp_path):
    study = arms(tmp_path)
    value = removal_plan(study, tmp_path, dict(fit_enabled_removals=True,
        primary=dict(variant='quality_spatial_redundancy')))
    assert len(value['new_fits']) == 2
    assert value['conditions']['no_redundancy']['reused_normal_fit']
    assert value['conditions']['no_redundancy']['fit_id'] == 'quality_spatial_ir'
    assert not value['conditions']['no_spatial']['reused_normal_fit']
    assert value['Test_accuracy_used'] is False


def test_quality_removal_reuses_spatial_fit_without_training_it_again(tmp_path):
    study = arms(tmp_path)
    value = removal_plan(study, tmp_path, dict(fit_enabled_removals=True, primary=dict(variant='quality_spatial')))
    assert len(value['new_fits']) == 1
    assert value['conditions']['no_frame_quality']['fit_id'] == 'spatial_ir'
    assert value['conditions']['no_frame_quality']['reused_normal_fit']
    assert 'no_redundancy' not in value['conditions']


def test_negative_visual_gate_keeps_no_fictitious_removal(tmp_path):
    study = arms(tmp_path)
    assert removal_plan(study, tmp_path, dict(fit_enabled_removals=False))['conditions'] == {}


def test_full_precision_parser_reads_original_dev_csv_without_modifying_it():
    path = ROOT/'docs/fusion_research/perceptual_20261122_official_tables/strong_rgb24_dev/F_rgb_rgb_s0_e24.csv'
    protocol = read(ROOT/'configs/cfp/protocol_v2.json')
    before = sha256(path)
    selected = full_precision_selection(path, 'dev', protocol)
    with path.open() as stream:
        original = next(r for r in csv.DictReader(stream) if r['seq'] == 'COMBINED')
    assert selected['HOTA'] == 100*float(original['HOTA___AUC'])
    assert selected['IDF1'] == 100*float(original['IDF1'])
    assert sha256(path) == before
    with pytest.raises(AssertionError):
        full_precision_selection(path, 'test', protocol)


def test_test_freeze_rejects_missing_or_changed_complete_dev(tmp_path):
    specification = tmp_path/'spec.json'
    dump(specification, dict(unit_fixture=True))
    with pytest.raises(FileNotFoundError):
        require_test_freeze(tmp_path, specification)
    dump(tmp_path/'complete_Dev.json', dict(unit_fixture=True))
    freeze = dict(status='FROZEN_SPATIAL_ALL_COMPLETE_DEV_BEFORE_TEST', study_sha256=sha256(STUDY),
        complete_Dev_before_Test=True, Test_accuracy_used_for_selection=False,
        complete_Dev_sha256=sha256(tmp_path/'complete_Dev.json'), specification_sha256=sha256(specification))
    dump(tmp_path/'Test_selection_frozen.json', freeze)
    assert require_test_freeze(tmp_path, specification) == freeze
    dump(tmp_path/'complete_Dev.json', dict(unit_fixture='changed'))
    with pytest.raises(AssertionError):
        require_test_freeze(tmp_path, specification)


def test_cpu_two_frame_canary_cannot_be_accepted_as_real_gpu_eligibility(tmp_path):
    study = arms(tmp_path)
    dump(tmp_path/'pipeline_registered.json', dict(unit_fixture=True))
    fit_id = study['normal_fits'][0]['id']
    meta = dict(study_sha256=sha256(STUDY), fit_config_sha256=sha256(arm_path(tmp_path, fit_id)),
        registry_sha256=sha256(tmp_path/'pipeline_registered.json'))
    proof = dict(status='COMPLETE_SPATIAL_GPU_TRAIN_B1_ENGINEERING', GPU_work=True, metadata=meta,
        record=dict(actual_updates=3, delivered_Train_frames=12, inference_B1_frames=2,
            source_zero_missing_checks=[dict(fresh_direct_packets_exact=True, zero_full_head_exact=True,
                learned_missing_full_head_exact=True)]*2,
            frozen_RGB_weights_BN_unchanged_before_explicit_head_precision_conversion=True,
            inference_RGB_state_unchanged=True))
    dump(tmp_path/'engineering'/fit_id/'proof.json', proof)
    with pytest.raises(AssertionError):
        launch.verify_probe(tmp_path, fit_id)


class FixtureQueue:
    def __init__(self, work, fail_probe=False):
        self.work, self.base = work, work/'native_spatial'
        self.state, self.called, self.fail_probe = dict(phases={}), [], fail_probe

    def save(self):
        pass

    def jobs(self, phase, tasks, gpu, check):
        self.called.append((phase, [t['id'] for t in tasks]))
        if phase == 'NORMAL_GPU_ENGINEERING':
            if self.fail_probe:
                raise RuntimeError('Synthetic failed GPU prerequisite; nothing actually launched')
            for task in tasks:
                dump(self.base/'engineering'/task['id']/'proof.json', dict(metadata=dict(initial_native_sha256='fixture_initial')))


def fixture_trial(queue, plan, protocol, phase, split, models, include_sources):
    rows = []
    if split == 'test':
        return rows, None, None
    for model in models+([dict(CANONICAL)] if include_sources else []):
        if model == FRESH and not include_sources:
            continue
        score = 48.5 if model.get('baseline') else {'rgb': 50., 'gray': 49., 'ir': 48.}[model['cue']]+model['epoch']*.01
        if model.get('intervention') == 'unavailable':
            score = 48.5
        rows.append(dict(model, split='dev', sequences=protocol['dev'], frames=4834,
            selection_precision='original276CSV_COMBINED_full_precision',
            selection_metrics=dict(HOTA=score, AssA=30., IDF1=50.)))
    return rows, None, None


def execution_fixture(tmp_path, monkeypatch, fail_probe=False):
    queue = FixtureQueue(tmp_path, fail_probe)
    study = arms(queue.base)
    dump(tmp_path/'native_spatial_worker_CPU_proof.json', dict(records=[dict(initial_native_sha256='fixture_initial')]))
    protocol = read(ROOT/'configs/cfp/protocol_v2.json')
    monkeypatch.setattr(launch, 'run_trial', fixture_trial)
    return queue, study, protocol


def test_all_nine_negative_normal_models_still_get_frozen_test_and_global_controls(tmp_path, monkeypatch):
    queue, study, protocol = execution_fixture(tmp_path, monkeypatch)
    launch.execute(queue, {}, study, protocol)
    trained = [k for phase, keys in queue.called if phase.startswith('TRAIN_') for k in keys]
    assert set(trained) == {f['id'] for f in study['normal_fits']} and len(trained) == 9
    freeze = read(queue.base/'Test_selection_frozen.json')
    assert len([k for k in freeze['roles'] if k.startswith('normal_')]) == 9
    assert freeze['roles']['global_Dev_selected_rgb'].endswith('_rgb_e06')
    assert freeze['roles']['global_Dev_selected_gray'].endswith('_gray_e06')
    assert not freeze['selection']['fit_enabled_removals'] and not freeze['Test_accuracy_used_for_selection']
    assert queue.called[-1][0] == 'INDIVIDUAL_RUNTIME'


def test_failed_gpu_prerequisite_prevents_any_formal_train_or_test_freeze(tmp_path, monkeypatch):
    queue, study, protocol = execution_fixture(tmp_path, monkeypatch, fail_probe=True)
    with pytest.raises(RuntimeError):
        launch.execute(queue, {}, study, protocol)
    assert len(queue.called) == 1
    assert not (queue.base/'Test_selection_frozen.json').exists()


def test_missing_normal_dev_stops_before_next_variant_training(tmp_path, monkeypatch):
    queue, study, protocol = execution_fixture(tmp_path, monkeypatch)
    def missing(*args, **kwargs):
        rows, a, b = fixture_trial(*args, **kwargs)
        index = next(i for i, r in enumerate(rows) if not r.get('baseline') and r['intervention'] == 'normal')
        return rows[:index]+rows[index+1:], a, b
    monkeypatch.setattr(launch, 'run_trial', missing)
    with pytest.raises(ValueError):
        launch.execute(queue, {}, study, protocol)
    assert [p for p, _ in queue.called] == ['NORMAL_GPU_ENGINEERING', 'TRAIN_spatial_06']
    assert not (queue.base/'Test_selection_frozen.json').exists()


def test_real_cached_rgb_prefix_matches_independent_full_emission_and_rejects_corruption(tmp_path, monkeypatch):
    source = Path('/data3/liuyeqiang/IR-MFT_assets/pseudo_ir_comparison_20261120/eval_rgb_matched_dev/detections/BT-005.npz')
    source_sha = sha256(source)
    with np.load(source, allow_pickle=False) as value:
        packets = [np.c_[value['boxes'][value['offsets'][f]:value['offsets'][f+1]],
            value['scores'][value['offsets'][f]:value['offsets'][f+1]]].astype(np.float32) for f in range(64)]
    writer = StreamTrackWriter(tmp_path, 'BT-005', ['fixture'])
    for frame, array in enumerate(packets, 1):
        writer.append(frame, dict(fixture=array))
        if frame == 32:
            prefix = writer.packet_hashes['fixture'].hexdigest()
    stream = writer.close(64)['fixture']
    lines = [line for line in Path(stream['track_path']).read_text().splitlines(keepends=True) if int(line.split(',')[0]) <= 32]
    reference = dict(prefix32_source_rows=[['BT-005', f, 'fixture_RGB_SHA', 'fixture_IR_SHA'] for f in range(1, 33)],
        prefix32_packet_sha256=dict(fixture=prefix),
        prefix32_literal_MOT_sha256=dict(fixture=hashlib.sha256(''.join(lines).encode()).hexdigest()))
    class Engine:
        needs_actual_IR_image = False
        def reset(self, seq):
            self.frame = 0
        def step(self, rgb, ir, seq, frame):
            self.frame += 1
            assert self.frame == frame
            return dict(fixture=packets[frame-1]), {}
    monkeypatch.setattr(timing, 'actual_frame', lambda *a: (None, None, 'fixture_RGB_SHA', None))
    assert timing.check_prefix(Engine(), dict(id='fixture'), {}, 'BT-005', reference)['exact_literal_MOT_prefix']
    damaged = copy.deepcopy(reference)
    damaged['prefix32_packet_sha256']['fixture'] = 'wrong'
    with pytest.raises(AssertionError):
        timing.check_prefix(Engine(), dict(id='fixture'), {}, 'BT-005', damaged)
    assert sha256(source) == source_sha


def test_failed_child_does_not_terminate_healthy_sibling_or_launch_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'registered', lambda *a: {})
    monkeypatch.setattr(launch, 'require_storage', lambda *a, **k: None)
    monkeypatch.setattr(launch, 'require_host_memory', lambda *a, **k: None)
    monkeypatch.setattr(launch.time, 'sleep', lambda *a: None)
    launched = []
    class Process:
        pid = os.getpid()
        def __init__(self, command, **kwargs):
            self.name, self.polls = command[0], 0
            launched.append(self.name)
        def poll(self):
            self.polls += 1
            return 1 if self.name == 'failure' else (0 if self.polls > 1 else None)
        def terminate(self):
            raise AssertionError('Healthy fixture sibling was terminated')
    monkeypatch.setattr(launch.subprocess, 'Popen', Process)
    queue = launch.Queue(tmp_path, tmp_path/'study', dict(phases={}), str(tmp_path))
    tasks = [dict(id=name, command=[name]) for name in ['failure', 'healthy', 'pending']]
    with pytest.raises(RuntimeError):
        queue.jobs('fixture', tasks, False, lambda e: True)
    assert launched == ['failure', 'healthy']
    assert queue.state['phases']['fixture']['healthy']['status'] == 'COMPLETE'
    assert queue.state['phases']['fixture']['pending']['status'] == 'PENDING'


def test_positive_removal_deltas_cannot_hide_a_failed_wrong_ir_control():
    from tools.export_spatial_native import scientific_effects
    roles = {key: key for key in ['primary_IR', 'source_RGB_fresh', 'source_RGB_canonical', 'matched_RGB',
        'matched_gray', 'global_Dev_selected_rgb', 'global_Dev_selected_gray', 'primary_IR_unavailable',
        'primary_IR_image_roll', 'removal_no_spatial_Dev_best', 'removal_no_spatial_paired_primary_epoch']}
    objects = {key: dict(combined47=dict(HOTA=49., IDF1=50., AssA=30., IDSW=20)) for key in roles}
    objects['primary_IR']['combined47']['HOTA'] = 50.
    choice = dict(selection_metrics=dict(HOTA=49.))
    freeze = dict(roles=roles, selection=dict(primary=dict(selection_metrics=dict(HOTA=50.))),
        removals=dict(no_spatial=dict(Dev_best=choice, paired_primary_epoch=choice)))
    assert scientific_effects(freeze, objects, objects)['removal_effects']['no_spatial']['necessary_numeric_conditions']
    wrong = copy.deepcopy(objects)
    wrong['primary_IR_image_roll']['combined47']['HOTA'] = 51.
    result = scientific_effects(freeze, objects, wrong)
    assert not result['removal_effects']['no_spatial']['necessary_numeric_conditions']
    assert not result['removal_effects']['no_spatial']['independently_validated_innovation']


def test_publisher_preserves_original276csv_and_all47_fields(tmp_path, monkeypatch):
    from tools import export_spatial_native as publish
    from tools import export_perceptual_metrics47 as metrics47
    import shutil
    plan, protocol = read(ROOT/'configs/perceptual_mft_20261122.json'), read(ROOT/'configs/cfp/protocol_v2.json')
    schema, _ = metrics47.official_schema(plan['evaluation']['TrackEval_root'])
    original = ROOT/'docs/fusion_research/perceptual_20261122_official_tables/strong_rgb24_dev/F_rgb_rgb_s0_e24.csv'
    converted, _ = metrics47.read_table(original, schema, protocol)
    combined = next(r for r in converted if r['seq'] == 'COMBINED')
    baseline_sha = sha256(original)
    base = tmp_path/'work'
    source = base/'evaluation/dev'/FRESH['id']
    source_csv = source/'trackeval_output/pedestrian_detailed.csv'
    source_csv.parent.mkdir(parents=True)
    shutil.copy2(original, source_csv)
    sources, coverage = {}, {}
    for sequence, frames in [('BT-005', 1834), ('SN-013', 3000)]:
        track = tmp_path/(sequence+'.txt')
        track.write_text('SYNTHETIC_PUBLISHER_METADATA_FIXTURE_NOT_PREDICTIONS\n')
        stream = dict(frames=frames, track_path=str(track), track_sha256=sha256(track))
        manifest_path = tmp_path/(sequence+'.json')
        dump(manifest_path, dict(streams={FRESH['id']: stream}))
        coverage[sequence] = stream
        sources[sequence] = dict(manifest_path=str(manifest_path), manifest_sha256=sha256(manifest_path))
    dump(source/'metrics.json', dict(status='COMPLETE', trackeval={s['metric']: combined[s['metric']]
        for s in schema if s['in_official_headline']},
        selection_metrics={k: combined[k] for k in ['HOTA', 'AssA', 'IDF1']},
        manifest=dict(study_sha256=sha256(STUDY), config=dict(id=FRESH['id'], model=FRESH),
            sequences=protocol['dev'], split='dev', frames=4834, sources=sources, coverage=coverage)))
    monkeypatch.setattr(publish, 'ROOT', tmp_path)
    monkeypatch.setattr(publish, 'DOCS', tmp_path/'docs')
    monkeypatch.setattr(metrics47, 'ROOT', tmp_path)
    payload, reference = publish.publish_case(base, 'dev', FRESH, protocol, schema)
    assert len(payload['combined47']) == 47 and len(payload['metrics39']) == 39
    assert len(payload['per_sequence47']) == 3
    assert payload['official_CSV_sha256'] == baseline_sha
    assert sha256(original) == baseline_sha
    assert reference['sha256'] == sha256(tmp_path/reference['file'])
