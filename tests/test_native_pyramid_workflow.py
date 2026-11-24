"""Actual queue isolation plus Dev-only plans, case coverage and release gates."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import pytest
import torch

from tools import native_pyramid_design as design
from tools import native_pyramid_experiment as experiment
from tools import launch_native_pyramid_study as launch
from tools.evaluate_native_pyramid import require_test_freeze
from tools.native_pyramid_study_rules import select_models
from yolox.models.native_pyramid_fusion import NativePyramidFusion


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    study = experiment.read(experiment.STUDY)
    configs = tmp_path / 'configs/fits'
    monkeypatch.setattr(design, 'ROOT', tmp_path)
    monkeypatch.setattr(design, 'FIT_CONFIGS', configs)
    fits = copy.deepcopy(study['normal_fits'])
    for row in fits:
        arm = experiment.make_arm(study, row['id'], row['mode'], row['cue'])
        experiment.frozen_json(configs / (row['id']+'.json'), arm)
        for epoch in range(1, 13):
            # Inert case-builder fixtures, never loadable trained weights.
            path = tmp_path / 'native_pyramid_fits' / row['id'] / f'epoch_{epoch:02d}.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((row['id']+str(epoch)).encode())
    return tmp_path, study, fits, configs


def test_first_complete_six_epoch_stage_has_all_matched_snapshots_and_controls(prepared):
    work, study, fits, configs = prepared
    models = design.stage_models(work, fits, 1, 6, True)
    assert len(models) == 61
    normal = [r for r in models if not r.get('baseline') and r['intervention'] == 'normal']
    assert len(normal) == 36
    assert {(r['fit_id'], r['epoch']) for r in normal} == {(f['id'], e) for f in fits for e in range(1, 7)}
    assert len([r for r in models if r.get('intervention') == 'unavailable']) == 12
    assert len([r for r in models if r.get('intervention') == 'IR_image_spatial_roll']) == 12
    specification, cases = design.register_trial(work, 'dev_normal_06', 'dev', models, True)
    assert len(cases) == 62
    assert json.loads(specification.read_text())['models'] == models
    assert design.register_trial(work, 'dev_normal_06', 'dev', models, True)[0] == specification
    extension = design.stage_models(work, fits, 7, 9, True)
    _, additional = design.register_trial(work, 'dev_normal_09', 'dev', extension, False)
    assert len(extension) == 31 and len(additional) == 30
    assert not {r['id'] for r in additional}.intersection(r['id'] for r in cases)
    assert json.loads(Path(cases[0]['config']).read_text())['stream_directory'].endswith('dev_normal_06')


def test_test_plan_preserves_every_normal_fit_and_dev_chosen_removal_policies(prepared):
    work, study, fits, configs = prepared
    rows = []
    for fit in fits:
        for epoch in range(1, 7):
            hota = 45.
            if fit['id'] == 'quality_ir' and epoch == 4: hota = 60.
            if fit['id'] == 'quality_rgb' and epoch == 5: hota = 59.
            if fit['id'] == 'quality_gray' and epoch == 3: hota = 58.
            rows.append(dict(id=f"NP_{fit['id']}_e{epoch:02d}", fit_id=fit['id'], mode=fit['mode'], cue=fit['cue'], epoch=epoch,
                split='dev', sequences=['BT-005', 'SN-013'], frames=4834, seed=0, intervention='normal',
                metrics=dict(HOTA=hota, AssA=30., IDF1=40.)))
    selection = select_models(rows, 6, 47.579)
    removal = 'no_fusion_quality_gate'
    fit_id = 'quality_'+removal
    experiment.frozen_json(configs / (fit_id+'.json'), experiment.make_arm(study, fit_id, 'quality', 'ir', removal))
    for epoch in [2, 4]:
        path = work / 'native_pyramid_fits' / fit_id / f'epoch_{epoch:02d}.pt'
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(str(epoch).encode())
    choices = {removal: dict(Dev_best=dict(fit_id=fit_id, epoch=2), paired_primary_epoch=dict(fit_id=fit_id, epoch=4))}
    models, roles = design.test_models(work, selection, choices)
    assert len(models) == 11
    assert len([r for r in roles if r.startswith('normal_')]) == 6
    assert roles['primary_IR'] == 'NP_quality_ir_e04'
    assert roles['matched_RGB'] == 'NP_quality_rgb_e05'
    assert roles['matched_gray'] == 'NP_quality_gray_e03'
    assert roles['primary_IR_unavailable'] == 'NP_quality_ir_e04_off'
    assert roles['primary_IR_image_roll'] == 'NP_quality_ir_e04_roll'
    assert roles['removal_'+removal+'_Dev_best'].endswith('_e02')
    assert roles['removal_'+removal+'_paired_primary_epoch'].endswith('_e04')


@pytest.mark.parametrize('mode', ['quality', 'temporal'])
def test_effective_removal_configs_keep_capacity_and_explicit_scope(mode):
    study = experiment.read(experiment.STUDY)
    removals = study['removals']['all_modes'] + (study['removals']['temporal_only'] if mode == 'temporal' else [])
    for removal in removals:
        arm = experiment.make_arm(study, mode+'_'+removal, mode, 'ir', removal)
        module = NativePyramidFusion(**arm['module_kwargs'])
        assert sum(p.numel() for p in module.parameters()) == study['architecture'][mode+'_parameters']
        if removal == 'no_RGB_protection':
            assert not module.protect_rgb and arm['fit']['lambda_reliable_RGB'] == 0
        if removal == 'fixed_original_RGB_geometry': assert arm['geometry_fixed']
        if removal == 'no_fusion_quality_gate':
            assert not module.use_quality
            if mode == 'temporal': assert module.noise is not None
    with pytest.raises(ValueError): experiment.make_arm(study, 'invalid', mode, 'rgb', removals[0])


def test_gpu_release_requires_full_predecessor_export_and_actual_process_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, 'ROOT', tmp_path)
    monkeypatch.setattr(experiment, 'require_predecessor', lambda work: True)
    data = dict(status='COMPLETE_ALL9_FORMAL_BACKFILL_CASES', controller_pid=1,
        jobs={str(n): dict(status='COMPLETE') for n in range(9)},
        producers={str(n): dict(status='COMPLETE', pid=n+2) for n in range(7)})
    experiment.frozen_json(tmp_path / 'structural_legacy_backfill_state.json', data)
    monkeypatch.setattr(experiment, 'pid_finished', lambda pid: True)
    assert not experiment.predecessor_ready(tmp_path)
    report = tmp_path / 'docs/fusion_research/PERCEPTUAL_MFT_20261124_STRUCTURAL_LEGACY_BACKFILL_TEST.json'
    experiment.frozen_json(report, dict(status='COMPLETE_INITIAL_STRUCTURAL_FORMAL_ALL7_TEST_BACKFILL'))
    assert experiment.predecessor_ready(tmp_path)
    monkeypatch.setattr(experiment, 'pid_finished', lambda pid: pid != 3)
    assert not experiment.predecessor_ready(tmp_path)
    monkeypatch.setattr(experiment, 'require_predecessor', lambda work: False)
    assert not experiment.predecessor_ready(tmp_path)


def test_test_production_refuses_incomplete_dev_or_different_frozen_spec(tmp_path):
    path = tmp_path / 'native_pyramid_Test_selection_frozen.json'
    spec = tmp_path / 'test_spec.json'; spec.write_text('{}')
    freeze = dict(status='FROZEN_FROM_COMPLETE_NATIVE_DEV_BEFORE_TEST', complete_Dev_before_Test=False,
        study_sha256=experiment.sha256(experiment.STUDY), specification_sha256=experiment.sha256(spec))
    path.write_text(json.dumps(freeze))
    with pytest.raises(AssertionError): require_test_freeze(tmp_path, spec)
    freeze['complete_Dev_before_Test'] = True
    path.write_text(json.dumps(freeze))
    require_test_freeze(tmp_path, spec)
    spec.write_text('{"changed":true}')
    with pytest.raises(AssertionError): require_test_freeze(tmp_path, spec)


def test_real_cpu_children_get_no_gpu_and_required_evidence_is_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, 'registry', lambda work: {})
    actual_sleep = time.sleep
    monkeypatch.setattr(launch.time, 'sleep', lambda seconds: actual_sleep(.01))
    state = dict(data_root=str(tmp_path))
    queue = launch.Queue(tmp_path, state)
    code = 'import os; assert os.environ["CUDA_VISIBLE_DEVICES"] == ""'
    tasks = [dict(id=str(n), command=[sys.executable, '-c', code]) for n in range(3)]
    queue.jobs('CPU_engineering', tasks, False, lambda entry: entry['exit_code'] == 0)
    rows = state['phases']['CPU_engineering']
    assert len(rows) == 3 and all(r['status'] == 'COMPLETE' for r in rows.values())
    assert {r['slot'] for r in rows.values()}.issubset({'CPU0', 'CPU1'})
    with pytest.raises(RuntimeError, match='required evidence'):
        queue.jobs('rejected_evidence', [dict(id='x', command=[sys.executable, '-c', code])], False, lambda entry: False)
