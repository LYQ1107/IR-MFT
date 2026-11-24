"""Deterministic case generation and small, immutable native-study artifacts."""
from pathlib import Path

from tools.native_pyramid_experiment import FIT_CONFIGS, ROOT, STUDY, frozen_json, read
from tools.run_pseudo_ir_comparison import sha256


FRESH = dict(id='NP_source_fresh_RGB', cue='rgb', seed=0, baseline='fresh_RGB_B1')
CANONICAL = dict(id='NP_source_canonical_RGB', cue='rgb', seed=0, baseline='historical_canonical_RGB_B4', canonical=True)


def snapshot_row(work, fit_id, epoch, intervention='normal'):
    if intervention not in ['normal', 'IR_image_spatial_roll', 'unavailable']:
        raise ValueError('Unregistered native intervention')
    arm_path = FIT_CONFIGS / (fit_id+'.json')
    arm = read(arm_path)
    if intervention != 'normal' and arm['cue'] != 'ir':
        raise ValueError('Only the actual IR branch has registered image interventions')
    suffix = {'normal': '', 'IR_image_spatial_roll': '_roll', 'unavailable': '_off'}[intervention]
    checkpoint = Path(work) / 'native_pyramid_fits' / fit_id / f'epoch_{epoch:02d}.pt'
    return dict(id=f'NP_{fit_id}_e{epoch:02d}'+suffix, fit_id=fit_id, epoch=epoch,
        mode=arm['mode'], cue=arm['cue'], removal=arm['removal'], seed=0,
        intervention=intervention, geometry_fixed=arm['geometry_fixed'],
        fit_config_sha256=sha256(arm_path), checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))


def stage_models(work, fits, first_epoch, last_epoch, interventions):
    if not 1 <= first_epoch <= last_epoch <= 12:
        raise ValueError('Unregistered snapshot interval')
    models = [dict(FRESH)]
    for fit in fits:
        for epoch in range(first_epoch, last_epoch+1):
            models.append(snapshot_row(work, fit['id'], epoch))
            if interventions and fit['cue'] == 'ir':
                models.extend(snapshot_row(work, fit['id'], epoch, value)
                              for value in ['IR_image_spatial_roll', 'unavailable'])
    assert len({r['id'] for r in models}) == len(models)
    return models


def register_trial(work, phase, split, models, include_source_cases):
    if split not in ['dev', 'test'] or not phase.replace('_', '').isalnum():
        raise ValueError('Safe registered phase and split required')
    specification_path = Path(work) / 'native_pyramid_specs' / (phase+'.json')
    directory = Path(work) / 'native_pyramid_streams' / phase
    specification = dict(status='FROZEN_BEFORE_NATIVE_OBSERVATION_PRODUCTION',
        study_sha256=sha256(STUDY), split=split, models=models, stream_directory=str(directory))
    frozen_json(specification_path, specification)
    cases = []
    directory_cases = ROOT / f'configs/perceptual_native_pyramid_{split}_cases_20261124'
    for row in models + ([dict(CANONICAL)] if include_source_cases else []):
        if row['id'] == FRESH['id'] and not include_source_cases:
            continue  # A full new RGB stream still anchors numerical reuse.
        case = dict(row, study_sha256=sha256(STUDY), split=split,
            specification=str(specification_path), specification_sha256=sha256(specification_path),
            stream_directory=str(directory))
        path = directory_cases / (row['id']+'.json')
        frozen_json(path, case)
        cases.append(dict(id=row['id'], config=str(path), config_sha256=sha256(path)))
    return specification_path, cases


def test_models(work, selection, removals):
    models, roles = {FRESH['id']: dict(FRESH)}, {}
    for row in selection['all_six_own_Dev_best']:
        model = snapshot_row(work, row['fit_id'], row['epoch'])
        models[model['id']] = model
        roles['normal_'+row['fit_id']] = model['id']
    primary = selection['primary']
    roles['primary_IR'] = snapshot_row(work, primary['fit_id'], primary['epoch'])['id']
    for value, role in [('unavailable', 'primary_IR_unavailable'), ('IR_image_spatial_roll', 'primary_IR_image_roll')]:
        model = snapshot_row(work, primary['fit_id'], primary['epoch'], value)
        models[model['id']] = model
        roles[role] = model['id']
    for removal, choices in removals.items():
        for policy, row in choices.items():
            model = snapshot_row(work, row['fit_id'], row['epoch'])
            models[model['id']] = model
            roles['removal_'+removal+'_'+policy] = model['id']
    roles.update(source_RGB_fresh=FRESH['id'], source_RGB_canonical=CANONICAL['id'],
        matched_RGB=roles['normal_'+selection['primary_mode_RGB_control']['fit_id']],
        matched_gray=roles['normal_'+selection['primary_mode_gray_control']['fit_id']])
    return list(models.values()), roles
