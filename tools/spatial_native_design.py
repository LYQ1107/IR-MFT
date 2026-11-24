"""Deterministic spatial cases, exact semantic reuse and Dev-frozen Test roles."""
from pathlib import Path

from tools.observe_spatial_native import CANONICAL, FRESH, full_precision_selection
from tools.spatial_native_rules import best_dev
from tools.spatial_native_study import STUDY, arm_path, computation_key, frozen_json, make_arm, read
from tools.run_pseudo_ir_comparison import sha256


def snapshot(base, fit_id, epoch, intervention='normal'):
    if not isinstance(epoch, int) or not 1 <= epoch <= 12 or intervention not in ['normal', 'unavailable', 'IR_image_spatial_roll']:
        raise ValueError('Registered snapshot and current-image intervention required')
    path = arm_path(base, fit_id)
    arm = read(path)
    if intervention != 'normal' and arm['cue'] != 'ir':
        raise ValueError('IR interventions cannot be applied to a different cue')
    checkpoint = Path(base)/'fits'/fit_id/f'epoch_{epoch:02d}.pt'
    suffix = {'normal': '', 'unavailable': '_off', 'IR_image_spatial_roll': '_roll'}[intervention]
    return dict(id=f'NS_{fit_id}_e{epoch:02d}'+suffix, fit_id=fit_id, epoch=epoch,
        variant=arm['variant'], mode='quality', cue=arm['cue'], removal=arm['removal'], seed=0,
        intervention=intervention, geometry_fixed=arm['geometry_fixed'],
        fit_config_sha256=sha256(path), checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))


def stage_models(base, fits, first_epoch, last_epoch, interventions):
    models = [dict(FRESH)]
    for fit in fits:
        for epoch in range(first_epoch, last_epoch+1):
            models.append(snapshot(base, fit['id'], epoch))
            if interventions and fit['cue'] == 'ir':
                models.extend(snapshot(base, fit['id'], epoch, i) for i in ['unavailable', 'IR_image_spatial_roll'])
    assert len({r['id'] for r in models}) == len(models)
    return models


def register_trial(base, phase, split, models, include_sources):
    if split not in ['dev', 'test'] or not phase.replace('_', '').isalnum():
        raise ValueError('Safe registered phase/split required')
    assert models[0] == FRESH and len({r['id'] for r in models}) == len(models)
    directory = Path(base)/'streams'/phase
    path = Path(base)/'specifications'/(phase+'.json')
    value = dict(status='FROZEN_SPATIAL_OBSERVATIONS_BEFORE_INFERENCE', study_sha256=sha256(STUDY),
        split=split, models=models, stream_directory=str(directory))
    frozen_json(path, value)
    cases = []
    for model in models+([dict(CANONICAL)] if include_sources else []):
        if model == FRESH and not include_sources:
            continue
        case = dict(id=model['id'], model=model, split=split, study_sha256=sha256(STUDY),
            canonical=model == CANONICAL, specification=str(path), specification_sha256=sha256(path),
            stream_directory=str(directory))
        config_path = Path(base)/'cases'/split/(model['id']+'.json')
        frozen_json(config_path, case)
        cases.append(dict(id=model['id'], config=str(config_path), config_sha256=sha256(config_path)))
    return path, cases


def metric_row(base, split, case, protocol):
    path = Path(base)/'evaluation'/split/case['id']/'metrics.json'
    data = read(path)
    config = read(case['config'])
    manifest = data['manifest']
    assert data['status'] == 'COMPLETE' and sha256(case['config']) == case['config_sha256']
    assert manifest['config_sha256'] == case['config_sha256'] and manifest['config'] == config
    assert manifest['study_sha256'] == sha256(STUDY) and manifest['split'] == split
    assert manifest['frames'] == {'dev': 4834, 'test': 15985}[split]
    assert manifest['sequences'] == protocol['dev' if split == 'dev' else 'historical_test']
    csv_path = path.parent/'trackeval_output/pedestrian_detailed.csv'
    assert sha256(csv_path) == data['official_CSV_sha256']
    selected = full_precision_selection(csv_path, split, protocol)
    assert selected == data['selection_metrics']
    return dict(config['model'], split=split, sequences=manifest['sequences'], frames=manifest['frames'],
        metrics39=data['trackeval'], selection_metrics=selected,
        selection_precision=data['selection_precision'], source_metrics=str(path),
        source_metrics_sha256=sha256(path), official_CSV_sha256=sha256(csv_path), config=str(case['config']),
        config_sha256=case['config_sha256'])


def removal_plan(study, base, selection):
    if not selection['fit_enabled_removals']:
        return dict(new_fits=[], conditions={}, Test_accuracy_used=False)
    variant = selection['primary']['variant']
    configured = next(v for v in study['variants'] if v['name'] == variant)
    normals = {r['id']: read(arm_path(base, r['id'])) for r in study['normal_fits'] if r['cue'] == 'ir'}
    conditions, new = {}, []
    for label, flag in [('no_spatial', 'use_spatial'), ('no_frame_quality', 'use_quality'), ('no_redundancy', 'use_redundancy')]:
        if not configured[flag]:
            continue
        key = 'remove_'+label
        arm = make_arm(study, key, variant, 'ir', label)
        matches = [fit_id for fit_id, normal in normals.items() if computation_key(normal) == computation_key(arm)]
        assert len(matches) <= 1
        if matches:
            conditions[label] = dict(fit_id=matches[0], reused_normal_fit=True,
                computation_key=computation_key(arm), source_arm_sha256=sha256(arm_path(base, matches[0])))
        else:
            path = arm_path(base, key)
            frozen_json(path, arm)
            new.append(dict(id=key, variant=variant, cue='ir', removal=label, physical_gpu=['4', '5'][len(new)%2]))
            conditions[label] = dict(fit_id=key, reused_normal_fit=False,
                computation_key=computation_key(arm), source_arm_sha256=sha256(path))
    assert len(new) <= 2
    return dict(new_fits=new, conditions=conditions, Test_accuracy_used=False)


def test_models(base, selection, removal_choices):
    models, roles = {FRESH['id']: dict(FRESH)}, {}
    for row in selection['all_nine_own_Dev_best']:
        model = snapshot(base, row['fit_id'], row['epoch'])
        models[model['id']] = model
        roles['normal_'+row['fit_id']] = model['id']
    primary = selection['primary']
    roles['primary_IR'] = snapshot(base, primary['fit_id'], primary['epoch'])['id']
    for intervention, role in [('unavailable', 'primary_IR_unavailable'), ('IR_image_spatial_roll', 'primary_IR_image_roll')]:
        model = snapshot(base, primary['fit_id'], primary['epoch'], intervention)
        models[model['id']] = model
        roles[role] = model['id']
    for label, choices in removal_choices.items():
        for policy, row in choices.items():
            model = snapshot(base, row['fit_id'], row['epoch'])
            models[model['id']] = model
            roles['removal_'+label+'_'+policy] = model['id']
    for cue in ['rgb', 'gray']:
        global_control = best_dev([r for r in selection['all_nine_own_Dev_best'] if r['cue'] == cue])
        roles['global_Dev_selected_'+cue] = snapshot(base, global_control['fit_id'], global_control['epoch'])['id']
    roles.update(source_RGB_fresh=FRESH['id'], source_RGB_canonical=CANONICAL['id'],
        matched_RGB=roles['normal_'+selection['primary_variant_RGB']['fit_id']],
        matched_gray=roles['normal_'+selection['primary_variant_gray']['fit_id']])
    return list(models.values()), roles
