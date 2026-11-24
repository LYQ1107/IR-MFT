#!/usr/bin/env python3
"""Freeze every full-Train formal Test role, including negative Dev removals."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pseudo_ir_comparison import dump, sha256


def declared_roles(analysis):
    roles = dict(primary_IR=analysis['primary'], matched_rgb=analysis['controls']['rgb'],
                 matched_gray=analysis['controls']['gray'])
    for removal in analysis['removals']:
        name = removal['mechanism']
        roles['no_' + name + '_paired'] = removal['paired']
        roles['no_' + name + '_Dev_best'] = removal['Dev_best']
    roles.update(IR_image_roll=analysis['interventions']['IR_image_spatial_roll'],
                 IR_unavailable=analysis['interventions']['unavailable'],
                 source_RGB_fresh='FT_source_RGB_fresh', source_RGB_canonical='FT_source_RGB_canonical')
    assert len(roles) == 13
    return roles


def main():
    plan = json.loads((ROOT / 'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    state = json.loads((work / 'structural_full_train_dev_state.json').read_text())
    assert state['status'] == 'COMPLETE' and len(state['jobs']) == 50
    assert all(j['status'] == 'COMPLETE' for j in state['jobs'].values())
    analysis_path = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261123_FULL_TRAIN_DEV_ANALYSIS.json'
    analysis = json.loads(analysis_path.read_text())
    assert analysis['status'] == 'COMPLETE_ALL50_FULL_TRAIN_DEV_AND_FROZEN_CHOICES'
    assert analysis['completed_Dev_state_sha256'] == sha256(work / 'structural_full_train_dev_state.json')
    study_path = ROOT / 'configs/perceptual_structural_fulltrain_study_20261123.json'
    study = json.loads(study_path.read_text())
    assert analysis['full_train_study_sha256'] == sha256(study_path)
    dev_spec_path = ROOT / 'configs/perceptual_structural_fulltrain_dev_20261123.json'
    dev_models = {r['id']: r for r in json.loads(dev_spec_path.read_text())['models']}
    dev_rows = {r['id']: r for r in analysis['rows']}
    roles = declared_roles(analysis)
    unique, cases, mapped, model_ids = {}, [], {}, {'CG3_source_RGB'}
    configurations = {}
    for role, dev_id in roles.items():
        row = dev_rows[dev_id]
        config = row['config']
        identity = (('canonical_RGB',) if config.get('baseline') and not config.get('fresh_RGB') else
                    ('fresh_RGB',) if config.get('fresh_RGB') else (config['model_id'], config['policy']))
        if identity not in unique:
            identifier = 'FTT_' + role
            unique[identity] = identifier
            cases.append(dict(id=identifier, source_Dev_case=dev_id, source_Dev_metrics_sha256=row['metrics_sha256']))
            configurations[identifier] = copy.deepcopy(config)
            if not config.get('baseline'):
                model_ids.add(config['model_id'])
        mapped[role] = unique[identity]
    assert set(model_ids) <= set(dev_models)
    specification = dict(status='FROZEN_FROM_ALL50_DEV_BEFORE_NEW_FULL_TRAIN_TEST',
        models=[dev_models[name] for name in sorted(model_ids)], cases=cases, roles=mapped,
        config_sha256=study['configuration_sha256'], full_train_study_sha256=sha256(study_path),
        single_seed_user_policy_sha256=study['single_seed_policy_sha256'],
        complete_Dev_analysis_sha256=sha256(analysis_path), primary_case=mapped['primary_IR'],
        primary_source_Dev_case=analysis['primary'], selection_rule=study['selection'],
        source_audit_policy='Same previously registered last4tail protocol: matched boxes<=2pixels,scores<=.005; '
            'strict and nonstrict .6/.7 membership exact; at most1unmatched sub.6 tail candidate. '
            'Not tracking equivalence. Both fresh/canonical controls and exact unavailable checks mandatory.',
        counts=dict(roles=13, unique_cases=len(cases), models=len(model_ids),
                    output_streams=2*(len(model_ids)-1)+1, sequences=7, frames_each=15985),
        mandatory_Test_even_if_Dev_negative=True, Test_accuracy_used_for_selection=False,
        goal_met=False, supported_innovations=0)
    path = ROOT / 'configs/perceptual_structural_fulltrain_test_20261123.json'
    assert not path.exists()
    dump(path, specification)
    directory = ROOT / 'configs/perceptual_structural_fulltrain_test_cases_20261123'
    directory.mkdir(exist_ok=False)
    for identifier, configuration in configurations.items():
        configuration['specification_sha256'] = sha256(path)
        dump(directory / (identifier + '.json'), configuration)
    sentinel = work / 'structural_full_train_Test_selection_frozen.json'
    assert not sentinel.exists()
    dump(sentinel, dict(status='FROZEN_BEFORE_ANY_NEW_FULL_TRAIN_TEST', study_sha256=sha256(study_path),
        complete50Dev_before_Test=True, complete_Dev_analysis_sha256=sha256(analysis_path),
        specification_sha256=sha256(path), roles=mapped, primary=mapped['primary_IR'],
        mandatory_Test_even_if_Dev_negative=True, fixed_seed=0))
    print(json.dumps(dict(status=specification['status'], counts=specification['counts'], primary=specification['primary_case'])))


if __name__ == '__main__':
    main()
