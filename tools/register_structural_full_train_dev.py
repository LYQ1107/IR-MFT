#!/usr/bin/env python3
"""Freeze every full-Train snapshot/control Dev case before training starts."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pseudo_ir_comparison import dump, sha256


def model_row(cue, bits, epoch, work, intervention=None):
    name = f'FT_{cue}_{bits}_s0_e{epoch:02d}'
    if intervention:
        name += '_' + intervention
    result = dict(id=name, cue=cue, components=bits, seed=0, epoch=epoch,
        logical_Train_epoch=epoch // 24,
        variant='no_cycle' if bits[1] == '1' else 'no_centering',
        checkpoint=str(work / 'structural_full_train_fit' / f'{cue}_{bits}_s0' / f'epoch{epoch:02d}.pt'))
    if intervention:
        result['intervention'] = intervention
    return result


def case_row(model, policy, study, spec_sha, producer_directory):
    return dict(model_id=model['id'], proposal_key=model['id'] + ('_raw' if policy == 'raw' else ''),
        cue=model['cue'], components=model['components'], epoch=model['epoch'], policy=policy,
        intervention=model.get('intervention'), baseline=False, seed=0,
        fusion_plan_path=study['configuration'], fusion_plan_sha256=study['configuration_sha256'],
        single_seed_user_policy_sha256=study['single_seed_policy_sha256'],
        full_train_study_sha256=sha256(ROOT / 'configs/perceptual_structural_fulltrain_study_20261123.json'),
        specification_sha256=spec_sha, producer_directory=producer_directory)


def main():
    plan = json.loads((ROOT / 'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    study_path = ROOT / 'configs/perceptual_structural_fulltrain_study_20261123.json'
    study = json.loads(study_path.read_text())
    spec_path = ROOT / 'configs/perceptual_structural_fulltrain_dev_20261123.json'
    assert not spec_path.exists()
    models = [dict(id='CG3_source_RGB', cue='rgb', seed=0, baseline=True)]
    for fit in study['fits']:
        for epoch in study['snapshots']:
            models.append(model_row(fit['cue'], fit['components'], epoch, work))
    for epoch in study['snapshots']:
        for intervention in ['IR_image_spatial_roll', 'unavailable']:
            models.append(model_row('ir', '111', epoch, work, intervention))
    spec = dict(status='REGISTERED_ALL50_DEV_CASES_BEFORE_FULL_TRAIN_FITS', models=models,
        config_sha256=study['configuration_sha256'], full_train_study_sha256=sha256(study_path),
        single_seed_user_policy_sha256=study['single_seed_policy_sha256'],
        source8_outputs=17, trained_outputs=49, nonbaseline_models=24,
        registered_cases=50, fixed_seed=0, additional_seed_work=False,
        scope='All6fits x3logicalTrain checkpoints x2policies=36,12IRinterventions and2RGBbaselines. '
              'All50must complete and source proofs must pass before family accuracy/selection.')
    dump(spec_path, spec)
    configs = ROOT / 'configs/perceptual_structural_fulltrain_dev_cases_20261123'
    configs.mkdir(exist_ok=False)
    producer = 'full_train_observations/structural_relative_predictions'
    for model in models[1:]:
        for policy in ['raw', 'priority']:
            case = case_row(model, policy, study, sha256(spec_path), producer)
            dump(configs / (model['id'] + '_' + policy + '.json'), case)
    for fresh in [False, True]:
        case = dict(baseline=True, fresh_RGB=fresh, seed=0, model_id='CG3_source_RGB',
            proposal_key='CG3_source_RGB', producer_directory=producer,
            fusion_plan_path=study['configuration'], fusion_plan_sha256=study['configuration_sha256'],
            single_seed_user_policy_sha256=study['single_seed_policy_sha256'],
            full_train_study_sha256=sha256(study_path), specification_sha256=sha256(spec_path))
        dump(configs / ('FT_source_RGB_' + ('fresh' if fresh else 'canonical') + '.json'), case)
    print(json.dumps(dict(status=spec['status'], models=len(models), output_streams=49, cases=50)))


if __name__ == '__main__':
    main()
