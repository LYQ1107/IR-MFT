#!/usr/bin/env python3
"""Register four distinct seed0 fits, matched curves and two image interventions."""
import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pseudo_ir_comparison import dump, sha256


def main():
    phase_path = ROOT/'configs/perceptual_structural_nocycle_20261123.json'
    phase = json.loads(phase_path.read_text())
    main_plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work = Path(main_plan['work_root'])
    assert json.loads((work/'structural_relative_dev_state.json').read_text())['status'] == 'COMPLETE'
    target = ROOT/'configs/perceptual_structural_nocycle_eval_20261123.json'
    folder = ROOT/'configs/perceptual_structural_nocycle_dev_20261123'
    assert not target.exists() and not folder.exists()
    models = []
    for fit in phase['fits']:
        cue, variant = fit['cue'], fit['variant']
        for epoch in phase['snapshots']:
            key = f'NC_{cue}_{variant}_s0_e{epoch:02d}'
            models.append(dict(id=key, cue=cue, fit_variant=variant,
                variant='no_centering' if variant.endswith('no_centering') else 'no_cycle',
                seed=0, epoch=epoch,
                checkpoint=str(work/'structural_nocycle_fit'/f'{cue}_{variant}_s0'/f'epoch{epoch:02d}.pt'),
                stage='matched_controls' if cue != 'ir' else 'module_removals'))
    anchor = dict(cue='ir', fit_variant='no_cycle', variant='no_cycle', seed=0, epoch=6,
                  checkpoint=phase['primary_IR']['checkpoint'], checkpoint_origin='initial_structural_relative')
    models.append(dict(anchor, id='NC_IR_anchor_s0_e06', stage='source_anchor', accuracy_replay=False))
    for key, intervention in [('NC_IR_image_roll_s0_e06', 'IR_image_spatial_roll'),
                              ('NC_IR_unavailable_s0_e06', 'unavailable')]:
        models.append(dict(anchor, id=key, intervention=intervention, stage='IR_interventions'))
    models.append(dict(id='CG3_source_RGB', cue='rgb', baseline=True, stage='source_anchor', accuracy_replay=False))
    spec = dict(status='REGISTERED_BEFORE_NO_CYCLE_REFINEMENT_ACCURACY',
        created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        config_sha256=phase['base_config_sha256'], refinement_plan_sha256=sha256(phase_path),
        single_seed_user_policy_sha256=phase['single_seed_policy_sha256'],
        fixed_seed=0, additional_seed_work=False, models=models,
        stages=dict(matched_controls=16, module_removals=16, IR_interventions=4),
        primary_IR=dict(id='SR_ir_no_cycle_s0_e06_raw', Dev_HOTA=51.329,
                        selection='Frozen from complete initial59Dev, before new matched control accuracy.'),
        retained_completed_references=dict(IR_no_cycle_curves=8, IR_no_structure_curves=8, original_RGB=1),
        counts=dict(loaded_models=20, emitted_streams=39, new_accuracy_replays=36, reused_completed_results=17),
        scope='No additional IR full fit. Reuse completed IR no_cycle/no_structure and canonical RGB results; '
              'report36 new rows plus17 referenced rows. Do not infer novelty or Test gain from Dev.')
    assert len(models) == 20
    dump(target, spec)
    folder.mkdir()
    count = 0
    for row in models:
        if not row.get('accuracy_replay', True):
            continue
        for policy in ['priority', 'raw']:
            key = row['id']+('_raw' if policy == 'raw' else '')
            config = dict(row, id=key, model_id=row['id'], proposal_key=key, output_policy=policy,
                producer_directory='nocycle_refinement/structural_relative_predictions',
                fusion_plan_sha256=spec['config_sha256'], specification_sha256=sha256(target),
                single_seed_user_policy_sha256=spec['single_seed_user_policy_sha256'],
                refinement_plan_sha256=sha256(phase_path),
                registration='Four distinct matched controls/removals; fixedseed0; all4snapshots and bothpolicies. '
                             'Bothsource8/prefix32/fullstream audits before accuracy. Existing17results referenced, not rerun.')
            dump(folder/(key+'.json'), config)
            count += 1
    assert count == 36
    print(json.dumps(dict(specification=str(target), new_replays=count, reused_results=17, fixed_seed=0)))


if __name__ == '__main__':
    main()
