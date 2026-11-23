#!/usr/bin/env python3
"""Register missing S/C/R component curves before their fitting and accuracy."""
import datetime
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.models.structural_factorial_repair import FLAGS


def main():
    phase_path=ROOT/'configs/perceptual_structural_factorial_20261123.json'
    phase=json.loads(phase_path.read_text())
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work=Path(plan['work_root'])
    parent_path=ROOT/'configs/perceptual_structural_nocycle_20261123.json'
    parent=json.loads(parent_path.read_text())
    analysis=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_STRUCTURAL_NOCYCLE_ANALYSIS.json'
    assert sha256(analysis)==phase['source_Dev_analysis_sha256']
    assert all(j['status']=='PENDING'for j in json.loads((work/'structural_factorial_fit_state.json').read_text())['jobs'].values())
    target=ROOT/'configs/perceptual_structural_factorial_eval_20261123.json'
    folder=ROOT/'configs/perceptual_structural_factorial_dev_20261123'
    assert not target.exists()and not folder.exists()
    models=[]
    for fit in phase['fits']:
        bits=fit['variant']
        for epoch in phase['snapshots']:
            key=f'FCT_ir_{bits}_s0_e{epoch:02d}'
            models.append(dict(id=key,cue='ir',components=bits,fit_variant=bits,
                variant='no_cycle'if FLAGS[bits]['centered']else'no_centering',seed=0,epoch=epoch,
                checkpoint=str(work/'structural_factorial_fit'/f'ir_{bits}_s0'/f'epoch{epoch:02d}.pt'),
                stage='new_components'))
    anchor=dict(cue='ir',components='111',fit_variant='no_cycle',variant='no_cycle',seed=0,epoch=6,
                checkpoint=parent['primary_IR']['checkpoint'],checkpoint_origin='initial_structural_relative',
                accuracy_replay=False,stage='source_anchor')
    models.append(dict(anchor,id='FCT_primary_IR111'))
    models.append(dict(anchor,id='FCT_IR_unavailable',intervention='unavailable'))
    models.append(dict(id='CG3_source_RGB',cue='rgb',baseline=True,accuracy_replay=False,stage='source_anchor'))
    spec=dict(status='REGISTERED_BEFORE_FACTORIAL_FITTING_AND_ACCURACY',
        created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        config_sha256=phase['base_config_sha256'],factorial_plan_sha256=sha256(phase_path),
        single_seed_user_policy_sha256=phase['single_seed_policy_sha256'],fixed_seed=0,
        models=models,stages=dict(new_components=32),
        counts=dict(loaded_models=19,emitted_streams=37,new_accuracy_replays=32,reused_completed_results=49),
        retained_IR_component_codes=['011','101','110','111'],
        primary='SR_ir_no_cycle_s0_e06_raw; unchanged from frozen parent',
        scope='32new component rows,32completed IR rows and17completed RGB/gray/baseline rows=81unique rows. '
              'No new fullIR or modality-control fitting. Single seed0; no Test-based selection.')
    assert len(models)==19
    dump(target,spec);folder.mkdir()
    for row in models:
        if not row.get('accuracy_replay',True):continue
        for policy in ['priority','raw']:
            key=row['id']+('_raw'if policy=='raw'else'')
            config=dict(row,id=key,model_id=row['id'],proposal_key=key,output_policy=policy,
                producer_directory='factorial_observations/structural_relative_predictions',
                fusion_plan_sha256=spec['config_sha256'],specification_sha256=sha256(target),
                single_seed_user_policy_sha256=spec['single_seed_user_policy_sha256'],
                factorial_plan_sha256=sha256(phase_path),
                registration='Missing single-component/static cases;all4snapshots/bothpolicies;'
                             'complete source17/prefix37/full37proofs before all32newaccuracy.')
            dump(folder/(key+'.json'),config)
    assert len(list(folder.glob('*.json')))==32
    print(json.dumps(dict(new_replays=32,reused_results=49,loaded_models=19,emitted_streams=37)))


if __name__=='__main__':main()
