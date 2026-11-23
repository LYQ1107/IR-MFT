#!/usr/bin/env python3
"""Freeze every missing component's paired and Dev-selected formal Test role."""
import datetime
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.models.structural_factorial_repair import FLAGS


def declared_roles(analysis):
    # Negative Dev evidence cannot remove an engineering-passed formal case.
    return [(f'{bits}_{label}',analysis[key][bits])for bits in ['000','001','010','100']
            for label,key in [('paired6','paired6_raw'),('Dev_best4','best4_by_components')]]


def main():
    phase_path=ROOT/'configs/perceptual_structural_factorial_20261123.json'
    phase=json.loads(phase_path.read_text())
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(plan['work_root'])
    docs=ROOT/'docs/fusion_research'
    analysis_path=docs/'PERCEPTUAL_MFT_20261123_STRUCTURAL_FACTORIAL_ANALYSIS.json'
    analysis=json.loads(analysis_path.read_text())
    state_path=work/'structural_factorial_dev_state.json';state=json.loads(state_path.read_text())
    assert state['status']=='COMPLETE'and all(j['status']=='COMPLETE'for j in state['jobs'].values())
    assert analysis['status']=='COMPLETE_SEED0_STRUCTURAL_FACTORIAL_DEV'
    assert analysis['provenance']['complete_state_sha256']==sha256(state_path)
    assert analysis['provenance']['factorial_plan_sha256']==sha256(phase_path)
    assert analysis['primary']['id']=='SR_ir_no_cycle_s0_e06_raw'
    parent_spec_path=ROOT/'configs/perceptual_structural_nocycle_test_20261123.json'
    parent_spec=json.loads(parent_spec_path.read_text())
    parent_report_path=docs/'PERCEPTUAL_MFT_20261123_STRUCTURAL_NOCYCLE_TEST.json'
    assert parent_report_path.exists()
    assert json.loads((work/'structural_nocycle_test_state.json').read_text())['status']=='COMPLETE'
    target=ROOT/'configs/perceptual_structural_factorial_test_20261123.json'
    folder=ROOT/'configs/perceptual_structural_factorial_test_20261123'
    sentinel=work/'structural_factorial_test_selection_frozen.json'
    assert not target.exists()and not folder.exists()and not sentinel.exists()
    models={};cases={};roles={};seen={}
    for label,row in declared_roles(analysis):
        c=row['config'];bits=c['components'];assert bits in ['000','001','010','100']
        name=f"FT_ir_{bits}_s0_e{c['epoch']:02d}"
        models[name]=dict(id=name,cue='ir',components=bits,fit_variant=bits,
            variant='no_cycle'if FLAGS[bits]['centered']else'no_centering',
            epoch=c['epoch'],seed=0,checkpoint=c['checkpoint'])
        identity=(name,c['output_policy'])
        if identity not in seen:
            key='FT_'+label
            cases[key]=dict(id=key,model_id=name,proposal_key=name+('_raw'if c['output_policy']=='raw'else''),
                output_policy=c['output_policy'],cue='ir',components=bits,epoch=c['epoch'],seed=0,
                checkpoint=c['checkpoint'])
            seen[identity]=key
        roles[label]=seen[identity]
    old_phase=json.loads((ROOT/'configs/perceptual_structural_nocycle_20261123.json').read_text())
    anchor=dict(cue='ir',components='111',fit_variant='no_cycle',variant='no_cycle',seed=0,epoch=6,
        checkpoint=old_phase['primary_IR']['checkpoint'],checkpoint_origin='initial_structural_relative')
    models['FCT_primary_IR111']=dict(anchor,id='FCT_primary_IR111')
    models['FCT_IR_unavailable']=dict(anchor,id='FCT_IR_unavailable',intervention='unavailable')
    models['CG3_source_RGB']=dict(id='CG3_source_RGB',cue='rgb',baseline=True,seed=0)
    off='FT_IR_unavailable';fresh='FT_source_RGB_fresh'
    cases[off]=dict(id=off,model_id='FCT_IR_unavailable',proposal_key='FCT_IR_unavailable_raw',
        output_policy='raw',cue='ir',seed=0)
    cases[fresh]=dict(id=fresh,model_id='CG3_source_RGB',proposal_key='CG3_source_RGB',
        output_policy='baseline',cue='rgb',seed=0,baseline=True,fresh_RGB=True)
    roles.update(IR_unavailable=off,source_RGB_fresh=fresh)
    spec=dict(status='FROZEN_FROM_COMPLETE_FACTORIAL_DEV_BEFORE_NEW_FAMILY_TEST',
        created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        config_sha256=phase['base_config_sha256'],factorial_plan_sha256=sha256(phase_path),
        single_seed_user_policy_sha256=phase['single_seed_policy_sha256'],fixed_seed=0,
        models=list(models.values()),cases=list(cases.values()),roles=roles,
        primary_case='PARENT:'+parent_spec['primary_case'],
        parent_primary_case=parent_spec['primary_case'],parent_specification_sha256=sha256(parent_spec_path),
        parent_report_sha256=sha256(parent_report_path),complete_Dev_analysis_sha256=sha256(analysis_path),
        complete_Dev_state_sha256=sha256(state_path),
        counts=dict(comparison_roles=10,unique_accuracy_replays=len(cases),models=len(models),
                    produced_streams=2*(len(models)-1)+1),coverage=dict(sequences=7,frames=15985),
        selection='All4new trained components tested at6/raw and their equally Dev-selected4checkpoint/bothpolicy '
                  'best, regardless Dev signs. Identical roles evaluated once. No Testaccuracy read by this selector; '
                  'parent report is hashed only. Fixed fullIR1116/raw is never replaced by a Testwinner.',
        reuse='Previous fullIR111,three existing paired/best removals,RGB/gray,canonical RGB and image-roll '
              'results reused from complete parent14Test. New primary-source anchors must match parent fullstreams.',
        source_protocol='Same registered2px/.005 matched tail audit,0.6/0.7memberships unchanged,final4frames '
                        'at most1unmatched score<.6. Offraw exact to newfreshRGB. Separate fresh/canonical controls.',
        attribution=phase['attribution'],goal_met=False,supported_innovations=0)
    assert len(roles)==10 and 6<=len(cases)<=10
    dump(target,spec);folder.mkdir()
    for row in cases.values():
        config=dict(row,producer_directory='factorial_observations/structural_relative_predictions',
            fusion_plan_sha256=spec['config_sha256'],specification_sha256=sha256(target),
            factorial_plan_sha256=sha256(phase_path),single_seed_user_policy_sha256=phase['single_seed_policy_sha256'])
        dump(folder/(row['id']+'.json'),config)
    dump(sentinel,dict(status='COMPLETE_FACTORIAL_DEV_SELECTION_FROZEN',complete_Dev_before_Test=True,
        specification_sha256=sha256(target),factorial_plan_sha256=sha256(phase_path),fixed_seed=0,
        primary_unchanged=True,complete_Dev_analysis_sha256=sha256(analysis_path),roles=roles))
    print(json.dumps(dict(status=spec['status'],roles=10,new_Test_cases=len(cases),primary_unchanged=True)))


if __name__=='__main__':main()
