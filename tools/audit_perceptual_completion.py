#!/usr/bin/env python3
"""Audit remaining scope and verify the three uncovered initial structural fits.

This tool reads completed artifacts only. It does not choose a new primary,
launch inference, alter a running pipeline or claim the research goal complete.
"""
import datetime
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from tools.run_pseudo_ir_comparison import dump,sha256


def main():
    plan_path=ROOT/'configs/perceptual_mft_20261122.json'
    plan=json.loads(plan_path.read_text())
    work=Path(plan['work_root'])
    docs=ROOT/'docs/fusion_research'
    protocol_path=ROOT/plan['protocol']
    assert sha256(plan_path)=='388e636d45840f3da8eae44cde2efc4731eeceadc7c169a39f19bf30488ecc03'
    assert sha256(protocol_path)=='fe3db8e991facfcd74981f76aa5c2271c18a9554386501eb2c2dd30a12bf6c60'
    covered=set()
    existing_specs=['perceptual_CGA_formal_test_20261123.json',
        'perceptual_structural_nocycle_test_20261123.json','perceptual_structural_factorial_test_20261123.json']
    for name in existing_specs:
        for model in json.loads((ROOT/'configs'/name).read_text())['models']:
            if model.get('checkpoint'):
                covered.add(str(Path(model['checkpoint']).parent))
    initial=[]
    for path in sorted((work/'structural_relative_fit').glob('*/complete.json')):
        record=json.loads(path.read_text())
        assert record['status']=='COMPLETE_TRAIN_STRUCTURAL_RELATIVE' and record['epochs']==24
        initial.append(dict(fit=path.parent.name,complete_sha256=sha256(path),
            represented_in_three_completed_Test_specifications=str(path.parent)in covered))
    missing=[row['fit'] for row in initial if not row['represented_in_three_completed_Test_specifications']]
    assert set(missing)=={'rgb_full_s0','gray_full_s0','ir_no_centering_s0'}
    analysis_path=docs/'PERCEPTUAL_MFT_20261123_STRUCTURAL_RELATIVE_ANALYSIS.json'
    analysis=json.loads(analysis_path.read_text())
    removal=[row for row in analysis['module_removals']if row['variant']=='no_centering']
    assert len(removal)==1
    removal_best=removal[0]['best_over_4epochs_and_both_policies']
    removal_rows=json.loads((docs/'PERCEPTUAL_MFT_20261123_STRUCTURAL_RELATIVE_REMOVALS_DEV.json').read_text())['rows']
    paired=next(row for row in removal_rows if row['id']=='SR_ir_no_centering_s0_e06_raw')
    choices=[('legacy_matched_RGB',analysis['best_by_cue']['rgb']),
             ('legacy_matched_gray',analysis['best_by_cue']['gray']),
             ('legacy_no_centering_paired6',paired),('legacy_no_centering_Dev_best',removal_best)]
    cases=[]
    for role,row in choices:
        config=row['config']
        checkpoint=Path(config['checkpoint'])
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['config_sha256']=='909eb6fae97ce7eb64c79a31ee93e76d713003c51cc7d9c13dbd276cf49519fd'
        assert (saved['metadata']['cue'],saved['metadata']['variant'],saved['metadata']['seed'])==(config['cue'],config['variant'],0)
        assert saved['completed_epochs']==config['epoch'] and sum(v.numel()for v in saved['adapter'].values())==145600
        assert all(torch.isfinite(v).all()for v in saved['adapter'].values())
        cases.append(dict(role=role,source_Dev_case=row['id'],source_Dev_HOTA=row['metrics']['HOTA'],
            source_Dev_metrics_sha256=row['metrics_sha256'],checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
            cue=config['cue'],variant=config['variant'],epoch=config['epoch'],policy=config['output_policy'],
            seed=0,trainable_parameters=145600,expected_hard_cycle=True))
    backlog=dict(status='VERIFIED_TRAINED_FORMAL_CHECKPOINTS_TEST_BACKFILL_PENDING',cases=cases,
        fits=missing,initial_six_fit_inventory=initial,source_Dev_analysis_sha256=sha256(analysis_path),
        selection='Complete original59Dev only; same6/raw and Dev-best centering removal. '
                  'This closes missing formal coverage, not Test-based primary promotion.',
        data_setting='Actual unchanged NovemberDiffV2IR, not historical EMAauxiliary.',
        required_Test_sequences=7,required_Test_frames_each=15985,
        required_source_proof='Before reusing restored-cycleIR reference, both raw/priority must exactly reproduce '
            'parentNT_ir_full_s0_e06 on all15985frames. Keep canonical/freshRGB and unavailable controls.',
        GPU_policy='Physical4/5 only; do not interrupt or compete with current fullTrain pipeline. '
                   'Backfill implementation/execution remains pending; this registry itself is not Test evidence.',
        fixed_seed=0,additional_seed_work=False,goal_met=False,supported_innovations=0)
    backlog_path=ROOT/'configs/perceptual_structural_legacy_backfill_20261124.json'
    assert not backlog_path.exists()
    dump(backlog_path,backlog)
    audit20=json.loads((docs/'OPEN_SOURCE_20_CODE_AUDIT.json').read_text())
    assert len(audit20)==20 and all(f['status']==200 for row in audit20 for f in row['files'])
    old_temporal=json.loads((docs/'PHASE_VI_F3_RESULTS.json').read_text())
    assert old_temporal['auxiliary_mode']=='causal_cache'
    current=json.loads((docs/'PERCEPTUAL_MFT_20261123_STRUCTURAL_FACTORIAL_TEST.json').read_text())
    report=dict(status='INCOMPLETE_GOAL_REQUIREMENT_AUDIT',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Read artifacts and completed checkpoints; no new Dev/Test metric evaluation or primary choice.',
        goal_definition_sha256=sha256(ROOT/'configs/perceptual_research_goal_20261123.json'),
        numerical_goal=dict(required_HOTA=34.529,current_completed_primary_HOTA=current['primary_HOTA'],
            baseline_HOTA=29.529,current_primary_minus_RGB=current['primary_minus_canonical_RGB_HOTA'],status='NOT_MET'),
        multimodal_contributions=dict(status='NOT_ESTABLISHED',supported=0,
            evidence='Completed factorial Test does not retain paired-and-best positive removal evidence.'),
        architecture_scope=dict(status='INCOMPLETE_INVENTORY_FOR_CURRENT_IR',
            verified_current_visual_families=['local cosine correlation repair','MIND structural relative repair'],
            historical_F1_F2_F3='Implemented/trained/evaluated with EMAcausal_cache, not NovemberDiffV2IR; do not count those Test results as current-modality validation.',
            conditional_F4='Fusion-aware memory scope/exclusion still needs explicit evidence and status; do not count generic RGB memory as multimodal innovation.',
            eight_SCR_combinations_are_one_family=True),
        formal_Test_coverage=dict(status='INCOMPLETE',verified_initial_structural_missing_fits=missing,
            backfill_registry_sha256=sha256(backlog_path),other_formal_native_metric_families_still_need_inventory=True,
            no_future_seed1_2_replication='Latest explicit user seed0 policy takes precedence; retain already finished evidence.'),
        source_audit=dict(records=20,all_recorded_source_fetches_HTTP200=True,
            code_and_commit_hashes_present=True,license_entries_missing=[dict(id=r['id'],name=r['name'])for r in audit20 if not r['license']],
            scope='Audit records inspected; missing license records are not a claim that the repositories have no license.'),
        current_fullTrain=dict(status=json.loads((work/'structural_full_train_fit_state.json').read_text())['status'],
            registered_fit_count=6,registered_logical_Train_epochs_each=3,checkpoints_or_new_HOTA_not_yet_complete=True,
            current_pipeline_pinned_code_not_modified=True),
        remaining=['Complete current fullTrain, all50Dev, frozen choices/runtime and allformalTest.',
            'Implement/execute verified missing formal backfill with exact source proofs.',
            'Inventory remaining formal seed0 candidates and current-IR architecture requirements; avoid counting ablations or old-auxiliary results.',
            'Assess2-3contributions, matchedcontrols, inputinterventions, priorart and numerical5pointtarget from complete evidence.',
            'Replace historical-final-report ambiguity and finish all original final-report fields without claiming a newly blind Test.'],
        source_sha256=sha256(__file__),completion_claim_permitted=False,goal_met=False,supported_innovations=0)
    destination=docs/'PERCEPTUAL_MFT_20261124_REQUIREMENT_AUDIT.json'
    assert not destination.exists()
    dump(destination,report)
    print(json.dumps(dict(status=report['status'],missing_formal_fits=missing,verified_checkpoints=len(cases),
        legacy_auxiliary=old_temporal['auxiliary_mode'],current_primary_HOTA=current['primary_HOTA'])))


if __name__=='__main__':
    main()
