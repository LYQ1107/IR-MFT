#!/usr/bin/env python3
"""Freeze seed0 full-Test choices from completed Dev, without opening Test GT."""
import copy
import datetime
import json
from pathlib import Path

from tools.run_pseudo_ir_comparison import dump, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root']); docs = ROOT/'docs/fusion_research'
    state_path = work/'single_seed_geometry_dev_state.json'
    state = json.loads(state_path.read_text()); assert state['status'] == 'COMPLETE'
    assert len(state['full_both_video_proofs']) == 2
    analysis_path = docs/'PERCEPTUAL_MFT_20261123_SINGLE_SEED_GEOMETRY_ANALYSIS.json'
    analysis = json.loads(analysis_path.read_text()); assert not analysis['parent_removals_pending']
    old_spec_path = ROOT/'configs/perceptual_single_seed_geometry_eval_20261123.json'
    old_spec = json.loads(old_spec_path.read_text()); all_models = {r['id']: r for r in old_spec['models']}
    models = {}; cases = []

    def add(model_id, role, label, group):
        row = copy.deepcopy(all_models[model_id]); assert row.get('seed', 0) == 0
        row['head_roles'] = list(dict.fromkeys(['full', role]))
        row.pop('evaluate_roles', None)
        if model_id in models:
            models[model_id]['head_roles'] = list(dict.fromkeys(models[model_id]['head_roles']+row['head_roles']))
        else:
            models[model_id] = row
        key = model_id+('' if role == 'full' else '_'+role)+'_raw'
        cases.append(dict(id=label, proposal_key=key, model_id=model_id, head_role=role,
                          output_policy='raw', cue=row['cue'], seed=0, group=group))

    for cue in ['rgb', 'gray', 'ir']:
        winner = analysis['best_by_cue'][cue]
        assert winner['policy'] == 'raw'
        add(f'GFX_{cue}_s0', winner['role'], 'CGA_'+cue+'_Dev_selected', 'matched_full_controls')
    add('GFX_ir_s0', 'full', 'CGA_IR_full_fused_readout', 'readout_diagnostic_not_Test_selection')
    for row in analysis['paired_epoch24_and_best_curve_removals']:
        if row['role'] != 'rgb_boxes' or row['policy'] != 'raw':
            continue
        variant = row['variant']
        for source_id, tag in [(row['paired_removal_id'], 'paired24'), (row['best_removal_id'], 'Dev_best4')]:
            model_id = source_id.removesuffix('_rgb_boxes_raw')
            add(model_id, 'rgb_boxes', f'CGA_IR_{variant}_{tag}', 'module_removals')
    for intervention in ['unavailable', 'spatial_roll']:
        add('CG3_IR_'+intervention, 'rgb_boxes', 'CGA_IR_'+intervention, 'IR_interventions')
    source = copy.deepcopy(all_models['CG3_source_RGB']); assert source['baseline']
    models[source['id']] = source
    cases += [dict(id='CGA_source_RGB_fresh', proposal_key=source['id'], baseline=False,
                   cue='rgb', group='numerical_source_controls'),
              dict(id='CGA_source_RGB_canonical', baseline=True, cue='rgb', group='numerical_source_controls')]
    assert len(cases) == 14 and len({r['id'] for r in cases}) == 14
    proofs = {}
    for sequence in ['BT-005', 'SN-013']:
        folder = work/'single_seed_geometry_predictions/dev'/sequence
        metadata = json.loads((folder/'complete.json').read_text())
        assert metadata['source_sha256'] == sha256(ROOT/'tools/extract_single_seed_geometry.py')
        selected = {}
        for row in cases:
            if row.get('baseline'):
                continue
            info = metadata['outputs'][row['proposal_key']]
            assert info['sha256'] == sha256(Path(info['path']))
            selected[row['proposal_key']] = info['sha256']
        proofs[sequence] = dict(complete_manifest_sha256=sha256(folder/'complete.json'),
                               selected_full_Dev_packets=selected)
    target = ROOT/'configs/perceptual_CGA_formal_test_20261123.json'
    cfgdir = ROOT/'configs/perceptual_CGA_formal_test_20261123'
    frozen = work/'correlation_selection_frozen.json'
    assert not target.exists() and not cfgdir.exists() and not frozen.exists()
    registration = dict(status='FROZEN_FROM_COMPLETE_DEV_BEFORE_NEW_TEST',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        config_sha256=old_spec['config_sha256'], models=list(models.values()), cases=cases,
        source_Dev_analysis_sha256=sha256(analysis_path), source_Dev_state_sha256=sha256(state_path),
        source_Dev_specification_sha256=sha256(old_spec_path), full_Dev_source_proofs=proofs,
        primary_case='CGA_ir_Dev_selected', fixed_seed=0, additional_seed_work=False,
        selection='RGB/gray/IR checkpoints were fixed18/18/24 before this factorial. '
                  'Each cue readout is selected using complete Dev only. IR primary is '
                  'RGB NMS/emission with fused confidence. Module removals include same '
                  'epoch24/readout and strongest Dev6/12/18/24 at that same readout. '
                  'All14 Test cases are diagnostics/comparisons; primary may not change after Test.',
        scheduling='Existing unchanged producer on physical2/3 only, after each card '
                   'finishes all structural-relative fits and its full Dev producer. '
                   'Do not run Test accuracy until all59 structural-relative Dev cases COMPLETE.',
        coverage='All7 historical_Test sequences,15985frames, original TrackEval/ByteTrack. '
                 'Actual November DiffV2IR and original RGB files remain unchanged.',
        attribution='Positive seed0 Dev removals do not establish original innovations or Test gains. '
                    'no_counterfactual removes both reliableRGB and wrong-aux losses; '
                    'no_gate removes feature risk/bound and priority, although raw readout excludes priority.',
        numerical='Publish fresh and canonical RGB baselines separately. Full counts and '
                  '0.6/0.7 memberships must match; predeclared diagnostic bounds2pixels/.005score '
                  'allow FP16 batch-size differences. Never call bounded difference bit-exact.',
        Test_read=False, goal_met=False, supported_innovations=0)
    dump(target, registration); cfgdir.mkdir()
    for row in cases:
        config = dict(row, producer_directory='single_seed_geometry_predictions',
            fusion_plan_sha256=registration['config_sha256'], specification_sha256=sha256(target),
            frozen_Dev_analysis_sha256=registration['source_Dev_analysis_sha256'],
            registration='Fixedseed0/frozenDevchoice/all7Test. No Test-driven model/readout selection.')
        dump(cfgdir/(row['id']+'.json'), config)
    dump(frozen, dict(status='COMPLETE_DEV_SELECTION_FROZEN', specification=str(target),
                      specification_sha256=sha256(target), registration=registration))
    print(json.dumps(dict(specification=str(target), cases=len(cases), models=len(models),
                          Test_read=False, new_GPU_work_started=False)))


if __name__ == '__main__':
    main()
