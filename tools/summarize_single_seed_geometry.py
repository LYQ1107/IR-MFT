#!/usr/bin/env python3
"""Read complete single-seed stages; isolate NMS and emitted localization."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT/'docs/fusion_research'
ROLES = ['full', 'rgb_boxes', 'rgb_scores', 'fused_nms_rgb_boxes', 'rgb_nms_fused_boxes']


def report_rows(tag):
    return {row['id']: row for row in json.loads((DOCS/f'PERCEPTUAL_MFT_20261123_{tag}.json').read_text())['rows']}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--include-removals', action='store_true'); args = parser.parse_args()
    work = Path(json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())['work_root'])
    state = json.loads((work/'single_seed_geometry_dev_state.json').read_text())
    assert state['stages']['main_factorial']['status'] == 'COMPLETE'
    old = report_rows('TEMPORAL_GEOMETRY_DEV'); current = report_rows('SINGLE_SEED_GEOMETRY_MAIN_DEV')
    original_interventions = report_rows('CORRELATION_GEOMETRY_DEV')
    assert len(current) == 20 and len(old) == 36
    full = []
    for cue in ['rgb', 'gray', 'ir']:
        for role in ROLES:
            for policy in ['priority', 'raw']:
                suffix = '_raw' if policy == 'raw' else ''
                if role in ROLES[:3]:
                    key = f'TC_{cue}_'+('none' if role == 'full' else role)+suffix
                    row = old[key]
                else:
                    key = f'GFX_{cue}_s0_{role}'+suffix
                    row = current[key]
                full.append(dict(cue=cue, role=role, policy=policy, id=key,
                                 metrics=row['metrics'], per_sequence=row['per_sequence']))
    lookup = {(r['cue'], r['role'], r['policy']): r for r in full}
    best = {cue: max([row for row in full if row['cue'] == cue], key=lambda r: r['metrics']['HOTA'])
            for cue in ['rgb', 'gray', 'ir']}
    factorials = {}
    for cue in ['rgb', 'gray', 'ir']:
        rows = {role: lookup[cue, role, 'raw'] for role in ['full', 'rgb_boxes', 'fused_nms_rgb_boxes', 'rgb_nms_fused_boxes']}
        values = {role: row['metrics']['HOTA'] for role, row in rows.items()}
        factorials[cue] = dict(raw_HOTA=values,
            RGB_emission_change_given_fused_NMS=values['fused_nms_rgb_boxes']-values['full'],
            RGB_NMS_change_given_fused_emission=values['rgb_nms_fused_boxes']-values['full'],
            interaction=values['rgb_boxes']-values['fused_nms_rgb_boxes']-values['rgb_nms_fused_boxes']+values['full'],
            per_sequence={seq: {role: row['per_sequence'][seq]['HOTA'] for role, row in rows.items()}
                          for seq in ['BT-005', 'SN-013']})
    interventions = []
    for intervention in ['unavailable', 'spatial_roll']:
        for role in ROLES:
            for policy in ['priority', 'raw']:
                key = f'CG3_IR_{intervention}'+('' if role == 'full' else '_'+role)+('_raw' if policy == 'raw' else '')
                row = (original_interventions if role in ROLES[:3] else current)[key]
                parent = lookup['ir', role, policy]
                interventions.append(dict(id=key, intervention=intervention, role=role, policy=policy,
                    metrics=row['metrics'], per_sequence=row['per_sequence'],
                    HOTA_change_from_correct_IR=row['metrics']['HOTA']-parent['metrics']['HOTA']))
    removals = []
    if args.include_removals:
        assert state['stages']['parent_removals']['status'] == 'COMPLETE'
        rows = report_rows('SINGLE_SEED_PARENT_REMOVALS_DEV'); assert len(rows) == 120
        for variant in ['aligned', 'no_gate', 'no_counterfactual']:
            for role in ROLES:
                for policy in ['priority', 'raw']:
                    suffix = ('' if role == 'full' else '_'+role)+('_raw' if policy == 'raw' else '')
                    paired = rows[f'GFXM_ir_{variant}_s0_e24'+suffix]
                    best_removed = max([rows[f'GFXM_ir_{variant}_s0_e{epoch:02d}'+suffix]
                        for epoch in [6, 12, 18, 24]], key=lambda r: r['metrics']['HOTA'])
                    parent = lookup['ir', role, policy]
                    removals.append(dict(variant=variant, role=role, policy=policy,
                        full_id=parent['id'], paired_removal_id=paired['id'], paired_removal_metrics=paired['metrics'],
                        full_minus_epoch24_removal_HOTA=parent['metrics']['HOTA']-paired['metrics']['HOTA'],
                        best_removal_id=best_removed['id'], best_removal_metrics=best_removed['metrics'],
                        full_minus_best_4epoch_removal_HOTA=parent['metrics']['HOTA']-best_removed['metrics']['HOTA']))
    report = dict(status='COMPLETE_SINGLE_SEED_GEOMETRY_AND_REMOVALS' if args.include_removals else 'COMPLETE_SINGLE_SEED_GEOMETRY_MAIN',
        scope='Full4834frame Dev, fixed fitted seed0 by explicit user directive. RGB18/gray18/IR24 weights unchanged. '
              'Raw2x2 crosses NMS geometry and emitted geometry while holding fused dense scores fixed. '
              'Priority integration adds another selection step; do not interpret its proposal sets as fixed.',
        full_cue_roles=full, best_by_cue=best, raw_factorials=factorials, IR_feature_interventions=interventions,
        paired_epoch24_and_best_curve_removals=removals, parent_removals_pending=not args.include_removals,
        IR_minus_best_control_HOTA=best['ir']['metrics']['HOTA']-max(best[c]['metrics']['HOTA'] for c in ['rgb', 'gray']),
        attribution='Coordinate copying/NMS/classical scoring alone is not an original multimodal mechanism. '
                    'no_counterfactual removes BOTH reliableRGBdistillation andwrong-auxiliary regularization. '
                    'no_gate removes feature risk/bound andfinalpriority integration;339gateweights remainallocatedbutunused. '
                    'Each removal has all4epochs; paired comparisons above hold the readout/epoch24/policy fixed.',
        Test_read=False, goal_met=False, supported_innovations=0)
    stem = DOCS/'PERCEPTUAL_MFT_20261123_SINGLE_SEED_GEOMETRY_ANALYSIS'
    stem.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Single-seed NMS / emitted geometry diagnosis', '', report['scope'], '',
             '| Cue | Role | Policy | HOTA | DetA | AssA | IDF1 | BT HOTA | SN HOTA |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in full:
        lines.append('| '+row['cue']+' | '+row['role']+' | '+row['policy']+' | '+
            ' | '.join(f"{row['metrics'][k]:.3f}" for k in ['HOTA', 'DetA', 'AssA', 'IDF1'])+
            ' | '+' | '.join(f"{row['per_sequence'][seq]['HOTA']:.3f}" for seq in ['BT-005', 'SN-013'])+' |')
    if removals:
        lines += ['', '| Removal | Role | Policy | Full − epoch24 removal HOTA | Full − best4epoch removal HOTA |',
                  '| --- | --- | --- | ---: | ---: |']
        for row in removals:
            lines.append(f"| {row['variant']} | {row['role']} | {row['policy']} | "
                f"{row['full_minus_epoch24_removal_HOTA']:+.3f} | {row['full_minus_best_4epoch_removal_HOTA']:+.3f} |")
    lines += ['', report['attribution'], '', 'No new Test or >=+5 Test HOTA /2–3contribution claim. '
              'All intervention metrics, per-sequence values and exact2x2 differences are in the adjacent JSON.']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(best_by_cue={cue:dict(id=row['id'], HOTA=row['metrics']['HOTA']) for cue,row in best.items()},
        IR_minus_best_control_HOTA=report['IR_minus_best_control_HOTA'], removals_complete=bool(removals), Test_read=False)))


if __name__ == '__main__':
    main()
