#!/usr/bin/env python3
"""Analyze only completed structural stages, with matched controls/removals."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT/'docs/fusion_research'


def rows(tag):
    return {r['id']: r for r in json.loads((DOCS/f'PERCEPTUAL_MFT_20261123_{tag}.json').read_text())['rows']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-removals', action='store_true')
    parser.add_argument('--include-interventions', action='store_true')
    args = parser.parse_args()
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    state = json.loads((Path(plan['work_root'])/'structural_relative_dev_state.json').read_text())
    assert state['stages']['full_controls']['status'] == 'COMPLETE'
    main_rows = rows('STRUCTURAL_RELATIVE_MAIN_DEV'); assert len(main_rows) == 25
    best = {}
    for cue in ['rgb', 'gray', 'ir']:
        candidates = [r for key, r in main_rows.items() if key.startswith(f'SR_{cue}_full_s0_')]
        assert len(candidates) == 8
        best[cue] = max(candidates, key=lambda r: r['metrics']['HOTA'])
    baseline = main_rows['CG3_source_RGB']
    winner = best['ir']; winner_config = winner['config']
    winner_epoch = winner_config['epoch']; winner_policy = winner_config['output_policy']
    suffix = '_raw' if winner_policy == 'raw' else ''
    removals = []
    if args.include_removals:
        assert state['stages']['module_removals']['status'] == 'COMPLETE'
        ablated = rows('STRUCTURAL_RELATIVE_REMOVALS_DEV'); assert len(ablated) == 24
        for variant in ['no_structure', 'no_cycle', 'no_centering']:
            all_rows = [r for key, r in ablated.items() if key.startswith(f'SR_ir_{variant}_s0_')]
            assert len(all_rows) == 8
            paired = ablated[f'SR_ir_{variant}_s0_e{winner_epoch:02d}'+suffix]
            best_removed = max(all_rows, key=lambda r: r['metrics']['HOTA'])
            removals.append(dict(variant=variant, full_id=winner['id'], paired=paired,
                best_over_4epochs_and_both_policies=best_removed,
                full_minus_paired_HOTA=winner['metrics']['HOTA']-paired['metrics']['HOTA'],
                full_minus_best_removal_HOTA=winner['metrics']['HOTA']-best_removed['metrics']['HOTA']))
    interventions = []
    if args.include_interventions:
        assert state['stages']['IR_interventions']['status'] == 'COMPLETE'
        intervened = rows('STRUCTURAL_RELATIVE_INTERVENTIONS_DEV'); assert len(intervened) == 10
        for key, row in intervened.items():
            cfg = row['config']; epoch = cfg['epoch']
            policy_suffix = '_raw' if cfg['output_policy'] == 'raw' else ''
            parent = main_rows[f'SR_ir_full_s0_e{epoch:02d}'+policy_suffix]
            interventions.append(dict(id=key, intervention=cfg['intervention'], epoch=epoch,
                output_policy=cfg['output_policy'], parent_id=parent['id'], row=row,
                HOTA_change_from_correct_IR=row['metrics']['HOTA']-parent['metrics']['HOTA']))
    control = max([best['rgb'], best['gray']], key=lambda r: r['metrics']['HOTA'])
    report = dict(status='COMPLETE_STRUCTURAL_RELATIVE_ANALYSIS' if args.include_removals and args.include_interventions
                  else 'COMPLETE_CURRENT_STRUCTURAL_RELATIVE_STAGES',
        fixed_seed=0, additional_seed_work=False, all_full_curves=list(main_rows.values()),
        baseline=baseline, best_by_cue=best, strongest_matched_control=control,
        IR_minus_original_RGB_HOTA=winner['metrics']['HOTA']-baseline['metrics']['HOTA'],
        IR_minus_strongest_matched_control_HOTA=winner['metrics']['HOTA']-control['metrics']['HOTA'],
        selected_IR_epoch=winner_epoch, selected_IR_policy=winner_policy,
        module_removals=removals, IR_interventions=interventions,
        pending=dict(module_removals=not args.include_removals, IR_interventions=not args.include_interventions),
        attribution='Two provisional mechanism groups: structural validation (no_structure; '
                    'no_cycle is its nested diagnostic) and relative semantic centering. '
                    'A nested cycle removal is not a third independent innovation. '
                    'Classical self-similarity, RGB geometry copying and response subtraction '
                    'are prior art. No novelty credit or Test gain is inferred from Dev alone.',
        scope='Full4834-frame Dev, official combined metrics; not Test. All4 checkpoints '
              'and both readout policies retained equally for each of6 seed0 fits. '
              'No seed-robustness claim. Original RGB pretraining included current Dev scenes.',
        Test_read=False, goal_met=False, supported_innovations=0)
    stem = DOCS/'PERCEPTUAL_MFT_20261123_STRUCTURAL_RELATIVE_ANALYSIS'
    stem.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Structural validation and relative semantic correction', '', report['scope'], '',
             '| Cue | Dev-selected configuration | HOTA | DetA | AssA | IDF1 | BT HOTA | SN HOTA |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for cue, row in [('original RGB', baseline)]+list(best.items()):
        values = [row['metrics'][k] for k in ['HOTA', 'DetA', 'AssA', 'IDF1']]
        values += [row['per_sequence'][seq]['HOTA'] for seq in ['BT-005', 'SN-013']]
        lines.append('| '+cue+' | '+row['id']+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines += ['', f"IR minus original RGB: {report['IR_minus_original_RGB_HOTA']:+.3f} HOTA; "
              f"IR minus strongest matched control: {report['IR_minus_strongest_matched_control_HOTA']:+.3f} HOTA."]
    if removals:
        lines += ['', '| Removal | Paired epoch/policy | Full − paired | Best removal | Full − best |',
                  '| --- | --- | ---: | --- | ---: |']
        for r in removals:
            lines.append(f"| {r['variant']} | {r['paired']['id']} | {r['full_minus_paired_HOTA']:+.3f} | "
                         f"{r['best_over_4epochs_and_both_policies']['id']} | {r['full_minus_best_removal_HOTA']:+.3f} |")
    if interventions:
        lines += ['', '| Intervention | HOTA | Change from same-epoch/policy correct IR |',
                  '| --- | ---: | ---: |']
        for r in interventions:
            lines.append(f"| {r['id']} | {r['row']['metrics']['HOTA']:.3f} | {r['HOTA_change_from_correct_IR']:+.3f} |")
    lines += ['', report['attribution'], '', 'The original >=+5 Test HOTA and2–3 contribution goal remains unestablished.']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(best_by_cue={k:dict(id=r['id'], HOTA=r['metrics']['HOTA']) for k,r in best.items()},
        IR_minus_strongest_control=report['IR_minus_strongest_matched_control_HOTA'], pending=report['pending'], Test_read=False)))


if __name__ == '__main__':
    main()
