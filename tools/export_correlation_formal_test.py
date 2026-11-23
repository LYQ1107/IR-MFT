#!/usr/bin/env python3
"""Read all frozen14 Test cases together, after full coverage/provenance checks."""
import json
from pathlib import Path

from tools.run_pseudo_ir_comparison import dump, sha256
from tools.summarize_pseudo_ir_comparison import detailed_metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root']); protocol = json.loads((ROOT/plan['protocol']).read_text())
    spec_path = ROOT/'configs/perceptual_CGA_formal_test_20261123.json'
    spec = json.loads(spec_path.read_text())
    state_path = work/'CGA_formal_test_state.json'; state = json.loads(state_path.read_text())
    assert state['status'] == 'COMPLETE' and len(state['jobs']) == 14
    assert all(j['status'] == 'COMPLETE' for j in state['jobs'].values())
    assert state['specification_sha256'] == sha256(spec_path)
    records = {}
    for case in spec['cases']:
        key = case['id']; job = state['jobs'][key]
        folder = work/'CGA_formal_test'/key; path = folder/'metrics.json'
        data = json.loads(path.read_text()); manifest = data['manifest']
        assert data['status'] == 'COMPLETE' and manifest['frames'] == 15985
        assert manifest['split'] == 'test' and manifest['sequences'] == protocol['historical_test']
        assert manifest['config_sha256'] == job['sha256'] == sha256(Path(job['config']))
        per = detailed_metrics(folder/'trackeval_output/pedestrian_detailed.csv',
                               protocol['historical_test'], data['trackeval'])
        assert sum(row['CLR_Frames'] for row in per.values()) == 15985
        records[key] = dict(id=key, config=manifest['config'], metrics=data['trackeval'],
            per_sequence=per, manifest=manifest, metrics_sha256=sha256(path),
            official_detailed_csv_sha256=sha256(folder/'trackeval_output/pedestrian_detailed.csv'))
    canonical = records['CGA_source_RGB_canonical']; fresh = records['CGA_source_RGB_fresh']
    assert abs(canonical['metrics']['HOTA']-plan['reference']['rgb_test_HOTA']) < .0005, 'Canonical baseline failed reproduction'
    off = records['CGA_IR_unavailable']
    assert off['metrics'] == fresh['metrics']
    assert off['official_detailed_csv_sha256'] == fresh['official_detailed_csv_sha256']
    for sequence in protocol['historical_test']:
        assert off['manifest']['coverage'][sequence]['output_sha256'] == fresh['manifest']['coverage'][sequence]['output_sha256']
    primary = records[spec['primary_case']]
    matched = [records['CGA_'+cue+'_Dev_selected'] for cue in ['rgb', 'gray']]
    control = max(matched, key=lambda r: r['metrics']['HOTA'])
    removals = []
    for variant in ['aligned', 'no_gate', 'no_counterfactual']:
        paired = records[f'CGA_IR_{variant}_paired24']; best = records[f'CGA_IR_{variant}_Dev_best4']
        removals.append(dict(variant=variant, paired24_id=paired['id'], Dev_selected_best4_id=best['id'],
            primary_minus_paired24_HOTA=primary['metrics']['HOTA']-paired['metrics']['HOTA'],
            primary_minus_Dev_selected_best4_HOTA=primary['metrics']['HOTA']-best['metrics']['HOTA']))
    delta = primary['metrics']['HOTA']-canonical['metrics']['HOTA']
    fresh_delta = primary['metrics']['HOTA']-fresh['metrics']['HOTA']
    report = dict(status='COMPLETE_FROZEN_SEED0_CGA_ALL7_TEST', rows=list(records.values()),
        primary=primary['id'], primary_HOTA=primary['metrics']['HOTA'],
        primary_minus_canonical_RGB_HOTA=delta, primary_minus_fresh_RGB_HOTA=fresh_delta,
        primary_minus_strongest_matched_RGBgray_HOTA=primary['metrics']['HOTA']-control['metrics']['HOTA'],
        fresh_minus_canonical_RGB_HOTA=fresh['metrics']['HOTA']-canonical['metrics']['HOTA'],
        module_removals=removals, unavailable_full39_metrics_tracks_and_detailedCSV_exact_to_fresh=True,
        numerical_5point_target_met=min(delta, fresh_delta) >= plan['reference']['test_required_improvement_pp'],
        fixed_seed=0, additional_seed_work=False, specification_sha256=sha256(spec_path),
        completed_state_sha256=sha256(state_path),
        source_audit_recovery=dict(
            plan_sha256=state.get('recovery_plan_sha256'),
            source_sha256=state.get('recovery_source_sha256'),
            original_failure_archive=state.get('original_source_failure_archive'),
            per_sequence={k: v.get('recovered_source_audit', v.get('source_audit'))
                          for k, v in state['producers'].items()},
            limitation='The original exact-count source audit failed on one weak NMS candidate at the MSK-003 tail. '
                       'Failure retained and tolerance amendment recorded before any14Test accuracy. '
                       'Matched numeric bounds and high/birth memberships were preserved. '
                       'Fresh and canonical RGB are separate controls; low-confidence tracking equivalence is not assumed.'),
        scope='Full7 historical Test sequences,15985 frames, official combined TrackEval. '
              'All14 cases were frozen from complete Dev. No Test-based checkpoint/readout selection. '
              'Historic Test was previously exposed; this remains exploratory research.',
        attribution=spec['attribution'],
        timing='Cached ByteTrack replay FPS excludes detector/encoder/adapter preparation. '
               'Shared pixel production time is not per-method end-to-end FPS.',
        goal_met=False, supported_innovations=0)
    stem = ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_CGA_FORMAL_TEST'
    dump(stem.with_suffix('.json'), report)
    lines = ['# Frozen seed0 CGA: all7 Test sequences', '', report['scope'], '',
             '| Frozen case | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in records.values():
        m = row['metrics']
        lines.append('| '+row['id']+' | '+' | '.join(f'{m[k]:.3f}' for k in ['HOTA','DetA','AssA','IDF1','MOTA'])+f" | {int(m['IDSW'])} |")
    lines += ['', f"Frozen primary versus canonical RGB: {delta:+.3f} HOTA; versus fresh RGB: {fresh_delta:+.3f}; "
              f"versus strongest matched RGB/gray: {report['primary_minus_strongest_matched_RGBgray_HOTA']:+.3f}.", '',
              '| Removal | Full − paired24 HOTA | Full − Dev-selected best4 HOTA |',
              '| --- | ---: | ---: |']
    for row in removals:
        lines.append(f"| {row['variant']} | {row['primary_minus_paired24_HOTA']:+.3f} | {row['primary_minus_Dev_selected_best4_HOTA']:+.3f} |")
    lines += ['', report['attribution'], '', report['timing'], '', report['source_audit_recovery']['limitation'], '',
              'Reaching a numeric threshold alone does not establish2–3 original multimodal contributions. '
              'Module support, strongest controls and prior-art boundaries require explicit interpretation.']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(primary=primary['id'], HOTA=primary['metrics']['HOTA'],
        delta_RGB=delta, delta_fresh_RGB=fresh_delta,
        numerical_target_met=report['numerical_5point_target_met'], Test_cases=14)))


if __name__ == '__main__':
    main()
