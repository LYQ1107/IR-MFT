#!/usr/bin/env python3
"""Export the complete frozen Test family and all its predeclared comparison roles."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.summarize_pseudo_ir_comparison import detailed_metrics


def main():
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    protocol = json.loads((ROOT/plan['protocol']).read_text())
    spec_path = ROOT/'configs/perceptual_structural_nocycle_test_20261123.json'
    spec = json.loads(spec_path.read_text())
    state_path = work/'structural_nocycle_test_state.json'
    state = json.loads(state_path.read_text())
    assert state['status'] == 'COMPLETE' and all(j['status'] == 'COMPLETE' for j in state['jobs'].values())
    assert state['specification_sha256'] == sha256(spec_path)
    rows = {}
    for case in spec['cases']:
        key = case['id']
        folder = work/'structural_nocycle_test'/key
        path = folder/'metrics.json'
        data = json.loads(path.read_text())
        manifest = data['manifest']
        assert data['status'] == 'COMPLETE' and manifest['frames'] == 15985
        assert manifest['split'] == 'test' and manifest['sequences'] == protocol['historical_test']
        assert manifest['config_sha256'] == state['jobs'][key]['sha256'] == sha256(state['jobs'][key]['config'])
        csv_path = folder/'trackeval_output/pedestrian_detailed.csv'
        per = detailed_metrics(csv_path, protocol['historical_test'], data['trackeval'])
        assert sum(r['CLR_Frames'] for r in per.values()) == 15985
        rows[key] = dict(id=key, config=manifest['config'], metrics=data['trackeval'], per_sequence=per,
            manifest=manifest, metrics_sha256=sha256(path), official_detailed_csv_sha256=sha256(csv_path))
    roles = spec['roles']
    primary = rows[spec['primary_case']]
    canonical, fresh = rows[roles['source_RGB_canonical']], rows[roles['source_RGB_fresh']]
    off, roll = rows[roles['IR_unavailable']], rows[roles['IR_image_roll']]
    assert abs(canonical['metrics']['HOTA']-plan['reference']['rgb_test_HOTA']) < .0005
    assert off['metrics'] == fresh['metrics']
    assert off['official_detailed_csv_sha256'] == fresh['official_detailed_csv_sha256']
    for seq in protocol['historical_test']:
        assert off['manifest']['coverage'][seq]['output_sha256'] == fresh['manifest']['coverage'][seq]['output_sha256']
    controls = [rows[roles['matched_'+cue]] for cue in ['rgb', 'gray']]
    strongest = max(controls, key=lambda r: r['metrics']['HOTA'])
    removals = []
    for r in spec['removals']:
        paired, best = rows[r['paired_id']], rows[r['Dev_best4_id']]
        removals.append(dict(r, primary_minus_paired_HOTA=primary['metrics']['HOTA']-paired['metrics']['HOTA'],
            primary_minus_Dev_best4_HOTA=primary['metrics']['HOTA']-best['metrics']['HOTA']))
    positive = [r['mechanism'] for r in removals
                if min(r['primary_minus_paired_HOTA'], r['primary_minus_Dev_best4_HOTA']) > 0]
    delta = primary['metrics']['HOTA']-canonical['metrics']['HOTA']
    fresh_delta = primary['metrics']['HOTA']-fresh['metrics']['HOTA']
    report = dict(status='COMPLETE_FROZEN_SEED0_STRUCTURAL_NOCYCLE_ALL7_TEST',
        rows=list(rows.values()), roles=roles, counts=spec['counts'], primary=primary['id'],
        primary_HOTA=primary['metrics']['HOTA'], primary_minus_canonical_RGB_HOTA=delta,
        primary_minus_fresh_RGB_HOTA=fresh_delta,
        primary_minus_strongest_matched_RGBgray_HOTA=primary['metrics']['HOTA']-strongest['metrics']['HOTA'],
        image_roll_HOTA_change_from_correct=roll['metrics']['HOTA']-primary['metrics']['HOTA'],
        fresh_minus_canonical_RGB_HOTA=fresh['metrics']['HOTA']-canonical['metrics']['HOTA'],
        canonical_fresh_differing_metrics={k:[v, fresh['metrics'][k]] for k,v in canonical['metrics'].items()
                                          if v != fresh['metrics'][k]},
        removals=removals, positive_paired_and_bestcurve_mechanism_groups=positive,
        numerical_5point_target_met=min(delta, fresh_delta) >= plan['reference']['test_required_improvement_pp'],
        unavailable_tracks_full39metrics_and_officialCSV_exact_to_fresh=True,
        fixed_seed=0, additional_seed_work=False, specification_sha256=sha256(spec_path),
        completed_state_sha256=sha256(state_path),
        source_audits={seq:row['source_proof'] for seq,row in state['producers'].items()},
        scope='Full7historicalTest sequences/15985frames, official combined TrackEval. '
              'Primary IR6/raw was fixed before earlierCGA14Test accuracy; controls/removal choices use complete Dev only. '
              'Identical paired/Dev-best cases are evaluated once and referenced by multiple declared roles. '
              'No bestTest method/checkpoint/seed selection. This is exploratory, not a newly blind Test.',
        attribution=spec['attribution'],
        timing='Cached proposal replay FPS excludes encoder/fusion/image preparation. Shared producer timings are '
               'not individual-method end-to-end performance. IR generation is also excluded.',
        goal_met=False, supported_innovations=0)
    stem = ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_STRUCTURAL_NOCYCLE_TEST'
    dump(stem.with_suffix('.json'), report)
    lines = ['# Frozen no-cycle IR: complete7-sequence Test', '', report['scope'], '',
        '| Frozen case | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows.values():
        m = row['metrics']
        lines.append('| '+row['id']+' | '+' | '.join(f'{m[k]:.3f}' for k in ['HOTA','DetA','AssA','IDF1','MOTA'])+
                     f" | {int(m['IDSW'])} |")
    lines += ['', f"Frozen primary versus canonical RGB: {delta:+.3f}; versus fresh RGB: {fresh_delta:+.3f}; "
        f"versus strongest matched RGB/gray: {report['primary_minus_strongest_matched_RGBgray_HOTA']:+.3f} HOTA.", '',
        '| Removal | Primary − same6/raw | Primary − Dev-selected best4 |',
        '| --- | ---: | ---: |']
    for r in removals:
        lines.append(f"| {r['mechanism']} | {r['primary_minus_paired_HOTA']:+.3f} | {r['primary_minus_Dev_best4_HOTA']:+.3f} |")
    lines += ['', f"Actual-image roll change: {report['image_roll_HOTA_change_from_correct']:+.3f} HOTA. "
        'Unavailable IR reproduces fresh RGB tracks, all39metrics and official detailedCSV exactly.', '',
        report['attribution'], '', 'A numerical threshold or positive removals alone do not establish originality. '
        'The goal remains unmarked until the complete evidence and prior-art boundaries are assessed.', '', report['timing']]
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(primary=primary['id'], HOTA=primary['metrics']['HOTA'],
        delta_canonical_RGB=delta, delta_fresh_RGB=fresh_delta,
        numerical_5point_target_met=report['numerical_5point_target_met'],
        positive_paired_and_bestcurve_groups=positive)))


if __name__ == '__main__':
    main()
