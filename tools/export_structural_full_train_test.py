#!/usr/bin/env python3
"""All frozen full-Train Test roles, without promoting a Test winner."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pseudo_ir_comparison import dump, sha256
from tools.summarize_pseudo_ir_comparison import detailed_metrics


def main():
    plan = json.loads((ROOT / 'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    protocol = json.loads((ROOT / plan['protocol']).read_text())
    spec_path = ROOT / 'configs/perceptual_structural_fulltrain_test_20261123.json'
    specification = json.loads(spec_path.read_text())
    state_path = work / 'structural_full_train_test_state.json'
    state = json.loads(state_path.read_text())
    assert state['status'] == 'COMPLETE' and all(j['status'] == 'COMPLETE' for j in state['jobs'].values())
    assert state['specification_sha256'] == sha256(spec_path)
    analysis_path = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261123_FULL_TRAIN_DEV_ANALYSIS.json'
    assert sha256(analysis_path) == specification['complete_Dev_analysis_sha256']
    analysis = json.loads(analysis_path.read_text())
    dev_rows = {r['id']: r for r in analysis['rows']}
    rows = {}
    for entry in specification['cases']:
        name = entry['id']
        folder = work / 'structural_full_train_test' / name
        path = folder / 'metrics.json'
        data = json.loads(path.read_text())
        manifest = data['manifest']
        assert data['status'] == 'COMPLETE' and manifest['frames'] == 15985
        assert manifest['split'] == 'test' and manifest['sequences'] == protocol['historical_test']
        assert manifest['config_sha256'] == state['jobs'][name]['sha256'] == sha256(state['jobs'][name]['config'])
        old = dev_rows[entry['source_Dev_case']]
        assert old['metrics_sha256'] == entry['source_Dev_metrics_sha256']
        case = manifest['config']
        assert all(case.get(key) == old['config'].get(key) for key in
                   ['model_id', 'policy', 'epoch', 'components', 'cue', 'intervention', 'baseline', 'fresh_RGB'])
        if not case.get('baseline'):
            assert all(info['checkpoint_sha256'] == old['checkpoint_sha256']
                       for info in manifest['model_sources'].values())
        csv_path = folder / 'trackeval_output/pedestrian_detailed.csv'
        per = detailed_metrics(csv_path, protocol['historical_test'], data['trackeval'])
        assert sum(r['CLR_Frames'] for r in per.values()) == 15985
        rows[name] = dict(id=name, source_Dev_case=entry['source_Dev_case'], config=case,
            metrics=data['trackeval'], per_sequence=per, coverage=manifest['coverage'],
            checkpoint_sha256=old['checkpoint_sha256'], proposal_sources=manifest['sources'],
            source_versions=manifest['source_versions'], metrics_sha256=sha256(path),
            official_detailed_csv_sha256=sha256(csv_path))
    roles = specification['roles']
    primary, canonical, fresh = (rows[roles[key]] for key in ['primary_IR','source_RGB_canonical','source_RGB_fresh'])
    assert abs(canonical['metrics']['HOTA'] - 29.529) < .0005
    off, roll = rows[roles['IR_unavailable']], rows[roles['IR_image_roll']]
    expected_RGB = fresh if primary['config']['policy'] == 'raw' else canonical
    assert off['metrics'] == expected_RGB['metrics']
    assert off['official_detailed_csv_sha256'] == expected_RGB['official_detailed_csv_sha256']
    assert all(off['coverage'][seq]['output_sha256'] == expected_RGB['coverage'][seq]['output_sha256']
               for seq in protocol['historical_test'])
    controls = [rows[roles['matched_' + cue]] for cue in ['rgb','gray']]
    strongest = max(controls, key=lambda r:r['metrics']['HOTA'])
    removals = [dict(mechanism=mechanism, paired=roles['no_'+mechanism+'_paired'],
        Dev_best=roles['no_'+mechanism+'_Dev_best'],
        primary_minus_paired_HOTA=primary['metrics']['HOTA']-rows[roles['no_'+mechanism+'_paired']]['metrics']['HOTA'],
        primary_minus_Dev_best_HOTA=primary['metrics']['HOTA']-rows[roles['no_'+mechanism+'_Dev_best']]['metrics']['HOTA'])
        for mechanism in ['S','C','R']]
    delta = primary['metrics']['HOTA'] - canonical['metrics']['HOTA']
    fresh_delta = primary['metrics']['HOTA'] - fresh['metrics']['HOTA']
    report = dict(status='COMPLETE_FROZEN_FULL_TRAIN_ALL7_TEST', rows=list(rows.values()), roles=roles,
        counts=specification['counts'], primary=primary['id'], primary_HOTA=primary['metrics']['HOTA'],
        primary_minus_canonical_RGB_HOTA=delta, primary_minus_fresh_RGB_HOTA=fresh_delta,
        primary_minus_strongest_matched_RGBgray_HOTA=primary['metrics']['HOTA']-strongest['metrics']['HOTA'],
        removals=removals, positive_paired_and_bestcurve_groups=[r['mechanism'] for r in removals
            if min(r['primary_minus_paired_HOTA'],r['primary_minus_Dev_best_HOTA'])>0],
        image_roll_change_HOTA=roll['metrics']['HOTA']-primary['metrics']['HOTA'],
        off_tracks_all39metrics_CSV_exact_to_policy_RGB=True,
        numerical_5point_target_met=min(delta,fresh_delta)>=5,
        canonical_fresh_differing_metrics={k:[v,fresh['metrics'][k]] for k,v in canonical['metrics'].items()
                                          if v!=fresh['metrics'][k]},
        specification_sha256=sha256(spec_path), completed_state_sha256=sha256(state_path),
        complete_Dev_analysis_sha256=sha256(analysis_path),
        source_proofs={seq:row['source_proof'] for seq,row in state['producers'].items()},
        fixed_seed=0, additional_seed_work=False, Test_primary_reselection=False,
        scope='All7historicalTest15985frames for every formal fullTrain configuration, including negative cases. '
              'Choices use all50complete Dev only. Three complete Train traversals/20736updates per fit, '
              'not72full-data epochs. PriorTest/pretrainingDev exposure remains; not newly blind. '
              'All metrics and failed versions retained. Conditional removal deltas cannot be added.',
        goal_met=False, supported_innovations=0,
        scientific_assessment='A numerical threshold or positive removals alone does not establish2-3innovations. '
                              'Manual complete-scope contribution and prior-art assessment is still required.')
    stem = ROOT / 'docs/fusion_research/PERCEPTUAL_MFT_20261123_FULL_TRAIN_TEST'
    assert not stem.with_suffix('.json').exists()
    dump(stem.with_suffix('.json'), report)
    lines=['# Frozen full-Train family: complete7-sequence Test','',report['scope'],'',
           '| Case | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows.values():
        m=row['metrics']
        lines.append('| '+row['id']+' | '+' | '.join(f'{m[k]:.3f}' for k in ['HOTA','DetA','AssA','IDF1','MOTA'])+f" | {int(m['IDSW'])} |")
    lines += ['',f'Frozen primary−canonicalRGB: {delta:+.3f}; primary−freshRGB: {fresh_delta:+.3f}.',
              '', '| Removal | Same epoch/policy | Dev-best removal |','| --- | ---: | ---: |']
    for row in removals:
        lines.append(f"| {row['mechanism']} | {row['primary_minus_paired_HOTA']:+.3f} | {row['primary_minus_Dev_best_HOTA']:+.3f} |")
    lines += ['',report['scientific_assessment'],'','Runtime is separately measured; cached replay FPS excludes encoder/fusion/IR generation.']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=report['status'],HOTA=report['primary_HOTA'],delta_RGB=delta,
                         numerical_target=report['numerical_5point_target_met'])))


if __name__=='__main__':
    main()
