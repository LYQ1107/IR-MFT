#!/usr/bin/env python3
"""Audit new full-Test cases and combine them with validated parent references."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_pseudo_ir_comparison import dump,sha256
from tools.summarize_pseudo_ir_comparison import detailed_metrics
from tools.summarize_structural_factorial import interactions


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    protocol=json.loads((ROOT/plan['protocol']).read_text());work=Path(plan['work_root'])
    docs=ROOT/'docs/fusion_research'
    spec_path=ROOT/'configs/perceptual_structural_factorial_test_20261123.json'
    spec=json.loads(spec_path.read_text())
    state_path=work/'structural_factorial_test_state.json';state=json.loads(state_path.read_text())
    assert state['status']=='COMPLETE'and all(j['status']=='COMPLETE'for j in state['jobs'].values())
    assert state['specification_sha256']==sha256(spec_path)
    parent_path=docs/'PERCEPTUAL_MFT_20261123_STRUCTURAL_NOCYCLE_TEST.json'
    assert sha256(parent_path)==spec['parent_report_sha256']
    parent=json.loads(parent_path.read_text())
    assert parent['status']=='COMPLETE_FROZEN_SEED0_STRUCTURAL_NOCYCLE_ALL7_TEST'
    parent_rows={r['id']:r for r in parent['rows']}
    dev_path=docs/'PERCEPTUAL_MFT_20261123_STRUCTURAL_FACTORIAL_ANALYSIS.json'
    assert sha256(dev_path)==spec['complete_Dev_analysis_sha256']
    dev=json.loads(dev_path.read_text())
    rows={}
    for case in spec['cases']:
        key=case['id'];folder=work/'structural_factorial_test'/key
        path=folder/'metrics.json';data=json.loads(path.read_text());manifest=data['manifest']
        assert data['status']=='COMPLETE'and manifest['frames']==15985
        assert manifest['split']=='test'and manifest['sequences']==protocol['historical_test']
        assert manifest['config_sha256']==state['jobs'][key]['sha256']==sha256(state['jobs'][key]['config'])
        csv=folder/'trackeval_output/pedestrian_detailed.csv'
        per=detailed_metrics(csv,protocol['historical_test'],data['trackeval'])
        assert sum(v['CLR_Frames']for v in per.values())==15985
        rows[key]=dict(id=key,config=manifest['config'],metrics=data['trackeval'],per_sequence=per,
            manifest=manifest,metrics_sha256=sha256(path),official_detailed_csv_sha256=sha256(csv))
    roles=spec['roles'];paired={};best={}
    primary=parent_rows[spec['parent_primary_case']]
    paired['111']=best['111']=primary
    for bits,name in [('011','structure'),('101','centering'),('110','feature_risk')]:
        paired[bits]=parent_rows[parent['roles']['no_'+name+'_paired6']]
        best[bits]=parent_rows[parent['roles']['no_'+name+'_Dev_best4']]
    for bits in ['000','001','010','100']:
        paired[bits]=rows[roles[bits+'_paired6']]
        best[bits]=rows[roles[bits+'_Dev_best4']]
    for bits in paired:
        for test_row,dev_row in [(paired[bits],dev['paired6_raw'][bits]),(best[bits],dev['best4_by_components'][bits])]:
            assert test_row['config']['checkpoint']==dev_row['config']['checkpoint']
            assert test_row['config']['output_policy']==dev_row['config']['output_policy']
    canonical=parent_rows[parent['roles']['source_RGB_canonical']]
    fresh=rows[roles['source_RGB_fresh']];off=rows[roles['IR_unavailable']]
    assert off['metrics']==fresh['metrics']
    assert off['official_detailed_csv_sha256']==fresh['official_detailed_csv_sha256']
    for seq in protocol['historical_test']:
        assert off['manifest']['coverage'][seq]['output_sha256']==fresh['manifest']['coverage'][seq]['output_sha256']
        proof=state['producers'][seq]['source_proof']
        assert proof['primary_IR111_exact_parent_raw']['all4_arrays_exact']
        assert proof['primary_IR111_exact_parent']['all4_arrays_exact']
    full=primary['metrics']['HOTA'];delta=full-canonical['metrics']['HOTA'];fresh_delta=full-fresh['metrics']['HOTA']
    removals=[]
    for bits,name in [('011','structure'),('101','centering'),('110','feature_risk')]:
        removals.append(dict(mechanism=name,primary_minus_paired6_HOTA=full-paired[bits]['metrics']['HOTA'],
            primary_minus_Dev_selected_removal_HOTA=full-best[bits]['metrics']['HOTA']))
    report=dict(status='COMPLETE_FROZEN_SEED0_STRUCTURAL_FACTORIAL_ALL7_TEST',
        rows=list(rows.values()),parent_reference_rows=parent['rows'],roles=roles,
        paired6_raw=paired,Dev_selected_best4=best,primary=primary['id'],primary_HOTA=full,
        primary_minus_canonical_RGB_HOTA=delta,primary_minus_new_fresh_RGB_HOTA=fresh_delta,
        primary_minus_trained_static_paired6_HOTA=full-paired['000']['metrics']['HOTA'],
        primary_minus_strongest_paired_subset_HOTA=full-max(r['metrics']['HOTA']for bits,r in paired.items()if bits!='111'),
        primary_minus_strongest_Dev_selected_subset_HOTA=full-max(r['metrics']['HOTA']for bits,r in best.items()if bits!='111'),
        removals=removals,interaction_decomposition=interactions({b:r['metrics']['HOTA']for b,r in paired.items()}),
        numerical_5point_target_met=min(delta,fresh_delta)>=5.,
        unavailable_tracks_all39metrics_officialCSV_exact_to_new_fresh=True,
        canonical_fresh_differing_metrics={k:[v,fresh['metrics'][k]]for k,v in canonical['metrics'].items()if v!=fresh['metrics'][k]},
        new_rows=len(rows),parent_references=len(parent['rows']),specification_sha256=sha256(spec_path),
        parent_report_sha256=sha256(parent_path),complete_state_sha256=sha256(state_path),
        source_audits={seq:r['source_proof']for seq,r in state['producers'].items()},
        fixed_seed=0,additional_seed_work=False,goal_met=False,supported_innovations=0,
        scope='Full7historicalTest/15985frames. All4new engineering-passed components at paired6/raw and '
              'completeDev-selected best4/bothpolicies. Parent full111and pair removals reused only after '
              'exact full-stream anchor checks; fresh RGB kept separate. No Test-based primary replacement. '
              'HistoricalTest exposed; original RGB pretraining included currentDev.',
        attribution=spec['attribution'],timing='Proposal replay FPS is not end-to-end FPS; see separate single-method runtime reports.')
    stem=docs/'PERCEPTUAL_MFT_20261123_STRUCTURAL_FACTORIAL_TEST'
    dump(stem.with_suffix('.json'),report)
    lines=['# Complete S/C/R full-Test component evidence','',report['scope'],'',
           '| S C R | Paired6/raw HOTA | Dev-selected best4 Test HOTA | Paired IDF1 | Paired DetA | Paired AssA |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for bits in sorted(paired):
        m=paired[bits]['metrics']
        lines.append(f"| {bits} | {m['HOTA']:.3f} | {best[bits]['metrics']['HOTA']:.3f} | "
                     f"{m['IDF1']:.3f} | {m['DetA']:.3f} | {m['AssA']:.3f} |")
    lines+=['',f'Fixed full versus canonical RGB: {delta:+.3f}; versus new freshRGB: {fresh_delta:+.3f} HOTA.',
            '',f"Full minus trained static at6/raw: {report['primary_minus_trained_static_paired6_HOTA']:+.3f}; "
            f"minus strongest Dev-selected subset: {report['primary_minus_strongest_Dev_selected_subset_HOTA']:+.3f}.",
            '',report['interaction_decomposition']['scope'],'',report['attribution'],
            '', 'No goal-completion or originality claim is automatically inferred from this table.']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(primary=primary['id'],HOTA=full,delta_RGB=delta,new_rows=len(rows),
                         numerical_5point_target_met=report['numerical_5point_target_met'])))


if __name__=='__main__':main()
