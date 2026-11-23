#!/usr/bin/env python3
"""Complete component table and nonadditive HOTA effects at the frozen readout."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_pseudo_ir_comparison import dump,sha256


def interactions(values):
    if set(values)!= {'000','001','010','011','100','101','110','111'}:
        raise ValueError('All eight trained component combinations are required')
    v={key:float(value)for key,value in values.items()}
    singles={key:v[key]-v['000']for key in ['100','010','001']}
    pairs={key:v[key]-sum(v[s]for s in parts)+v['000']for key,parts in
           [('110',['100','010']),('101',['100','001']),('011',['010','001'])]}
    third=v['111']-sum(v[k]for k in pairs)+sum(v[k]for k in singles)-v['000']
    total=v['111']-v['000']
    assert abs(sum(singles.values())+sum(pairs.values())+third-total)<1e-8
    return dict(single_component_effects=singles,pair_interactions=pairs,
                three_component_interaction=third,full_minus_trained_static=total,
                scope='Algebraic decomposition of paired6/raw HOTA, not independent additive removal gains '
                      'or a uniquely identified physical causal effect.')


def main():
    docs=ROOT/'docs/fusion_research'
    new_path=docs/'PERCEPTUAL_MFT_20261123_STRUCTURAL_FACTORIAL_DEV.json'
    parent_path=docs/'PERCEPTUAL_MFT_20261123_STRUCTURAL_NOCYCLE_ANALYSIS.json'
    phase_path=ROOT/'configs/perceptual_structural_factorial_20261123.json'
    phase=json.loads(phase_path.read_text());assert sha256(parent_path)==phase['source_Dev_analysis_sha256']
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    state_path=Path(plan['work_root'])/'structural_factorial_dev_state.json'
    state=json.loads(state_path.read_text());assert state['status']=='COMPLETE'
    assert len(state['jobs'])==32 and all(j['status']=='COMPLETE'for j in state['jobs'].values())
    new=json.loads(new_path.read_text())['rows'];parent=json.loads(parent_path.read_text())
    assert len(new)==32
    groups={bits:[r for r in new if r['config']['components']==bits]for bits in ['000','001','010','100']}
    for bits,variant in [('111','no_cycle'),('011','no_structure')]:
        groups[bits]=[r for r in parent['all_retained_reference_rows']if r['config'].get('variant')==variant]
    for bits,variant in [('101','no_cycle_no_centering'),('110','no_cycle_no_risk')]:
        groups[bits]=[r for r in parent['all_new_rows']if r['config'].get('fit_variant')==variant]
    assert all(len(rows)==8 for rows in groups.values())
    controls=[r for r in parent['all_new_rows']if r['config']['cue']in ['rgb','gray']]
    assert len(controls)==16
    baseline=parent['baseline']
    paired,best={},{}
    for bits,rows in groups.items():
        paired[bits]=next(r for r in rows if r['config']['epoch']==6 and r['config']['output_policy']=='raw')
        best[bits]=max(rows,key=lambda r:r['metrics']['HOTA'])
    assert paired['111']==parent['primary']
    full=paired['111'];static=paired['000'];proper=[r for bits,r in paired.items()if bits!='111']
    best_proper=[r for bits,r in best.items()if bits!='111']
    all_rows=[baseline]+controls+[r for bits in sorted(groups)for r in groups[bits]]
    assert len(all_rows)==len({r['id']for r in all_rows})==81
    report=dict(status='COMPLETE_SEED0_STRUCTURAL_FACTORIAL_DEV',primary=full,baseline=baseline,
        groups=groups,paired6_raw=paired,best4_by_components=best,
        matched_controls=parent['best_matched_controls'],
        primary_minus_original_RGB_HOTA=full['metrics']['HOTA']-baseline['metrics']['HOTA'],
        primary_minus_trained_static_paired6_HOTA=full['metrics']['HOTA']-static['metrics']['HOTA'],
        primary_minus_best_paired_subset_HOTA=full['metrics']['HOTA']-max(r['metrics']['HOTA']for r in proper),
        primary_minus_best_subset_curve_HOTA=full['metrics']['HOTA']-max(r['metrics']['HOTA']for r in best_proper),
        interaction_decomposition=interactions({bits:r['metrics']['HOTA']for bits,r in paired.items()}),
        all_rows=all_rows,counts=dict(new_accuracy_replays=32,reused_completed_results=49,total_rows=81),
        provenance=dict(new_family_sha256=sha256(new_path),parent_analysis_sha256=sha256(parent_path),
                        factorial_plan_sha256=sha256(phase_path),complete_state_sha256=sha256(state_path)),
        scope='Full4834frame Dev. Four missing IRcomponent fits,all4snapshots/bothpolicies,seed0. '
              'Previously completed IRfull/pair curves and RGB/gray controls reused. Primary1116/raw unchanged. '
              'Original RGB pretraining included currentDev. No Test-based choice.',
        attribution=phase['attribution'],additional_seed_work=False,goal_met=False,supported_innovations=0)
    stem=docs/'PERCEPTUAL_MFT_20261123_STRUCTURAL_FACTORIAL_ANALYSIS'
    dump(stem.with_suffix('.json'),report)
    lines=['# Complete S/C/R component Dev evidence','',report['scope'],'',
           'S:structure weights/validity/cost confidence; C:semantic centering; R:feature confidence placement.',
           '000 is a trained static local residual, not RGB identity. None of the scalar removal deltas are added.', '',
           '| S C R | Same6/raw HOTA | Best4/policy HOTA | Best case | Same6/raw IDF1 |',
           '| --- | ---: | ---: | --- | ---: |']
    for bits in sorted(groups):
        lines.append(f"| {bits} | {paired[bits]['metrics']['HOTA']:.3f} | {best[bits]['metrics']['HOTA']:.3f} | "
                     f"{best[bits]['id']} | {paired[bits]['metrics']['IDF1']:.3f} |")
    lines+=['',f"Fixed full minus original RGB: {report['primary_minus_original_RGB_HOTA']:+.3f}; "
            f"minus trained static at6/raw: {report['primary_minus_trained_static_paired6_HOTA']:+.3f}; "
            f"minus strongest proper subset curve: {report['primary_minus_best_subset_curve_HOTA']:+.3f} HOTA.",
            '',report['interaction_decomposition']['scope'],'',report['attribution'],
            '', 'These Dev results do not establish the >=+5Test or2–3original-innovation goal.']
    stem.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(primary=full['id'],HOTA=full['metrics']['HOTA'],new_rows=32,referenced_rows=49)))


if __name__=='__main__':main()
