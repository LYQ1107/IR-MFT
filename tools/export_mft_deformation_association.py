#!/usr/bin/env python3
"""Independently check completed association metrics/references/selection."""
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_mft_deformation_association import registered,context
from tools.audit_mft_coverage_recall import parse_official,audit_validation_views
from tools.native_pyramid_experiment import read,frozen_json
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity
from tools.fit_mft_proposal_quality import utc
from tools.fusion_memory_observation import LENGTHS
from tools.fusion_memory_evaluation import validate_mot
from tools.export_mft_coverage_recall import compact


def main():
    r=registered();plan,protocol,work,parent,base=context();e=read(base/'run_r1.execution.json');state=read(base/'state.json')
    assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(e['identity']['PID'])is None
    jobs=[j for phase in state['phases'].values()for j in phase.values()]
    assert all(j['status']=='COMPLETE'and type(j['exit_code'])is int and j['exit_code']==0 and j['argv_identity_observed']and identity(j['pid'])is None for j in jobs)
    views=audit_validation_views(r);tables=fields=streams=cleanup=0
    for split in ['trainval','dev','test']:
        if not(base/'streams'/split).exists():continue
        logical='train'if split=='trainval'else split;lengths=r['validation_lengths']if split=='trainval'else LENGTHS[split]
        for sequence,n in lengths.items():
            m=read(base/'streams'/split/'sequence_manifests'/(sequence+'.json'))
            assert m['source_versions']==r['source_hashes']and m['frames']==n and m['all_past_frames_consumed']==LENGTHS[logical][sequence]
            assert not m['GT_labels_read']and not m['future_frame_input']and not m['past_outputs_rewritten']
            assert m['reference_PP_MOT_exact']and m['publisher_retains_own_main_rows']
            source_meta=read(parent/'inputs'/logical/(sequence+'.json'))
            assert m['source_NN_bank_sha256']==source_meta['source_NN_bank_sha256']
            old=read(parent/'streams'/split/'sequence_manifests'/(sequence+'.json'))
            assert m['streams']['DA_reference']['track_sha256']==old['streams'][r['selected_parent']['id']]['track_sha256']
            for key,v in m['streams'].items():
                assert sha256(v['track_path'])==v['track_sha256'];validate_mot(Path(v['track_path']),n,v['track_rows'])
                aliases=v['main_aliases'];assert len(aliases)==len(set(aliases.values()))
                assert v['publisher_statistics']['frames']==LENGTHS[logical][sequence]
                assert v['original_observed_scores_boxes_and_order_preserved']
                if key=='DA_reference':assert v['source_statistics']['original_association_cost_entries_changed']==0
                streams+=1
        for path in sorted((base/'evaluation'/split).glob('*/metrics.json')):
            v=read(path);csv=path.parent/'trackeval_output/pedestrian_detailed.csv';raw=parse_official(csv,v['official_schema'],lengths)
            assert sha256(csv)==v['official_CSV_sha256']and len(raw)==len(v['per_sequence47'])
            assert {k:raw[-1][k]for k in v['combined47']}==v['combined47']
            assert all(all(a[k]==b[k]for k in b)for a,b in zip(v['per_sequence47'],raw));fields+=47*len(raw);tables+=1
            lines=(path.parent/'trackeval_output/pedestrian_summary.txt').read_text().splitlines();keys=lines[0].split();values=list(map(float,lines[1].split()))
            assert len(keys)==len(values)==39 and all(abs(v['combined47'][k]-x)<=.000501 for k,x in zip(keys,values))
            for x in read(path.parent/'derived_duplicate_cleanup.json')['rows']:
                assert not Path(x['path']).exists()and sha256(x['retained_canonical'])==x['sha256'];cleanup+=x['bytes']
    td=read(base/'complete_Train_Dev.json');val,dev=td['validation_rows'],td['Dev_rows']
    ref=next(v for v in val if v['case']['id']=='DA_reference');dref=next(v for v in dev if v['case']['id']=='DA_reference')
    assert ref['combined47']==r['parent_validation47']and dref['combined47']==r['parent_Dev47']
    gains={v['case']['id']:v['combined47']['HOTA']-ref['combined47']['HOTA']for v in val};assert gains==td['validation_HOTA_gains_over_PP_pp']
    eligible=[v for v in dev if v['case']['id']!='DA_reference'and gains[v['case']['id']]>=r['specification']['minimum_validation_HOTA_gain_over_PP_pp']and v['combined47']['HOTA']>=dref['combined47']['HOTA']-r['specification']['Dev_noninferiority_tolerance_pp']]
    chosen=min(eligible,key=lambda v:(-gains[v['case']['id']],*(-v['combined47'][k]for k in ['HOTA','AssA','IDF1']),v['case']['id']))if eligible else None
    assert td['eligible_ids']==[v['case']['id']for v in eligible]and td['selected_case']==(None if chosen is None else chosen['case'])
    test=None
    if chosen is None:
        assert not(base/'Test_selection_frozen.json').exists()and not(base/'full_Test_results.json').exists()
    else:
        freeze=read(base/'Test_selection_frozen.json');test=read(base/'full_Test_results.json')
        assert freeze['chosen_case']==td['selected_case']==test['selected_case']and not test['Test_used_for_selection']
        assert freeze['complete_Train_Dev_sha256']==sha256(base/'complete_Train_Dev.json')
        selected=next(v for v in test['rows']if v['case']==test['selected_case'])
        assert test['original_RGB_baseline']['official_CSV_sha256']=='c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
        assert test['HOTA_gain_over_original_RGB_pp']==selected['combined47']['HOTA']-29.528970687393972
        assert all(j['started_UTC']>freeze['UTC']for phase,js in state['phases'].items()if phase.startswith('DEFORMATION_TEST_')for j in js.values())
    audit=dict(status='COMPLETE_ASSOCIATION_OFFICIAL_REFERENCE_AND_SELECTION_AUDIT',UTC=utc(),actual_controller_exit0=True,
        actual_related_jobs_exit0=len(jobs),official_tables_reparsed=tables,official47_fields_checked=fields,
        final_MOT_files_checked=streams,duplicate_cleanup_verified_bytes=cleanup,validation_GT_views_reconstructed=views,
        selection_independently_rebuilt=True,own_source_sha256=sha256(Path(__file__)),
        scope='Official47/39, canonical MOT validity/hashes, exact reference, inputs metadata/source hashes, cleanup and selection. Alternative own-main geometry retention is checked by the frozen producer, not by a second independent full association/state replay.',
        goal_met=False,supported_innovations=0)
    frozen_json(base/'independent_results_audit.json',audit)
    doc=ROOT/'docs/fusion_research';directory=doc/'perceptual_20261127_deformation_association';assert not directory.exists();directory.mkdir()
    names=['registered.json','unit_regressions.json','run_r1.supervisor.py','run_r1.execution.json','run_r1.stdout.txt','state.json','engineering.json','complete_Train_Dev.json','independent_results_audit.json']
    names.extend(n for n in ['Test_selection_frozen.json','full_Test_results.json','geometry_convention_audit.json','geometry_convention_audit_r2.json']if(base/n).exists())
    for folder in ['streams','evaluation','logs']:
        for p in sorted((base/folder).rglob('*')):
            if not p.is_file():continue
            if folder=='streams'and 'sequence_manifests'not in p.parts:continue
            if folder=='evaluation'and p.name not in ['metrics.json','pedestrian_detailed.csv','pedestrian_summary.txt','derived_duplicate_cleanup.json']:continue
            names.append(str(p.relative_to(base)))
    assert len(names)==len(set(names));paths=[]
    for name in names:
        published_name='retracted_geometry_probe_r1.json'if name=='geometry_convention_audit.json'else name
        p,target=base/name,directory/published_name;assert p.stat().st_size<2*2**20 and p.suffix not in ['.npz','.pt','.pth']and 'track_results'not in p.parts
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);assert sha256(p)==sha256(target);paths.append(str(target.relative_to(ROOT)))
    report=dict(status='PUBLISHED_COMPLETE_CONFIDENCE_SHAPE_ASSOCIATION_FACTORIAL',UTC=utc(),specification=r['specification'],
        validation_rows=[compact(v)for v in val],Dev_rows=[compact(v)for v in dev],validation_gains_over_PP_pp=gains,
        selected_case=td['selected_case'],Test_rows=[]if test is None else[compact(v)for v in test['rows']],
        Test_gain_over_original_RGB_pp=None if test is None else test['HOTA_gain_over_original_RGB_pp'],
        numeric_plus5_met=False if test is None else test['target_plus5_met'],numeric_plus10_met=False if test is None else test['target_plus10_met'],
        baseline_HOTA=29.528970687393972,minimum_HOTA=34.52897068739397,stretch_HOTA=39.52897068739397,
        independent_audit=audit,geometry_convention_note='In this frozen runtime, all147 equal-size original/candidate-scale cost entries matched exactly. The first probe had a prewritten mismatch narrative contradicted by its own0/0 numeric output; that narrative is retracted in the preserved R2 check. No inference code, candidate, threshold or selection changed.',
        modality_note='Classical geometric hypotheses inherit fixed pseudo-IR visual inputs and the prior paired gate, without new independent IR evidence.',
        historical_Test_exposed=True,strongest_RGB_crossarchitecture_sweeps_deferred=True,no_large_assets=True,goal_met=False,supported_innovations=0)
    root=doc/'PERCEPTUAL_MFT_20261127_DEFORMATION_ASSOCIATION.json';assert not root.exists();root.write_text(json.dumps(report,indent=2)+'\n');paths.append(str(root.relative_to(ROOT)))
    lines=['# Confidence/shape association: completed fixed factorial','',
        'Original RGB-only YOLOX-X+ByteTrack baseline stays29.5289706874 HOTA, full Test7/15,985 frames. Minimum34.5289706874, stretch39.5289706874. Fixed November DiffV2IR, ProtocolV2 and seed0; strongestRGB/crossarchitecture sweeps remain deferred.','',
        'Four fixed cases combine original/candidate-scale matching with original/no score fusion. Candidate-scale matching uses the predicted track center and current candidate width/height at all three matching stages. Original observations, score partitions, birth/update rules, template reliability, trained heads/calibrations/archive and PP publication remain fixed. Changed associations may change future states and identities. Exact-original-state/MOT equivalence applies to the reference; each alternative publisher retains its own main observations.','',
        'Three regressions and six real-Train64-frame checks passed, including literal reference state/MOT and32-frame prefix invariance under future-input corruption. Warm all27,251 Train frames, score its last6,814; full Dev covers4,834. Require validation gain>=0.1 over selected PP and no Dev degradation within1e-10, then rank validation HOTA and Dev HOTA/AssA/IDF1/identifier. Freeze before any new Test. Controls cannot be promoted from Test.','',
        '| Case | Forward Train HOTA | Gain over PP | Dev HOTA |','|---|---:|---:|---:|']
    dv={v['case']['id']:v for v in dev}
    for v in val:
        k=v['case']['id'];lines.append(f"| {k} | {v['combined47']['HOTA']:.9f} | {gains[k]:+.9f} | {dv[k]['combined47']['HOTA']:.9f} |")
    if test is None:lines.extend(['','No candidate qualified; no new Test run occurred. Prior selected PP remains32.4955880280 (+2.9666173406), below+5.'])
    else:
        selected=next(v for v in test['rows']if v['case']==test['selected_case'])
        lines.extend(['',f"Frozen primary **{test['selected_case']['id']}**: full Test HOTA **{selected['combined47']['HOTA']:.9f}**, original-RGB gain **{test['HOTA_gain_over_original_RGB_pp']:+.9f}**. Numeric+5: **{test['target_plus5_met']}**; +10: **{test['target_plus10_met']}**.",'','| Fixed Test case/control | HOTA | DetA | AssA | IDF1 |','|---|---:|---:|---:|---:|'])
        for v in test['rows']:lines.append('| '+v['case']['id']+' | '+' | '.join(f"{v['combined47'][k]:.9f}"for k in ['HOTA','DetA','AssA','IDF1'])+' |')
    lines.extend(['',
        'The supplemental geometry check compares147 equal-size original/candidate-scale cost entries across three sizes and fractional/integer displacements: all match exactly in this runtime. The first probe had an incorrect prewritten convention-mismatch narrative even though both numeric touching-edge overlaps were0. That narrative is retracted, with the original note and corrected numeric check both retained. No inference code, thresholds or candidate selection changed.','',
        f'Independent checks reparse{tables} original276-column CSV tables and{fields} official47 values, validate{streams} canonical MOT streams and reconstruct original references, GT windows, cleanup and prospective selection. All{len(jobs)} jobs and controller have actualexit0. It is not a second visual producer or complete independent alternative-association/state replay.','',
        'No new independent IR information is introduced by these classical geometric variants. They cannot by themselves establish multimodal novelty. Historical Test and prior PP results were exposed; legacy encoder Train identities, inspected development windows and the older RGB detector’s Dev exposure remain disclosed. Generated pseudo-IR is RGB-derived, not physical thermal.','',
        'Only small code/configuration/official aggregate and per-sequence47 values/raw CSV/summary/logs/provenance are published. Canonical MOT, NN banks, images, weights and GT stay private. No new NN training or GPU work; physicalGPU4/5 maximum two if required. New private cap256MiB inside64GiB work cap with40GiB reserve. Cached costs exclude original neural/image/generation costs and are not end-to-end FPS.','',
        '**Goal remains active pending final assessment; independently supported new multimodal innovations:0.**',''])
    md=doc/'PERCEPTUAL_MFT_20261127_DEFORMATION_ASSOCIATION.md';assert not md.exists();md.write_text('\n'.join(lines));paths.append(str(md.relative_to(ROOT)))
    manifest=directory/'publication_manifest.json';manifest.write_text(json.dumps(dict(paths=paths,file_SHA256={n:sha256(ROOT/n)for n in paths},total_bytes=sum((ROOT/n).stat().st_size for n in paths),large_assets_included=False),indent=2)+'\n')
    print(json.dumps(dict(status='EXPORTED_COMPLETE_DEFORMATION_FACTORIAL',paths=len(paths)+1,actual_jobs_exit0=len(jobs),official_tables=tables)))


if __name__=='__main__':main()
