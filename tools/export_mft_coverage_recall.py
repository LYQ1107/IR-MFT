#!/usr/bin/env python3
"""Export lightweight completed recall evidence and all official47 tables."""
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.native_pyramid_experiment import read
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity
from tools.fit_mft_proposal_quality import utc

W=Path('/var/tmp/liuyeqiang_ir_mft_perceptual_20261122')
B=W/'coverage_recall_20261127'
DOC=ROOT/'docs/fusion_research'


def compact(row):
    return {k:row[k]for k in ['case','frames','combined47','per_sequence47','official_CSV','official_CSV_sha256']}


def main():
    e=read(B/'run_r1.execution.json');a=read(B/'independent_results_audit.json');ae=read(B/'audit_r1.execution.json')
    for value in [e,ae]:assert value['actual_child_exit_code']==0 and value['source_unchanged']and identity(value['identity']['PID'])is None
    r=read(B/'registered.json');v=read(B/'complete_forward_Train_validation.json');d=read(B/'complete_Dev.json')
    assert a['controller_execution_sha256']==sha256(B/'run_r1.execution.json')and a['selected_case']==d['selected_case']
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    failure=read(B/'failure_analysis_Train_false_need.json');probe=read(B/'false_need_probe_actual_tool.json')
    assert probe['actual_tool_exit_code']==0 and probe['source_sha256']==sha256(ROOT/'tools/diagnose_mft_coverage_false_need.py')
    assert probe['result_sha256']==sha256(B/'failure_analysis_Train_false_need.json')
    t=read(B/'full_Test_results.json')if(B/'full_Test_results.json').exists()else None
    directory=DOC/'perceptual_20261127_coverage_recall';assert not directory.exists();directory.mkdir()
    names=['registered.json','unit_regressions.json','run_r1.supervisor.py','audit_r1.supervisor.py','state.json',
        'real_Train_engineering.json','validation_GT_view_audit.json','complete_forward_Train_validation.json','complete_Dev.json','independent_results_audit.json',
        'failure_analysis_Train_false_need.json','false_need_probe_actual_tool.json']
    for stem in ['run_r1','audit_r1']:names.extend([stem+'.execution.json',stem+'.stdout.txt'])
    names.extend(n for n in ['Test_selection_frozen.json','complete_Dev_interventions.json','full_Test_results.json','full_Test_interventions.json']if(B/n).exists())
    for folder in ['logs','evaluation','streams']:
        for path in sorted((B/folder).rglob('*')):
            if not path.is_file():continue
            if folder=='streams'and 'sequence_manifests'not in path.parts:continue
            if folder=='evaluation'and not(path.name in ['metrics.json','pedestrian_detailed.csv','pedestrian_summary.txt']):continue
            names.append(str(path.relative_to(B)))
    artifacts=[]
    for name in names:
        source=B/name;target=directory/name;assert source.stat().st_size<2*2**20
        assert source.suffix not in ['.npz','.pt','.pth']and 'track_results'not in source.parts
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);assert sha256(target)==sha256(source)
        artifacts.append(dict(path=str(target.relative_to(ROOT)),bytes=target.stat().st_size,sha256=sha256(target),original_source=str(source)))
    report=dict(status='PUBLISHED_COMPLETE_COVERAGE_RECALL_WITH_PROSPECTIVE_SELECTION',UTC=utc(),
        original_RGB_baseline_HOTA=29.528970687393972,minimum_required_Test_HOTA=34.52897068739397,
        stretch_Test_HOTA=39.52897068739397,validation_rows=[compact(row)for row in v['rows']],
        validation_HOTA_gains_pp=v['HOTA_gains_pp'],Dev_rows=[compact(row)for row in d['rows']],
        selected_case=d['selected_case'],eligible_ids=d['eligible_ids'],selection_rule=d['selection_rule'],
        Test_rows=[]if t is None else[compact(row)for row in t['rows']],
        Test_HOTA_gain_over_original_RGB_pp=None if t is None else t['HOTA_gain_over_original_RGB_pp'],
        target_plus5_met=False if t is None else t['target_plus5_met'],target_plus10_met=False if t is None else t['target_plus10_met'],
        original_RGB_baseline=None if t is None else t['original_RGB_baseline'],
        previous_completed_formal_Test_HOTA=30.717081066539237,previous_completed_formal_gain_pp=1.1881103791452645,
        independent_audit=a,Train_failure_geometry=failure,artifacts=artifacts,exporter_sha256=sha256(Path(__file__)),
        stronger_RGB_cross_architecture_sweeps_deferred=True,Test_used_for_selection=False,
        gate_controls_do_not_remove_frozen_IR_visual_detector_or_prior_quality=True,
        standalone_validation_excludes_late_calibrated_parent_thresholds=True,
        legacy_ReID_encoders_Train_identity_exposure=True,full_Dev_parent_includes_old_Train_calibration=True,
        historical_Test_previously_exposed=True,new_GPU_work=False,
        runtime_scope='Cached CPU replay and official evaluation only; excludes original image IO, neural inference and fixed IR generation. No new end-to-end FPS claim.',
        independent_multimodal_contributions_not_established_by_numeric_gain_alone=True,
        goal_met=False,supported_innovations=0)
    for name,key in [('complete_Dev_interventions.json','Dev_gate_interventions'),('full_Test_interventions.json','Test_gate_interventions')]:
        if(B/name).exists():report[key]=[compact(row)for row in read(B/name)['rows']]
    root=DOC/'PERCEPTUAL_MFT_20261127_COVERAGE_RECALL.json';assert not root.exists();root.write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Coverage-triggered weak recall: complete registered study','',
        'The primary comparator remains the original direct-RGB YOLOX-X + ByteTrack, **29.5289706874 HOTA** over the complete historical Test7 / 15,985 frames. Minimum target **34.5289706874**; stretch target **39.5289706874**. Strongest-RGB and cross-architecture sweeps remain deferred under the user’s revised scope.','',
        'The new method uses the unchanged original teacher’s confirmed unused proposal mass to trigger recall. It retains original detector scores/boxes/order, completes existing primary/secondary and high-score pending associations first, then permits remaining proposals above 0.3 with at least three independent original-score shadow confirmations to start tracks. Recall-born pending tracks can confirm against remaining low detections. Extra tracks can change future candidate graphs. Ordinary thresholding, confirmation and ByteTrack extensions do not receive novelty credit by themselves.','',
        'The prospective standalone validation warms every Train sequence from frame 1 and scores only its last quarter: 6,814 scored frames and 27,251 past/current input frames. It excludes the old late-calibrated birth-quality/archive thresholds. Coverage classifiers and foreground directions were fitted on first-75-percent Train data; old ReID encoders have already seen Train identities. This tests the new controller forward in time, not an unseen encoder/domain. Full Dev uses the old selected BO_paired_q999_c3 parent and its existing Train calibration.','',
        'The selection rule was committed before new HOTA results: eligible paired cases need at least +0.1 standalone-validation HOTA and full Dev HOTA ≥52.11311409153856 (tolerance 1e-10). Exact Dev ties are prospectively allowed. Rank by validation HOTA, then Dev HOTA/AssA/IDF1 and identifier. RGB/duplicated/count/unconditional controls cannot be promoted after Test. The previously completed conditional-quality study remains closed.','',
        '| Case | Standalone validation HOTA | Gain | Full Dev HOTA | Eligible |','|---|---:|---:|---:|---|']
    dev={row['case']['id']:row for row in d['rows']}
    for row in v['rows']:
        name=row['case']['id'];lines.append('| '+name+' | '+f"{row['combined47']['HOTA']:.9f}"+' | '+f"{v['HOTA_gains_pp'][name]:+.9f}"+' | '+f"{dev[name]['combined47']['HOTA']:.9f}"+' | '+str(name in d['eligible_ids'])+' |')
    if t is None:
        lines.extend(['','No paired case satisfied the registered validation/Dev gate. No new Test selection was frozen and no new Test result was produced. The previous completed formal result remains 30.7170810665 (+1.1881103791).'])
    else:
        selected=next(row for row in t['rows']if row['case']==t['selected_case'])
        lines.extend(['',f"Frozen selected case: **{t['selected_case']['id']}**. Full Test7 HOTA **{selected['combined47']['HOTA']:.9f}**, **{t['HOTA_gain_over_original_RGB_pp']:+.9f}** points against original RGB. Minimum +5 target: **{t['target_plus5_met']}**; stretch +10: **{t['target_plus10_met']}**.",'',
            '| Frozen Test case/control | HOTA | DetA | AssA | IDF1 |','|---|---:|---:|---:|---:|'])
        for row in t['rows']:
            lines.append('| '+row['case']['id']+' | '+' | '.join(f"{row['combined47'][key]:.9f}"for key in ['HOTA','DetA','AssA','IDF1'])+' |')
        lines.extend(['','Controls remain controls irrespective of their Test scores. ROI gate RGB/duplicated/roll/missing variants retain the same fixed IR visual detector and old quality/archive. They isolate the new gate; they are not whole-system RGB-only baselines. Missing gate IR is checked for literal equality with the matched RGB gate. The numeric difference alone cannot establish 2–3 independent multimodal contributions.'])
    reference=next(row for row in v['rows']if row['case']['id']=='CR_reference')
    paired=next(row for row in v['rows']if row['case']['id']=='CR_paired_t025')
    lines.extend(['','The conservative paired 0.25 candidate increases standalone validation DetA from '+f"{reference['combined47']['DetA']:.9f}"+' to '+f"{paired['combined47']['DetA']:.9f}"+', while AssA falls from '+f"{reference['combined47']['AssA']:.9f}"+' to '+f"{paired['combined47']['AssA']:.9f}"+'. This is a recall/identity tradeoff, not a positive total HOTA result.','',
        '| Train validation sequence | Reference HOTA | Paired 0.25 HOTA | Difference |','|---|---:|---:|---:|'])
    for first,second in zip(reference['per_sequence47'],paired['per_sequence47']):
        if first['seq']=='COMBINED':continue
        lines.append('| '+first['seq']+' | '+f"{first['HOTA']:.9f}"+' | '+f"{second['HOTA']:.9f}"+' | '+f"{second['HOTA']-first['HOTA']:+.9f}"+' |')
    lines.extend(['',
        'A read-only follow-up chose the worst Train validation sequence, PF-001, then its first three preregistered 0.25 need triggers (frames 1065, 1083, 1306). The unused high candidates retain unknown labels (-1); they are not relabeled as background. Their maximum fractions of area covered by an existing measured fish box are 0.89880944, 0.68543055, and 0.74809882. The first two have IoU below 0.5. Thus IoU-only duplicate checks can miss boxes that mostly depict part of an already measured fish. RGB and fixed pseudo-IR frame 1065 were visually inspected without copying or pushing the images. Three frames support this geometric explanation; they do not independently prove the cause of the full identity loss.','',
        'The next hypothesis is to keep the trusted teacher unchanged, handle unexplained proposals in an independent peripheral stream, and test spatial explanation plus one-to-one identity handoff. This is a proposed next study, not an implemented or validated result in this report. The completed failed study remains closed; it cannot be advanced by changing its gate retroactively.'])
    lines.extend(['',f"Independent audit verified {a['actual_related_jobs_exit0']} real worker exits, reparsed {a['official_tables_reparsed']} official tables ({a['official47_fields_rechecked']} aggregate/per-sequence fields), checked {a['original_MOT_files_checked']} original trajectory hashes/formats, and rebuilt {a['Train_Dev_gate_counts_rebuilt_from_preexisting_independent_diagnostic_scalar_bank']} gate activation counts from the predecessor’s private scalar diagnostic bank. It reconstructed the prospective selection. Its scope excludes a second visual producer, independent Kalman/LAP replay and end-to-end runtime.",'',
        'The original detailed 276-column CSVs and 39-field summaries are preserved byte-for-byte. Source hashes, actual execution receipts, small raw logs, every sequence and all 47 metrics are included. Weights, images, descriptor/scalar banks, validation GT copies and MOT trajectory dumps are excluded. CPU replay/evaluation time excludes original neural/image/generation costs; it must not be reported as end-to-end FPS.','',
        'Fixed November DiffV2IR and ProtocolV2 remain unchanged, seed0 only, at most physical GPU4/5. This study reuses existing banks on CPU. Historical Test exposure, old detector Dev exposure and ReID Train-identity exposure remain disclosed. Generated pseudo-IR is an RGB-derived auxiliary view, not a physical thermal sensor.','',
        '**Goal remains active pending numeric and scientific requirements. Independently supported multimodal innovations: 0.**',''])
    markdown=DOC/'PERCEPTUAL_MFT_20261127_COVERAGE_RECALL.md';assert not markdown.exists();markdown.write_text('\n'.join(lines))
    paths=[str(root.relative_to(ROOT)),str(markdown.relative_to(ROOT))]+[v['path']for v in artifacts]
    manifest=directory/'publication_manifest.json';manifest.write_text(json.dumps(dict(paths=paths,file_SHA256={p:sha256(ROOT/p)for p in paths},
        total_bytes=sum((ROOT/p).stat().st_size for p in paths),large_assets_included=False,goal_met=False),indent=2)+'\n')
    print(json.dumps(dict(status='EXPORTED_COMPLETE_COVERAGE_RECALL',paths=len(paths)+1,total_bytes=sum((ROOT/p).stat().st_size for p in paths),report=str(root))),flush=True)


if __name__=='__main__':main()
