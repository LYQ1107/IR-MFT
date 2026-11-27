#!/usr/bin/env python3
"""Publish completed lightweight protected-peripheral evidence, all47 and failures."""
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.mft_protected_peripheral_experiment import registered,locations
from tools.export_mft_coverage_recall import compact
from tools.native_pyramid_experiment import read
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity
from tools.fit_mft_proposal_quality import utc


def main():
    r=registered();plan,protocol,work,base,previous=locations();e=read(base/'run_r1.execution.json');ae=read(base/'audit_r2.execution.json')
    for x in [e,ae]:assert x['actual_child_exit_code']==0 and x['source_unchanged']and identity(x['identity']['PID'])is None
    a=read(base/'independent_results_audit.json');assert a['controller_execution_sha256']==sha256(base/'run_r1.execution.json')
    val=read(base/'complete_forward_Train_validation.json');dev=read(base/'complete_Dev.json')
    test=read(base/'full_Test_results.json')if(base/'full_Test_results.json').exists()else None
    diagnostic=read(base/'complete_source_diagnostic.json');assert a['selected_case']==dev['selected_case']
    doc=ROOT/'docs/fusion_research';directory=doc/'perceptual_20261127_protected_peripheral';assert not directory.exists();directory.mkdir()
    names=['registered.json','unit_regressions.json','run_r1.supervisor.py','audit_r1.supervisor.py','audit_r2.supervisor.py','audit_r1_failure_preserved.json','state.json',
        'run_r1.execution.json','run_r1.stdout.txt','audit_r1.execution.json','audit_r1.stdout.txt','audit_r2.execution.json','audit_r2.stdout.txt',
        'real_Train_engineering.json','source_Train_Dev_ready.json','complete_source_diagnostic.json',
        'complete_forward_Train_validation.json','complete_Dev.json','independent_results_audit.json']
    names.extend(n for n in ['Test_selection_frozen.json','complete_Dev_interventions.json','source_Test_ready.json','full_Test_results.json','full_Test_interventions.json']if(base/n).exists())
    for folder in ['inputs','streams','evaluation','logs']:
        for path in sorted((base/folder).rglob('*')):
            if not path.is_file():continue
            if folder=='inputs'and path.suffix!='.json':continue
            if folder=='streams'and 'sequence_manifests'not in path.parts:continue
            if folder=='evaluation'and path.name not in ['metrics.json','pedestrian_detailed.csv','pedestrian_summary.txt','derived_duplicate_cleanup.json']:continue
            names.append(str(path.relative_to(base)))
    assert len(names)==len(set(names));artifacts=[]
    for name in names:
        source=base/name;target=directory/name
        assert source.stat().st_size<2*2**20 and source.suffix not in ['.npz','.pt','.pth']and 'track_results'not in source.parts
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);assert sha256(source)==sha256(target)
        artifacts.append(dict(path=str(target.relative_to(ROOT)),bytes=target.stat().st_size,sha256=sha256(target),original_source=str(source)))
    report=dict(status='PUBLISHED_COMPLETE_REGISTERED_PROTECTED_PERIPHERAL_STUDY',UTC=utc(),
        original_RGB_baseline_HOTA=29.528970687393972,minimum_Test_HOTA=34.52897068739397,stretch_Test_HOTA=39.52897068739397,
        validation_rows=[compact(v)for v in val['rows']],validation_HOTA_gains_pp=val['HOTA_gains_pp'],
        Dev_rows=[compact(v)for v in dev['rows']],eligible_ids=dev['eligible_ids'],selected_case=dev['selected_case'],
        Test_rows=[]if test is None else[compact(v)for v in test['rows']],
        Test_HOTA_gain_over_original_RGB_pp=None if test is None else test['HOTA_gain_over_original_RGB_pp'],
        target_plus5_met=False if test is None else test['target_plus5_met'],target_plus10_met=False if test is None else test['target_plus10_met'],
        original_RGB_baseline=None if test is None else test['original_RGB_baseline'],
        previously_completed_formal_Test_method_HOTA=30.717081066539237,previous_formal_gain_pp=1.1881103791452645,
        source_diagnostic=diagnostic,independent_audit=a,artifacts=artifacts,
        execution_correction='R1 missing mot_line import caused actual engineering/controllerexit1 before sources/HOTA; immutable failure/source snapshots and R2 correction already published. Algorithm/cases/thresholds unchanged.',
        audit_correction='Original audit had actualexit1 because it expected wrapper-only statistics absent from the plain publisher manifest. The original94dbcf auditor and actual failure remain preserved; the separate R2 auditor independently checks final MOT projections and full source-main hashes. No inference/result/source bank was changed.',
        trusted_contract='Frozen trusted parent produces every current main row before publisher. Independent source processes exit before formal publishers; source/publisher interfaces contain no mutable teacher. Every main geometry/score and immutable injective alias is audited from final streams. Actual hidden-state comparison is limited to real prefixes and PF1330 engineering, not claimed as a second full Kalman replay.',
        gate_modality_scope='Only new gate IR is rolled/missing/RGB/duplicated; fixed IR visual detector and prior quality/archive stay common. Whole-system IR absence and same-budget gray require separate evidence.',
        standalone_scope='Heads/directions fit first75percent; final25percent forward replay is development data previously inspected in earlier studies. Legacy encoders saw Train identities; no new untouched/unseen-domain claim.',
        runtime_scope='Cached source feature/tracking and publisher wall time exclude original neural/image/IR-generation costs; no new end-to-end FPS claim.',
        stronger_RGB_crossarchitecture_sweeps_deferred=True,Test_used_for_selection=False,historical_Test_exposed=True,
        neural_weights_images_descriptors_frame_banks_MOT_and_GT_not_pushed=True,exporter_sha256=sha256(Path(__file__)),
        goal_met=False,supported_innovations=0)
    for name,key in [('complete_Dev_interventions.json','Dev_interventions'),('full_Test_interventions.json','Test_interventions')]:
        if(base/name).exists():report[key]=[compact(v)for v in read(base/name)['rows']]
    root=doc/'PERCEPTUAL_MFT_20261127_PROTECTED_PERIPHERAL.json';assert not root.exists();root.write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Protected trusted and peripheral tracking: completed registered study','',
        'Primary comparison: original direct-RGB YOLOX-X + ByteTrack, **29.5289706874 HOTA** on all historical Test7 / 15,985 frames. Minimum **34.5289706874**; stretch **39.5289706874**. Strongest-RGB and cross-architecture sweeps remain deferred under the user’s revised scope.','',
        'The trusted parent and its scores, observations, matching and memory remain unchanged. Independent original-score shadow observations supply auxiliary trajectories. Exact union of current trusted measured rectangles suppresses candidates whose area is at least half explained. First publication also needs score strictly between 0.3 and 0.7, shadow streak at least3, an unowned proposal and area>100. Main rows survive. A recently published unowned auxiliary identity may transfer to a brand-new trusted output at the same current shadow proposal; aliases never overwrite an already published main identity or past output. Positive IDs stay compact without future-ID lookahead.','',
        'Six core/serialization regressions and all six real-Train prefix checks passed, including direct/source-cache equivalence, literal RGB missing-gate fallback, causal prefixes and trusted state/row comparison. PF’s first1,330 Train frames also passed a longer check. R1 had a missing import and actual engineering/controller exit1 before any source/HOTA; the original failure, source and corrected R2 registration are preserved. No algorithm, candidate or threshold was changed for that import correction.','',
        'All6Train +2Dev GT-free source producers finish before the separate label diagnostic. No images, descriptors, GT labels or weights enter the new source cache. The diagnostic retains unknown labels and compares original need values to the prior independent scalar bank. It cannot fit parameters or change thresholds. Confidence mass is a heuristic, not a calibrated count or error probability.','',
        'The 18 normal cases combine original/expanded need, thresholds0.1/0.25 and count/RGB/paired/duplicatedRGB, plus unchanged and unconditional references. Standalone validation warms all27,251 Train frames and scores its last quarter6,814. It omits late-calibrated old quality/archive thresholds. Coverage heads/directions were fitted on first75percent; old encoders saw Train identities and earlier development inspected this window. Full Dev uses the old selected BO_paired_q999_c3 parent over all4,834 frames.','',
        'The prospective rule admits only paired cases with validation gain≥0.1 and full Dev HOTA≥52.11311409153856 (tolerance1e-10). Rank by validation HOTA, then Dev HOTA/AssA/IDF1 and identifier. Exact Dev ties are allowed; controls cannot be promoted from Test.','',
        '| Case | Validation HOTA | Gain | Full Dev HOTA | Eligible |','|---|---:|---:|---:|---|']
    d={v['case']['id']:v for v in dev['rows']}
    for v in val['rows']:
        key=v['case']['id'];lines.append('| '+key+' | '+f"{v['combined47']['HOTA']:.9f}"+' | '+f"{val['HOTA_gains_pp'][key]:+.9f}"+' | '+f"{d[key]['combined47']['HOTA']:.9f}"+' | '+str(key in dev['eligible_ids'])+' |')
    if test is None:
        lines.extend(['','No paired case passed the registered joint validation/Dev rule. No Test selection or new Test result was produced. The earlier completed formal Test method remains30.7170810665 (+1.1881103791).'])
    else:
        selected=next(v for v in test['rows']if v['case']==test['selected_case'])
        lines.extend(['',f"Frozen selected case **{test['selected_case']['id']}**: complete Test7 HOTA **{selected['combined47']['HOTA']:.9f}**, gain **{test['HOTA_gain_over_original_RGB_pp']:+.9f}** over original RGB. Numeric+5 reached: **{test['target_plus5_met']}**; +10: **{test['target_plus10_met']}**.",'',
            '| Frozen Test case/control | HOTA | DetA | AssA | IDF1 |','|---|---:|---:|---:|---:|'])
        for v in test['rows']:lines.append('| '+v['case']['id']+' | '+' | '.join(f"{v['combined47'][k]:.9f}"for k in ['HOTA','DetA','AssA','IDF1'])+' |')
        lines.extend(['','The matched RGB/duplicated/count variants alter only the new gate. They retain the common fixed IR visual detector and old quality/archive. Missing gate IR must equal the matched RGB gate. These controls are not whole-system RGB-only baselines. Their Test scores do not change the frozen choice.'])
    lines.extend(['','The first auditor exited1 on a manifest-schema KeyError: plain publishers stored counters, while the script expected wrapper-only statistics. That original source and actual failure are retained. A separate corrected auditor checks the final scored MOT projections and whole source-main hashes directly. Inference code, source banks and original metrics did not change.','',f"The independent audit reparsed {a['official_tables_reparsed']} official tables and {a['official47_fields_checked']} aggregate/per-sequence fields, checked {a['original_main_MOT_projections_checked']} original-MOT projections with {a['original_main_rows_preserved']} retained trusted rows, and rebuilt {a['gate_activation_counts_rebuilt']} gate counts. A separate clipped-cell algorithm recomputed {a['independent_union_geometry_rows']} spatial-union fractions; maximum error {a['maximum_union_fraction_error']:.3g}. All {a['related_jobs_actual_exit0']} related jobs and the controller have actual exit0. Its scope excludes a second pixel/encoder producer and complete independent Kalman/LAP replay.",'',
        'Source processes complete before publishers, and publishers receive only immutable snapshots. This prevents feedback into trusted inference. Every final main row’s geometry, score and constant injective alias is checked against the preserved source. Six-prefix/PF engineering compares the actual original tracker state; full-state replay beyond that scope is not claimed.','',
        f"To conserve space, {a['new_duplicate_MOT_copy_cleanup_verified_bytes']} bytes of newly derived TrackEval input copies were removed only after exact comparison to retained canonical streams, with cleanup receipts. Original CSV276 columns, summary39 fields, all47 values, per-sequence values, logs, model/source hashes and failures are preserved. Canonical trajectories and source banks remain private, excluded from Git along with images, weights, descriptors and GT. Cached source and publisher costs exclude original neural/image/generation costs and are not end-to-end FPS.",'',
        'Fixed November DiffV2IR and ProtocolV2, seed0 only, at most physicalGPU4/5 if needed; this study uses cached CPU workers. Work cap64GiB/reserve40GiB retained. Historical Test exposure, legacy detector Dev exposure and encoder Train-identity exposure remain disclosed. Generated pseudo-IR is RGB-derived, not a physical thermal sensor. ByteTrack, adaptive births, spatial exclusion and identity transfer remain prior components; numeric gains alone cannot establish2–3 unique multimodal contributions.','',
        '**Goal remains active pending numeric and scientific requirements. Supported independently attributable multimodal innovations:0.**',''])
    md=doc/'PERCEPTUAL_MFT_20261127_PROTECTED_PERIPHERAL.md';assert not md.exists();md.write_text('\n'.join(lines))
    paths=[str(root.relative_to(ROOT)),str(md.relative_to(ROOT))]+[v['path']for v in artifacts]
    manifest=directory/'publication_manifest.json';manifest.write_text(json.dumps(dict(paths=paths,file_SHA256={p:sha256(ROOT/p)for p in paths},
        total_bytes=sum((ROOT/p).stat().st_size for p in paths),no_large_assets=True,goal_met=False),indent=2)+'\n')
    print(json.dumps(dict(status='EXPORTED_PROTECTED_PERIPHERAL',paths=len(paths)+1,total_bytes=sum((ROOT/p).stat().st_size for p in paths))),flush=True)


if __name__=='__main__':main()
