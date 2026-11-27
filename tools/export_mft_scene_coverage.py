#!/usr/bin/env python3
"""Publish compact completed coverage diagnostics; exclude private frame bank."""
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
B=W/'scene_coverage_diagnostic_20261127'
DOC=ROOT/'docs/fusion_research'


def write(path,value):
    assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    result=read(B/'complete_Train_Dev.json');audit=read(B/'independent_scalar_audit.json');reg=read(B/'registered.json')
    assert audit['complete_report_sha256']==sha256(B/'complete_Train_Dev.json')
    for n,h in reg['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for name in ['engineering_r1','run_r1','audit_r1']:
        e=read(B/(name+'.execution.json'));assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(e['identity']['PID'])is None
        for n,h in e['source_hashes'].items():assert sha256(ROOT/n)==h,n
    directory=DOC/'perceptual_20261127_scene_coverage';assert not directory.exists();directory.mkdir()
    names=['registered.json','unit_regressions.json','additional_dependencies_registered.json','real_Train_engineering.json',
        'complete_Train_Dev.json','independent_scalar_audit.json','run_r1.supervisor.py','run_r2.supervisor.py','audit_r1.supervisor.py']
    for stem in ['engineering_r1','run_r1','audit_r1']:
        names.extend([stem+'.execution.json',stem+'.stdout.txt'])
    artifacts=[]
    for name in names:
        source=B/name;target=directory/name;assert source.stat().st_size<1024**2
        shutil.copyfile(source,target);assert sha256(target)==sha256(source)
        artifacts.append(dict(path=str(target.relative_to(ROOT)),bytes=target.stat().st_size,sha256=sha256(target),source_path=str(source)))
    root=DOC/'PERCEPTUAL_MFT_20261127_SCENE_COVERAGE.json'
    report=dict(status='PUBLISHED_COMPLETE_TRAIN_DEV_COVERAGE_DIAGNOSTIC_NOT_TRACKING_ACCURACY',UTC=utc(),
        original_RGB_HOTA=29.528970687393972,minimum_required_Test_HOTA=34.52897068739397,
        stretch_Test_HOTA=39.52897068739397,latest_completed_formal_Test_HOTA=30.717081066539237,
        latest_completed_formal_Test_gain_pp=1.1881103791452645,current_diagnostic=result,
        independent_scalar_audit=audit,artifacts=artifacts,exporter_sha256=sha256(Path(__file__)),
        evidence_interpretation='Coverage signal detects many undercovered MSK Train frames, while threshold0.25 paired/RGB gates never activate on Dev. RGB/count controls are close and sometimes better; no independent IR innovation or HOTA improvement follows from this proxy.',
        next_study='Separately preregistered coverage-recall standalone forward-Train validation and full-Dev noninferiority; no diagnostic-only Test advancement.',
        historical_Test_exposed=True,legacy_ReID_encoders_trained_on_Train_identities=True,
        no_new_descriptors_images_or_models_pushed=True,private_scalar_bank_excluded=True,
        Test_frames_in_this_diagnostic=0,goal_met=False,supported_innovations=0)
    write(root,report)
    lines=['# Scene-level coverage diagnostic: complete Train and Dev','',
        'The original RGB YOLOX-X + ByteTrack Test7 baseline remains **29.5289706874 HOTA**. The minimum target remains **34.5289706874**. The latest completed formal selected result remains **30.7170810665**, or **+1.1881103791** points. This diagnostic creates no new Test accuracy result.','',
        'All 27,251 Train frames and 4,834 Dev frames were processed. The unchanged original teacher and an independent low-threshold geometric shadow produce the current confirmed unused proposal mass. A deficit is the confirmed unused mass divided by that mass plus the original currently measured mass. Confidence sums are uncalibrated proxies, not expected fish counts or error probabilities.','',
        'Every sequence finishes its complete GT-free inference before GT is opened. The read-only Dev adapter cannot fit models, call the Train-only label interface on Dev, or label Test. Unknown/ignored proposals remain unknown. The analysis target is a local-IoU opportunity proxy; it is not a counterfactual HOTA gain.','',
        'At the preregistered deficit threshold 0.25, the target is a missed-opportunity ratio of at least 0.1. Columns below are frame counts.','',
        '| Split / sequence | Gate | Activated | TP | FP | FN |','|---|---|---:|---:|---:|---:|']
    for split in ['train','dev']:
        for sequence,row in result['rows'][split].items():
            for kind in ['count','RGB','paired','IR_roll']:
                v=row['full'][kind]['thresholds']['0.25']
                lines.append('| '+split+' / '+sequence+' | '+kind+' | '+' | '.join(str(v[k])for k in ['selected_frames','TP','FP','FN'])+' |')
    lines.extend(['',
        'Paired confidence selects 927 MSK-002 frames, of which 902 meet the target, with 25 false positives and 536 false negatives. RGB selects 936 with 910 true positives. The paired signal is useful as a coverage proxy; the comparison does not establish a unique IR contribution. At threshold 0.25, paired and RGB select zero frames in both Dev sequences. Simple count selects five frames in SN-013. No sequence-specific threshold or Test observation enters the rule.','',
        'The independent audit recomputed scalar tables and count conservation for all 32,085 frames, verified producer/engineering actual exit zero and frozen source hashes, and checked the existing Train original-MOT/label hashes and opportunity counts. It did not independently replay the visual producer, frame GT matching, or tracker states. The private scalar table is only 165,054 compressed bytes and is excluded from Git. No new GPU computation was needed.','',
        'Engineering passed causal prefix checks on all six real Train sequences and literal RGB fallback checks. This supports the engineering properties only. The next closed-loop recall study has its own prospective selection rule: standalone forward Train validation of the new controller, followed by complete Dev noninferiority. It must not relax or reopen the previously failed conditional-quality study.','',
        'The earlier Test history is exposed. Existing ReID encoders have seen Train identities. The original RGB detector has legacy Dev exposure. Fixed November DiffV2IR remains a generated auxiliary view of RGB, not a physical thermal sensor. No new tracking, domain-generalization, innovation-count or runtime claim is established here.','',
        '**Goal remains active and unmet. Supported independent multimodal innovations: 0.**',''])
    markdown=DOC/'PERCEPTUAL_MFT_20261127_SCENE_COVERAGE.md';assert not markdown.exists();markdown.write_text('\n'.join(lines))
    paths=[str(root.relative_to(ROOT)),str(markdown.relative_to(ROOT))]+[v['path']for v in artifacts]
    write(directory/'publication_manifest.json',dict(paths=paths,source_SHA={p:sha256(ROOT/p)for p in paths},
        total_bytes=sum((ROOT/p).stat().st_size for p in paths),private_bank_pushed=False,goal_met=False))
    print(json.dumps(dict(status='EXPORTED',paths=len(paths)+1,total_bytes=sum((ROOT/p).stat().st_size for p in paths),report=str(root))),flush=True)


if __name__=='__main__':main()
