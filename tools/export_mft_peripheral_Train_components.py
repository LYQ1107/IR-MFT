#!/usr/bin/env python3
"""Independently audit and publish completed Train-only factorial removals."""
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.native_pyramid_experiment import read,frozen_json
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity
from tools.audit_mft_coverage_recall import parse_official
from tools.export_mft_coverage_recall import compact
from tools.fit_mft_proposal_quality import utc

W=Path('/var/tmp/liuyeqiang_ir_mft_perceptual_20261122')
B=W/'protected_peripheral_Train_components_20261127'


def main():
    r=read(B/'registered.json');e=read(B/'run_r1.execution.json');f=read(B/'complete_Train_components.json')
    assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(e['identity']['PID'])is None
    assert not r['Test_metrics_read']and not f['Test_metrics_read']and not r['main_Test_results_existed_at_registration']
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in r['artifact_hashes'].items():assert sha256(n)==h,n
    lengths={'BT-001':750,'BT-003':61,'MSK-002':439,'PF-001':3750,'SN-001':171,'SN-015':1643};fields=0
    for row in f['removal_rows']:
        path=B/'evaluation'/row['case']['id']/'trackeval_output/pedestrian_detailed.csv'
        raw=parse_official(path,row['official_schema'],lengths);assert sha256(path)==row['official_CSV_sha256']
        assert {k:raw[-1][k]for k in row['combined47']}==row['combined47']
        for own,source in zip(row['per_sequence47'],raw):assert all(own[k]==v for k,v in source.items());fields+=47
        cleanup=read(path.parent.parent/'derived_duplicate_cleanup.json')
        for item in cleanup['rows']:assert not Path(item['path']).exists()and sha256(item['retained'])==item['sha256']
    values={v['case']['id']:v['combined47']['HOTA']for v in f['removal_rows']};full=f['selected_original_validation']['combined47']['HOTA']
    no_spatial=values['PC_without_spatial'];no_handoff=values['PC_without_handoff'];neither=values['PC_without_both']
    expected=dict(full_over_reference=full-f['reference_original_validation']['combined47']['HOTA'],spatial_given_handoff=full-no_spatial,
        handoff_given_spatial=full-no_handoff,spatial_without_handoff=no_handoff-neither,handoff_without_spatial=no_spatial-neither,
        interaction=full-no_handoff-no_spatial+neither,need_gate_given_both=full-f['unconditional_original_validation']['combined47']['HOTA'])
    assert f['HOTA_differences_pp']==expected
    audit=dict(status='COMPLETE_INDEPENDENT_COMPONENT_CSV_AND_FACTORIAL_AUDIT',UTC=utc(),actual_worker_exit0=True,
        complete_sha256=sha256(B/'complete_Train_components.json'),official_tables_reparsed=3,official47_fields_checked=fields,
        factorial_differences_recomputed=True,source_hashes_verified=True,Test_metrics_read=False,
        scope='Independent official CSV and arithmetic; original main-row retention was checked by the component worker against frozen source using shared audit_projection helper.',
        exporter_source_sha256=sha256(Path(__file__)),goal_met=False,supported_innovations=0)
    frozen_json(B/'independent_component_table_audit.json',audit)
    doc=ROOT/'docs/fusion_research';directory=doc/'perceptual_20261127_peripheral_Train_components';assert not directory.exists();directory.mkdir()
    names=['registered.json','run_r1.supervisor.py','run_r1.execution.json','run_r1.stdout.txt','complete_Train_components.json','independent_component_table_audit.json']
    for p in sorted((B/'evaluation').rglob('*')):
        if p.is_file()and p.name in ['metrics.json','pedestrian_detailed.csv','pedestrian_summary.txt','derived_duplicate_cleanup.json']:names.append(str(p.relative_to(B)))
    artifacts=[]
    for name in names:
        source=B/name;target=directory/name;assert source.stat().st_size<2*2**20
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);assert sha256(target)==sha256(source)
        artifacts.append(dict(path=str(target.relative_to(ROOT)),sha256=sha256(target),bytes=target.stat().st_size))
    report=dict(status='PUBLISHED_COMPLETE_SELECTED_METHOD_TRAIN_FACTORIAL',UTC=utc(),selected_case=f['selected_case'],
        full=compact(f['selected_original_validation']),reference=compact(f['reference_original_validation']),
        unconditional=compact(f['unconditional_original_validation']),removals=[compact(v)for v in f['removal_rows']],
        HOTA_differences_pp=expected,independent_audit=audit,artifacts=artifacts,
        original_RGB_Test_baseline_HOTA=29.528970687393972,minimum_Test_target=34.52897068739397,
        scope='Train development-window component evidence, not new Test scores or standalone proof of IR contribution; old encoder Train-identity and validation-development exposure disclosed.',
        Test_metrics_read=False,main_method_selection_unchanged=True,goal_met=False,supported_innovations=0)
    root=doc/'PERCEPTUAL_MFT_20261127_PERIPHERAL_TRAIN_COMPONENTS.json';assert not root.exists();root.write_text(json.dumps(report,indent=2)+'\n')
    text=['# Selected protected method: Train-only component removals','',
        'The method was selected from the registered Train/Dev rule before Test. These component cases were fixed before reading any new Test metrics and cannot alter that selection. Existing GT-free source snapshots and the same selected need mode/threshold are reused, with no new neural training, descriptor bank, image or seed.','',
        'All27,251 Train frames warm the publisher; only the final6,814 frames are scored. This is development-window evidence; older studies inspected the window and old ReID encoders saw Train identities. It is not an untouched or unseen-domain test.','',
        '| Case | HOTA |','|---|---:|',f'| Original same-input reference | {f["reference_original_validation"]["combined47"]["HOTA"]:.9f} |',
        f'| Full selected method | {full:.9f} |',f'| Without spatial explanation | {no_spatial:.9f} |',f'| Without identity handoff | {no_handoff:.9f} |',f'| Without both | {neither:.9f} |',
        f'| Without need gate (unconditional, same spatial/handoff) | {f["unconditional_original_validation"]["combined47"]["HOTA"]:.9f} |','',
        f'The spatial contribution with handoff retained is **{expected["spatial_given_handoff"]:+.9f} HOTA**; handoff with spatial explanation retained is **{expected["handoff_given_spatial"]:+.9f}**. Their factorial interaction is **{expected["interaction"]:+.9f}**. The selected need gate versus unconditional is **{expected["need_gate_given_both"]:+.9f}**. These are conditional ablation differences, not additive independent gains.','',
        'Every trusted-row geometry/score and injective constant alias was checked by the component worker against the immutable source. Three original detailed276-column CSVs, all47 aggregate/per-sequence metrics, summaries, source hashes and actual worker exit0 are preserved. A separate parser recomputed987 official fields and all factorial arithmetic; its scope does not include a second visual producer or independent full Kalman/LAP replay.','',
        'Numeric and mechanism differences do not establish an IR-specific effect. All variants retain the fixed IR visual detector and older parent modules. The primary original-RGB Test7 baseline stays29.5289706874; the minimum Test target stays34.5289706874. This report produces no new Test result and counts no new independently validated multimodal innovation. Goal remains active.','']
    md=doc/'PERCEPTUAL_MFT_20261127_PERIPHERAL_TRAIN_COMPONENTS.md';assert not md.exists();md.write_text('\n'.join(text))
    paths=[str(root.relative_to(ROOT)),str(md.relative_to(ROOT))]+[v['path']for v in artifacts]
    manifest=directory/'publication_manifest.json';manifest.write_text(json.dumps(dict(paths=paths,file_SHA256={p:sha256(ROOT/p)for p in paths},
        total_bytes=sum((ROOT/p).stat().st_size for p in paths),large_assets_included=False),indent=2)+'\n')
    print(json.dumps(dict(status='EXPORTED_TRAIN_FACTORIAL',fields=fields,paths=len(paths)+1,bytes=sum((ROOT/p).stat().st_size for p in paths))),flush=True)


if __name__=='__main__':main()
