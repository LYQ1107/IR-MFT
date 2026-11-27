#!/usr/bin/env python3
"""Independent official-table, source, gate-count and selection audit."""
import csv
import configparser
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.native_pyramid_experiment import read,frozen_json
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity
from tools.fit_mft_proposal_quality import utc
from tools.native_pyramid_training_loop import guard_storage

W=Path('/var/tmp/liuyeqiang_ir_mft_perceptual_20261122')
B=W/'coverage_recall_20261127'


def audit_validation_views(registration):
    rows={}
    for sequence,q in registration['qualified']['train'].items():
        v=registration['validation_GT_views'][sequence];cut=(3*q['frames'])//4
        assert v['window_cut']==cut and v['frames']==q['frames']-cut
        assert sha256(q['GT_path'])==q['GT_sha256']==v['original_GT_sha256']
        expected=[];n=0
        with Path(q['GT_path']).open()as source:
            for raw in source:
                cells=raw.rstrip('\r\n').split(',');frame=int(cells[0])
                assert 1<=frame<=q['frames']
                if frame>cut:
                    n+=1;expected.append(str(frame-cut)+','+','.join(cells[1:])+'\n')
        path=Path(v['validation_GT_path']);assert path.read_text()==''.join(expected)
        assert sha256(path)==v['validation_GT_sha256']and n==v['GT_rows']
        ini=configparser.ConfigParser();ini.read(path.parent.parent/'seqinfo.ini')
        assert int(ini['Sequence']['seqLength'])==v['frames']
        rows[sequence]=dict(original_frames=q['frames'],window_cut=cut,scored_frames=v['frames'],
            all_original_window_rows_retained=n,GT_view_sha256=sha256(path),only_frame_column_shifted=True)
    assert sum(v['scored_frames']for v in rows.values())==6814
    return rows


def parse_official(path,schema,lengths):
    with path.open(newline='')as f:
        reader=csv.DictReader(f);columns=reader.fieldnames;raw=list(reader)
    assert len(columns)==len(set(columns))==276 and len(schema)==47
    assert [r['seq']for r in raw]==[*lengths,'COMBINED']
    assert {k for k in columns if k!='seq'and('___'not in k or k.endswith('___AUC'))}=={v['source_column']for v in schema}
    result=[]
    for row in raw:
        assert all(math.isfinite(float(row[k]))for k in columns if k!='seq')
        v={}
        for item in schema:
            raw_value=float(row[item['source_column']]);value=raw_value*item['scale_from_raw_detailed']
            if item['unit']=='count':assert raw_value.is_integer();value=int(raw_value)
            v[item['metric']]=value
        frames=sum(lengths.values())if row['seq']=='COMBINED'else lengths[row['seq']]
        assert v['CLR_Frames']==frames and v['CLR_TP']+v['CLR_FN']==v['GT_Dets']
        assert v['IDTP']+v['IDFN']==v['GT_Dets']and v['IDTP']+v['IDFP']==v['Dets']
        assert math.isclose(v['HOTA_TP']+v['HOTA_FN'],v['GT_Dets'],abs_tol=1e-8,rel_tol=1e-12)
        assert math.isclose(v['HOTA_TP']+v['HOTA_FP'],v['Dets'],abs_tol=1e-8,rel_tol=1e-12)
        result.append(dict(seq=row['seq'],**v))
    return result


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r=read(B/'registered.json');e=read(B/'run_r1.execution.json');s=read(B/'state.json')
    assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(e['identity']['PID'])is None
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in r['artifact_hashes'].items():assert sha256(n)==h,n
    views=audit_validation_views(r)
    jobs=[v for phase in s['phases'].values()for v in phase.values()]
    assert all(v['status']=='COMPLETE'and type(v['exit_code'])is int and v['exit_code']==0 and v['argv_identity_observed']and identity(v['pid'])is None for v in jobs)
    diag=read(W/'scene_coverage_diagnostic_20261127/complete_Train_Dev.json');p=diag['private_scalar_table']
    assert sha256(p['path'])==p['sha256']
    with np.load(p['path'],allow_pickle=False)as z:scalar={k:z[k]for k in z.files}
    kinds=p['feature_columns'];fields=0;tables=0;track_files=0;gate_checks=0
    from tools.fusion_memory_observation import LENGTHS
    for label_dir in sorted((B/'streams').iterdir()):
        label=label_dir.name;logical='train'if label=='trainval'else label.split('_')[0]
        for path in sorted((label_dir/'sequence_manifests').glob('*.json')):
            m=read(path);sequence=m['sequence']
            assert m['source_versions']==r['source_hashes']and not m['GT_labels_read']and not m['future_frame_input']and not m['past_outputs_rewritten']
            assert m['quality_model_sha256']==r['quality_model_sha256']and m['archive_model_sha256']==r['archive_model_sha256']
            assert m['original_detector_scores_boxes_order_preserved']
            if logical in ['train','dev']:
                x=scalar[logical+'_'+sequence+'_features']
                for c in m['cases']:
                    kind=c['kind'];mode=c.get('intervention')
                    if kind=='reference':expected=0
                    elif kind=='unconditional':expected=len(x)
                    else:
                        k='IR_roll'if mode=='roll'else'IR_missing'if mode=='missing'else kind
                        expected=int(np.count_nonzero(x[:,kinds.index(k)]>=c['threshold']))
                    assert m['statistics'][c['id']]['coverage_gated_frames']==expected,(logical,sequence,c['id'])
                    assert m['statistics'][c['id']]['standalone']==(logical=='train');gate_checks+=1
            if label=='trainval':
                v=r['validation_GT_views'][sequence];assert m['frames']==v['frames']and m['validation_window_cut']==v['window_cut']
                assert m['all_past_frames_consumed']==r['qualified']['train'][sequence]['frames']
            else:assert m['frames']==m['all_past_frames_consumed']==LENGTHS[logical][sequence]
            for name,v in m['streams'].items():
                f=Path(v['track_path']);assert sha256(f)==v['track_sha256'];seen=set();previous=0;count=0
                with f.open()as stream:
                    for line in stream:
                        cells=line.strip().split(',');assert len(cells)==10
                        frame=int(cells[0]);track_id=int(cells[1]);box=list(map(float,cells[2:7]))
                        assert previous<=frame<=m['frames']and frame>=1 and track_id>0 and all(math.isfinite(a)for a in box)
                        assert box[2]>0 and box[3]>0 and 0<=box[4]<=1
                        if frame!=previous:seen=set()
                        assert track_id not in seen;seen.add(track_id);previous=frame;count+=1
                assert count==v['track_rows'];track_files+=1
            if label in ['dev','test']:
                old=read(W/'birth_only_evaluation_20261127/streams'/logical/'sequence_manifests'/(sequence+'.json'))
                assert m['streams']['CR_reference']['track_sha256']==old['streams']['BO_paired_q999_c3']['track_sha256']
            if label.endswith('_intervention'):
                normal=read(B/'streams'/logical/'sequence_manifests'/(sequence+'.json'));freeze=read(B/'Test_selection_frozen.json')
                rgb=next(c['id']for c in r['normal_cases']if c['kind']=='RGB'and c['threshold']==freeze['chosen_case']['threshold'])
                assert m['streams']['CR_gate_missing']['track_sha256']==normal['streams'][rgb]['track_sha256']
    for folder in sorted((B/'evaluation').iterdir()):
        label=folder.name;logical='trainval'if label=='trainval'else label.split('_')[0]
        lengths=r['specification']['validation_lengths']if logical=='trainval'else LENGTHS[logical]
        for path in sorted(folder.glob('*/metrics.json')):
            v=read(path);csvpath=path.parent/'trackeval_output/pedestrian_detailed.csv';raw=parse_official(csvpath,v['official_schema'],lengths)
            assert sha256(csvpath)==v['official_CSV_sha256']
            assert {k:raw[-1][k]for k in v['combined47']}==v['combined47']
            for own,source in zip(v['per_sequence47'],raw):
                assert all(own[k]==value for k,value in source.items());fields+=47
            summary=(path.parent/'trackeval_output/pedestrian_summary.txt').read_text().splitlines()
            headers=summary[0].split();rounded=list(map(float,summary[1].split()))
            assert len(headers)==len(rounded)==39 and all(abs(v['combined47'][k]-value)<=.000501 for k,value in zip(headers,rounded))
            tables+=1
    val=read(B/'complete_forward_Train_validation.json');dev=read(B/'complete_Dev.json');spec=r['specification']
    reference=next(v for v in val['rows']if v['case']['kind']=='reference')
    gains={v['case']['id']:v['combined47']['HOTA']-reference['combined47']['HOTA']for v in val['rows']}
    assert gains==val['HOTA_gains_pp']==dev['validation_HOTA_gains_pp']
    eligible=[v for v in dev['rows']if v['case']['kind']in spec['candidate_selection_kinds']and gains[v['case']['id']]>=spec['standalone_min_validation_HOTA_gain_pp']and v['combined47']['HOTA']>=spec['full_Dev_noninferiority_HOTA']-spec['full_Dev_noninferiority_tolerance_pp']]
    assert [v['case']['id']for v in eligible]==dev['eligible_ids']
    chosen=min(eligible,key=lambda v:(-gains[v['case']['id']],*(-v['combined47'][k]for k in ['HOTA','AssA','IDF1']),v['case']['id']))if eligible else None
    assert dev['selected_case']==(None if chosen is None else chosen['case'])
    ref=next(v for v in dev['rows']if v['case']['kind']=='reference');assert ref['combined47']==r['full_parent_Dev47']
    test=None
    if chosen is None:
        assert not(B/'Test_selection_frozen.json').exists()and not(B/'full_Test_results.json').exists()
        assert s['status']=='COMPLETE_COVERAGE_RECALL_NO_ELIGIBLE_VALIDATION_DEV_CASE_NO_NEW_TEST'
    else:
        freeze=read(B/'Test_selection_frozen.json');test=read(B/'full_Test_results.json')
        assert freeze['chosen_case']==chosen['case']==test['selected_case']and not test['Test_used_for_selection']
        assert freeze['complete_Dev_sha256']==sha256(B/'complete_Dev.json')and freeze['validation_sha256']==sha256(B/'complete_forward_Train_validation.json')
        selected=next(v for v in test['rows']if v['case']==chosen['case'])
        assert test['HOTA_gain_over_original_RGB_pp']==selected['combined47']['HOTA']-test['original_RGB_baseline']['combined47']['HOTA']
        assert test['original_RGB_baseline']['official_CSV_sha256']=='c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
        assert test['target_plus5_met']==(test['HOTA_gain_over_original_RGB_pp']>=5)
        assert test['target_plus10_met']==(test['HOTA_gain_over_original_RGB_pp']>=10)
    value=dict(status='COMPLETE_INDEPENDENT_COVERAGE_RECALL_RESULTS_AUDIT',UTC=utc(),
        controller_actual_exit_code=0,controller_execution_sha256=sha256(B/'run_r1.execution.json'),
        actual_related_jobs_exit0=len(jobs),source_files_verified=len(r['source_hashes']),official_tables_reparsed=tables,
        official47_fields_rechecked=fields,original_MOT_files_checked=track_files,
        validation_GT_views_independently_reconstructed=views,
        Train_Dev_gate_counts_rebuilt_from_preexisting_independent_diagnostic_scalar_bank=gate_checks,
        independent_Train_validation_and_Dev_selection_exact=True,
        selected_case=None if chosen is None else chosen['case'],
        Test_HOTA_gain_pp=None if test is None else test['HOTA_gain_over_original_RGB_pp'],
        Test_read_for_selection=False,auditor_source_sha256=sha256(Path(__file__)),budget=guard_storage(W),
        scope='Raw official47/39, MOT validity and hashes, unchanged parent references, Train/Dev gate activation from prior scalar bank, source/actual exits and prospective selection; not a second visual producer, independent Kalman/LAP replay or end-to-end runtime test.',
        goal_met=False,supported_innovations=0)
    frozen_json(B/'independent_results_audit.json',value);print(json.dumps(value),flush=True)


if __name__=='__main__':main()
