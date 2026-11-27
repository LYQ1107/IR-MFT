#!/usr/bin/env python3
"""Read-only official47, whole-sequence output and frozen-choice ledger audit."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_mft_identity_ledger import registered,locations,PREFIX
from tools.mft_identity_ledger_evaluation import policies,matched
from tools.native_pyramid_experiment import read,frozen_json
from tools.fusion_memory_observation import LENGTHS
from tools.fusion_memory_evaluation import official_csv47,validate_mot
from tools.export_perceptual_metrics47 import official_schema
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity
from tools.audit_mft_candidate_risk import require_closed_jobs


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def audit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    started=time.monotonic();main=registered();plan,protocol,work,base,train=locations()
    execution=read(base/'run_r1.execution.json');recovered=(base/'run_r2.execution.json').exists()
    assert execution['actual_child_exit_code']==(1 if recovered else 0)and execution['source_unchanged']
    assert identity(execution['identity']['PID'])is None
    fit=read(train/'fit_complete.json');assert sha256(fit['model_file'])==fit['model_sha256']
    state=read(base/'state.json');dev=read(base/'complete_Dev.json');freeze_path=base/'Test_selection_frozen.json'
    test_state=state
    if recovered:
        re=read(base/'run_r2.execution.json');assert re['actual_child_exit_code']==0 and re['source_unchanged']
        assert identity(re['identity']['PID'])is None
        test_state=read(base/'Test_continuation_r2/state.json')
        assert state['status']=='FAILED_PRESERVE_LEDGER_EVALUATION_ATTEMPT'
        recovery=read(base/'Test_continuation_r2/registered.json')
        for n,h in recovery['source_hashes'].items():assert sha256(ROOT/n)==h,n
        for n,h in recovery['artifact_hashes'].items():assert sha256(n)==h,n
    assert dev['Test_read_for_selection']is False and dev['references_all47_and_MOT_exact']
    require_closed_jobs(state,[PREFIX+'_ENGINEERING'],1)
    schema,_=official_schema(plan['evaluation']['TrackEval_root']);artifacts={};fields=0;motfiles=0
    def record(p):artifacts[str(p)]=sha256(p)
    for p in [base/'pipeline_registered.json',base/'run_r1.execution.json',base/'state.json',base/'complete_Dev.json',
              train/'fit_complete.json',base/'real_Train_engineering.json'] :record(p)
    ae=read(train/'audit_r2.execution.json');ta=read(train/'independent_fit_audit.json')
    assert ae['actual_child_exit_code']==0 and ae['source_unchanged']and identity(ae['identity']['PID'])is None
    assert ta['model_sha256']==fit['model_sha256']and ta['all6_Train_replayed']
    assert read(train/'audit_r1.execution.json')['actual_child_exit_code']==1
    for p in [train/'audit_r1.execution.json',train/'audit_r2.execution.json',train/'independent_fit_audit.json']:record(p)
    if recovered:
        for p in [base/'run_r2.execution.json',base/'Test_continuation_r2/state.json',base/'Test_continuation_r2/registered.json']:record(p)
    def split(source,split,stage,expected_cases):
        nonlocal fields,motfiles
        assert [r['case']for r in source['rows']]==expected_cases
        label=split if stage=='normal'else split+'_intervention'
        phases=[PREFIX+'_'+label.upper()+'_REPLAY',PREFIX+'_'+label.upper()+'_OFFICIAL']
        require_closed_jobs(test_state if split=='test'else state,phases,len(LENGTHS[split])+len(expected_cases))
        manifests={}
        for seq,length in LENGTHS[split].items():
            path=base/'streams'/label/'sequence_manifests'/(seq+'.json');m=read(path);record(path)
            assert m['frames']==length and m['cases']==expected_cases and m['source_versions']==main['source_hashes']
            assert m['model_sha256']==fit['model_sha256']and not m['GT_read']and not m['past_outputs_rewritten']
            assert m['same_detections_and_physical_geometry']and m['original_references_or_missing_RGB_MOT_exact']
            for c in expected_cases:
                s=m['streams'][c['id']];assert sha256(s['track_path'])==s['track_sha256']
                assert s['frames']==length and s['detection_packet_sha256']==m['bank']['packet_sha256']
                validate_mot(s['track_path'],length,s['track_rows']);motfiles+=1
            if stage=='normal':
                old=read(work/'archive_reasoning_evaluation_20261127/streams'/split/'sequence_manifests'/(seq+'.json'))
                for own,previous in [('IL_native_control','AR_native_zero'),('IL_archive_control','AR_RGB_q999_shape')]:
                    if own in m['streams']:assert m['streams'][own]['track_sha256']==old['streams'][previous]['track_sha256']
            else:
                frozen=read(freeze_path);rgb=matched(frozen['chosen_case'],'RGB')['id']
                normal=read(base/'streams/dev/sequence_manifests'/(seq+'.json'))
                assert m['streams']['IL_control_missing']['track_sha256']==normal['streams'][rgb]['track_sha256']
            manifests[seq]=m
        rows=[]
        for r in source['rows']:
            c=r['case'];assert r['frames']==sum(LENGTHS[split].values())
            assert r['source_versions']==main['source_hashes']and r['model_sha256']==fit['model_sha256']
            p=Path(r['official_CSV']);assert sha256(p)==r['official_CSV_sha256'];record(p)
            v=official_csv47(p,schema,split,r['metrics39']);assert v['combined47']==r['combined47']and v['per_sequence47']==r['per_sequence47']
            assert r['coverage']=={s:m['streams'][c['id']]for s,m in manifests.items()}
            fields+=47*(len(LENGTHS[split])+1)
            rows.append(dict(case=c,combined47=v['combined47'],per_sequence47=v['per_sequence47'],
                official_CSV=str(p),official_CSV_sha256=sha256(p)))
        return rows
    devrows=split(dev,'dev','normal',policies())
    chosen=min(dev['rows'],key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),r['case']['id']))
    assert dev['chosen_case']==chosen['case']
    gain=chosen['combined47']['HOTA']-main['parent_best_Dev_HOTA'];assert gain==dev['gain_over_parent_best_Dev_pp']
    result=dict(Dev_rows=devrows,Dev_gain_pp=gain,chosen_case=chosen['case'],Test_rows=[],controls=[],new_Test=False)
    if gain<=1e-8:
        assert state['status']=='COMPLETE_LEDGER_NO_DEV_GAIN_NO_NEW_TEST'and not freeze_path.exists()
        assert not(base/'full_Test_results.json').exists()
    else:
        assert test_state['status']=='COMPLETE_LEDGER_FULL_TEST_REQUIRES_ASSESSMENT'
        f=read(freeze_path);record(freeze_path)
        assert f['complete_Dev_sha256']==sha256(base/'complete_Dev.json')and f['Test_read_for_selection']is False
        assert f['chosen_case']==chosen['case']and f['matched_identity_control']==matched(chosen['case'],'critic'if chosen['case']['kind']=='RGB'else'RGB')
        controls_path=base/'complete_Dev_interventions.json';test_path=base/'full_Test_results.json';record(controls_path);record(test_path)
        control=read(controls_path);test=read(test_path)
        result['controls']=split(control,'dev','intervention',f['Dev_interventions'])
        rgb=next(r for r in dev['rows']if r['case']==matched(chosen['case'],'RGB'))
        assert control['rows'][0]['combined47']==rgb['combined47']and control['rows'][0]['per_sequence47']==rgb['per_sequence47']
        assert control['no_reselection']and test['Test_used_for_selection']is False
        result['Test_rows']=split(test,'test','normal',f['cases']);result['new_Test']=True
        canonical=work/'native_pyramid_evaluation/test/NP_source_canonical_RGB/trackeval_output/pedestrian_detailed.csv'
        assert sha256(canonical)=='c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a';record(canonical)
        original=official_csv47(canonical,schema,'test',read(canonical.parent.parent/'metrics.json')['trackeval'])
        assert abs(original['combined47']['HOTA']-29.528970687393972)<1e-10
        selected=next(r for r in result['Test_rows']if r['case']==chosen['case'])
        delta=selected['combined47']['HOTA']-original['combined47']['HOTA'];assert delta==test['HOTA_gain_over_original_RGB_pp']
        result['HOTA_gain_over_original_RGB_pp']=delta;fields+=47*8
        te=read(train/'audit_r2.execution.json');ta=read(train/'independent_fit_audit.json')
        assert te['actual_child_exit_code']==0 and te['source_unchanged']and identity(te['identity']['PID'])is None
        assert ta['model_sha256']==fit['model_sha256']and ta['all6_Train_replayed']
        record(train/'independent_fit_audit.json');record(train/'audit_r2.execution.json')
    assert sha256(fit['model_file'])==fit['model_sha256'];registered()
    frozen_json(base/'independent_results_audit.json',dict(status='COMPLETE_READ_ONLY_LEDGER_OFFICIAL47_AND_FROZEN_CHOICE_AUDIT',
        UTC=utc(),**result,official_fields_reparsed=fields,full_MOT_files_hash_and_schema_checked=motfiles,
        model_sha256=fit['model_sha256'],model_bytes_unchanged=True,original_study_exit=execution['actual_child_exit_code'],
        recovered_Test_controller_actual_exit_zero=recovered,independent_Train_audit_r1_actual_exit1_preserved=True,
        independent_Train_audit_r2_actual_exit0=True,
        all_required_inference_and_official_jobs_actual_exit_zero=True,source_hashes={**main['source_hashes'],
            'tools/audit_mft_identity_ledger_results.py':sha256(Path(__file__))},artifact_hashes=artifacts,
        new_model_fits=0,new_optimizer_steps=0,new_GPU_work=False,Test_selection_changed=False,
        historical_Test_previously_exposed=True,goal_met=False,supported_innovations=0,seconds=time.monotonic()-started))
    print(json.dumps(dict(status='COMPLETE_READ_ONLY_LEDGER_OFFICIAL47_AND_FROZEN_CHOICE_AUDIT',fields=fields,MOT_files=motfiles,new_Test=result['new_Test'])),flush=True)


if __name__=='__main__':audit()
