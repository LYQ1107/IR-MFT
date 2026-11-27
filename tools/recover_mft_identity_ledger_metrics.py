#!/usr/bin/env python3
"""Complete frozen ledger Dev metrics with equivalent sparse-ID preprocessing.

Official MOT preprocessing sorts unique IDs then remaps them to0..K-1, but its
dense lookup allocates max(ID)+1 cells. This evaluation-only sorted bijection
gives identical internal IDs without that allocation. Raw online MOT stays
immutable. No detector/tracker, fit, parameter, frame or choice is rerun.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_mft_identity_ledger as original
from tools.native_pyramid_experiment import read,frozen_json
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity
from tools.task_utility_queue import Queue
from tools.fusion_memory_observation import LENGTHS
from tools.fusion_memory_evaluation import run_official_tracks,official_csv47,validate_mot
from tools.export_perceptual_metrics47 import official_schema
from tools.audit_mft_candidate_risk import require_closed_jobs


def locations():
    plan,protocol,work,parent,train=original.locations()
    return plan,protocol,work,parent/'compact_official_r2',parent,train


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    main=original.registered();plan,protocol,work,base,parent,train=locations()
    e=read(parent/'run_r1.execution.json');state=read(parent/'state.json')
    assert e['actual_child_exit_code']==1 and e['source_unchanged']and identity(e['identity']['PID'])is None
    assert state['status']=='FAILED_PRESERVE_LEDGER_EVALUATION_ATTEMPT'
    failed=state['phases'][original.PREFIX+'_DEV_OFFICIAL']['IL_RGB_q95']
    assert failed['exit_code']==-15 and failed['observed_peak_RSS_bytes']>8*2**30
    require_closed_jobs(state,[original.PREFIX+'_ENGINEERING',original.PREFIX+'_DEV_REPLAY'],3)
    ae=read(train/'audit_r2.execution.json');assert ae['actual_child_exit_code']==0 and ae['source_unchanged']
    assert identity(ae['identity']['PID'])is None
    assert not(parent/'complete_Dev.json').exists()and not(parent/'Test_selection_frozen.json').exists()
    sources={**main['source_hashes'],'tools/recover_mft_identity_ledger_metrics.py':sha256(Path(__file__))}
    paths=[parent/'pipeline_registered.json',parent/'run_r1.execution.json',parent/'state.json',
        train/'independent_fit_audit.json',train/'audit_r1.execution.json',train/'audit_r2.execution.json']
    for seq,length in LENGTHS['dev'].items():
        p=parent/'streams/dev/sequence_manifests'/(seq+'.json');m=read(p);paths.append(p)
        assert m['frames']==length and m['cases']==original.policies()and m['source_versions']==main['source_hashes']
        for c in original.policies():
            s=m['streams'][c['id']];assert sha256(s['track_path'])==s['track_sha256']
            validate_mot(s['track_path'],length,s['track_rows'])
    dataset=Path(plan['evaluation']['TrackEval_root'])/'trackeval/datasets/mot_challenge_2d_box.py';paths.append(dataset)
    completed=[]
    for c in original.policies():
        p=parent/'evaluation/dev'/c['id']/'metrics.json'
        if p.exists():
            j=state['phases'][original.PREFIX+'_DEV_OFFICIAL'][c['id']]
            assert j['status']=='COMPLETE'and j['exit_code']==0
            completed.append(c['id']);paths.append(p)
    v=dict(status='REGISTERED_EQUIVALENT_OFFICIAL_DEV_ID_COMPACTION_RECOVERY',UTC=original.utc(),source_hashes=sources,
        artifact_hashes={str(p):sha256(p)for p in paths},normal_cases=original.policies(),
        parent_best_Dev_HOTA=main['parent_best_Dev_HOTA'],model_sha256=main['model_sha256'],
        completed_original_official_cases_to_verify_exact=completed,failed_original_worker_exit=-15,
        failed_original_peak_RSS_bytes=failed['observed_peak_RSS_bytes'],original_controller_exit1_preserved=True,
        original_raw_MOT_immutable=True,sorted_bijection_matches_official_internal_IDs=True,
        inference_replayed=False,model_refit=False,Test_used=False,goal_met=False,supported_innovations=0)
    assert not(base/'registered.json').exists();frozen_json(base/'registered.json',v)
    print(json.dumps(dict(status=v['status'],cases=len(v['normal_cases']),original_completed=completed)),flush=True)


def registered():
    original.registered();plan,protocol,work,base,parent,train=locations();r=read(base/'registered.json')
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in r['artifact_hashes'].items():assert sha256(n)==h,n
    return r


def compact(source,target,frames,rows):
    source,target=Path(source),Path(target);validate_mot(source,frames,rows)
    before=sha256(source);labels=set()
    with source.open()as stream:
        for line in stream:labels.add(int(line.split(',')[1]))
    ordered=sorted(labels);mapping={v:i+1 for i,v in enumerate(ordered)};inverse={v:k for k,v in mapping.items()}
    assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True)
    roundtrip=hashlib.sha256();geometry=hashlib.sha256();output_geometry=hashlib.sha256()
    with source.open()as stream,target.open('x')as out:
        for line in stream:
            cells=line.split(',');old=int(cells[1]);geometry.update(','.join([cells[0],*cells[2:]]).encode())
            cells[1]=str(mapping[old]);out.write(','.join(cells))
    with target.open()as stream:
        for line in stream:
            cells=line.split(',');output_geometry.update(','.join([cells[0],*cells[2:]]).encode())
            cells[1]=str(inverse[int(cells[1])]);roundtrip.update(','.join(cells).encode())
    assert roundtrip.hexdigest()==before==sha256(source)and geometry.hexdigest()==output_geometry.hexdigest()
    validate_mot(target,frames,rows)
    return dict(source_path=str(source),source_sha256=before,track_path=str(target),track_sha256=sha256(target),frames=frames,
        track_rows=rows,unique_IDs=len(ordered),maximum_original_ID=max(ordered,default=0),maximum_compact_ID=len(ordered),
        sorted_ID_map_sha256=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest(),
        inverse_recovers_original_MOT_byte_exact=True,nonidentity_columns_and_order_exact=True,
        compact_internal_sorted_ranks_identical_to_original=True,original_file_unchanged=True)


def evaluate(case_id):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();plan,protocol,work,base,parent,train=locations();case=next(c for c in reg['normal_cases']if c['id']==case_id)
    cover={}
    for seq,length in LENGTHS['dev'].items():
        m=read(parent/'streams/dev/sequence_manifests'/(seq+'.json'));s=m['streams'][case_id]
        target=base/'inputs'/case_id/'track_results'/(seq+'.txt')
        cover[seq]=compact(s['track_path'],target,length,s['track_rows'])
    frozen_json(base/'inputs'/case_id/'manifest.json',dict(case=case,coverage=cover,raw_predictions_not_modified=True,
        model_sha256=reg['model_sha256'],registration_sha256=sha256(base/'registered.json')))
    output=base/'evaluation'/case_id;assert not output.exists();output.mkdir(parents=True)
    values=run_official_tracks(base/'inputs'/case_id/'track_results',output,plan,'dev')
    if case_id in reg['completed_original_official_cases_to_verify_exact']:
        old=read(parent/'evaluation/dev'/case_id/'metrics.json')
        assert values['combined47']==old['combined47']and values['per_sequence47']==old['per_sequence47']
    registered()
    frozen_json(output/'metrics.json',dict(status='COMPLETE_ORIGINAL47_FROM_EQUIVALENT_COMPACT_LEDGER_IDS',case=case,
        frames=4834,coverage=cover,model_sha256=reg['model_sha256'],source_versions=reg['source_hashes'],
        original_completed_case_exact_verified=case_id in reg['completed_original_official_cases_to_verify_exact'],
        **values,goal_met=False,supported_innovations=0))


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();plan,protocol,work,base,parent,train=locations()
    with(base/'state.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(base/'state.json').exists()
        state=dict(status='STARTING_EQUIVALENT_COMPACT_LEDGER_DEV47',controller_pid=os.getpid(),phases={},seed=0,
            original_controller_exit1_preserved=True,goal_met=False,supported_innovations=0)
        queue=Queue(work,base,state,plan['data_root'],registered);queue.save();cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))]
        try:
            queue.jobs('LEDGER_COMPACT_DEV47',[dict(id=c['id'],command=cmd+['evaluate','--case',c['id']])for c in reg['normal_cases']],False,
                lambda j:read(base/'evaluation'/j['id']/'metrics.json')['frames']==4834)
            rows=[read(base/'evaluation'/c['id']/'metrics.json')for c in reg['normal_cases']]
            chosen=min(rows,key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),r['case']['id']))
            gain=chosen['combined47']['HOTA']-reg['parent_best_Dev_HOTA']
            frozen_json(base/'complete_Dev.json',dict(status='COMPLETE_ALL16_FROZEN_LEDGER_NORMAL_DEV_VIA_EQUIVALENT_IDS',UTC=original.utc(),
                rows=rows,chosen_case=chosen['case'],parent_best_Dev_HOTA=reg['parent_best_Dev_HOTA'],gain_over_parent_best_Dev_pp=gain,
                Test_read_for_selection=False,all_raw_online_predictions_unchanged=True,all_original_completed_official_cases_exact=True,
                original_controller_and_worker_failures_preserved=True,historical_Test_previously_exposed=True,goal_met=False,supported_innovations=0))
            if gain>1e-8:
                frozen_json(base/'Test_selection_frozen.json',dict(status='FROZEN_RECONSTRUCTED_COMPLETE_DEV_LEDGER_CHOICE',UTC=original.utc(),
                    chosen_case=chosen['case'],complete_Dev_sha256=sha256(base/'complete_Dev.json'),Test_read_for_selection=False,
                    requires_pretest_qualification=True,historical_Test_previously_exposed=True))
            state.update(status='COMPLETE_COMPACT_LEDGER_DEV_NO_GAIN_NO_TEST'if gain<=1e-8 else'COMPLETE_COMPACT_LEDGER_DEV_GAIN_REQUIRES_PRETEST_QUALIFICATION',Dev_gain_pp=gain);queue.save()
        except Exception as error:
            state.update(status='FAILED_PRESERVE_COMPACT_LEDGER_EVALUATION',error=repr(error));queue.save();raise


def audit():
    reg=registered();plan,protocol,work,base,parent,train=locations();e=read(base/'run_r1.execution.json')
    assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(e['identity']['PID'])is None
    state=read(base/'state.json');require_closed_jobs(state,['LEDGER_COMPACT_DEV47'],16)
    result=read(base/'complete_Dev.json');schema,_=official_schema(plan['evaluation']['TrackEval_root']);fields=0
    assert [r['case']for r in result['rows']]==reg['normal_cases']
    for r in result['rows']:
        assert sha256(r['official_CSV'])==r['official_CSV_sha256']
        p=official_csv47(r['official_CSV'],schema,'dev',r['metrics39']);assert p['combined47']==r['combined47']and p['per_sequence47']==r['per_sequence47'];fields+=141
        for seq,c in r['coverage'].items():
            assert sha256(c['source_path'])==c['source_sha256']and sha256(c['track_path'])==c['track_sha256']
            validate_mot(c['track_path'],LENGTHS['dev'][seq],c['track_rows'])
            a=Path(c['source_path']).read_text().splitlines();b=Path(c['track_path']).read_text().splitlines()
            labels=sorted({int(v.split(',')[1])for v in a});inverse={i+1:v for i,v in enumerate(labels)}
            recovered=[]
            for line in b:
                cells=line.split(',');cells[1]=str(inverse[int(cells[1])]);recovered.append(','.join(cells))
            assert a==recovered
        if r['case']['id']in reg['completed_original_official_cases_to_verify_exact']:
            old=read(parent/'evaluation/dev'/r['case']['id']/'metrics.json')
            assert r['combined47']==old['combined47']and r['per_sequence47']==old['per_sequence47']
    chosen=min(result['rows'],key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),r['case']['id']))
    assert chosen['case']==result['chosen_case']and result['Test_read_for_selection']is False
    assert sha256(read(train/'fit_complete.json')['model_file'])==reg['model_sha256'];registered()
    frozen_json(base/'independent_official_audit.json',dict(status='COMPLETE_READ_ONLY_COMPACT_LEDGER_OFFICIAL47_AUDIT',UTC=original.utc(),
        official_fields_reparsed=fields,full_original_and_compact_MOT_files_checked=64,all_inverse_maps_byte_exact=True,
        all3_original_completed_official_tables_exact=True,all16_evaluation_jobs_actual_exit0=True,
        original_controller_exit1_and_worker_minus15_preserved=True,model_sha256=reg['model_sha256'],model_bytes_unchanged=True,
        complete_Dev_sha256=sha256(base/'complete_Dev.json'),new_fit_or_inference=False,Test_selection_changed=False,
        chosen_case=chosen['case'],Dev_gain_pp=result['gain_over_parent_best_Dev_pp'],new_Test=False,goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_READ_ONLY_COMPACT_LEDGER_OFFICIAL47_AUDIT',fields=fields)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','run','evaluate','audit']);p.add_argument('--case');a=p.parse_args()
    if a.mode=='evaluate':evaluate(a.case)
    else:{'register':register,'run':run,'audit':audit}[a.mode]()
