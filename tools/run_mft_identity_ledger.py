#!/usr/bin/env python3
"""Full-Dev ledger comparison; freeze before controls and conditional Test7."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.fit_mft_identity_ledger import registered as Train_registered,locations as Train_locations
from tools.mft_identity_ledger import LedgerLogistic
from tools.mft_identity_ledger_evaluation import policies,matched,interventions,make_ledger
from tools.mft_candidate_risk_teacher import CandidateTeacher
from tools.fit_mft_candidate_risk import threshold_values
from tools.run_mft_archive_reasoning import fitted as archive_fitted
from tools.mft_archive_evaluation import policies as archive_policies,ArchiveTracker
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.run_mft_candidate_train_bank import atomic_json
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity
from tools.task_utility_queue import Queue
from tools.fusion_memory_observation import LENGTHS
from tools.fusion_memory_streams import FusionMemoryStreamWriter
from tools.fusion_memory_evaluation import run_official_tracks,validate_mot,official_csv47
from tools.export_perceptual_metrics47 import official_schema
from tools.audit_mft_tracklet_r2 import bank_audit
from tools.run_mft_tracklet_permanence_r2 import registered as peer_registered
from tools.mft_candidate_risk_data import load_arrays
from tools.native_pyramid_stream_tracks import mot_line

SPEC=ROOT/'configs/perceptual_identity_ledger_evaluation_20261127.json'
PREFIX='IDENTITY_LEDGER'


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def locations():
    plan,protocol,work,train,parent=Train_locations()
    return plan,protocol,work,work/'identity_ledger_evaluation_20261127',train


def fitted():
    r=Train_registered();plan,protocol,work,base,train=locations();e=read(train/'fit_r1.execution.json')
    assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(e['identity']['PID'])is None
    f=read(train/'fit_complete.json');assert f['Train_frames']==27251 and f['Train_sequences']==6
    assert f['Dev_rows']==f['Test_rows']==0 and f['all_teacher_MOT_exact']and f['truncated_pairs']==0
    assert f['registration_sha256']==sha256(train/'fit_registered.json')and sha256(f['model_file'])==f['model_sha256']
    p=read(f['model_file']);models={k:LedgerLogistic.from_dict(v)for k,v in p['models'].items()}
    assert p['thresholds']==f['thresholds']
    return r,f,models,f['thresholds']


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg,fit,models,thresholds=fitted();plan,protocol,work,base,train=locations();spec=read(SPEC)
    assert spec['normal_cases']==policies()and len(policies())==16
    old=read(work/'archive_reasoning_evaluation_20261127/complete_Dev.json')
    assert old['Test_read_for_selection']is False
    chosen=min(old['rows'],key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),r['case']['id']))
    assert chosen['case']['id']=='AR_RGB_q999_shape'
    proof=read(base/'CPU_controls_r1.json');assert proof['actual_tool_exit_code']==0 and proof['tests_passed']==3
    for n,h in proof['source_hashes'].items():assert sha256(ROOT/n)==h,n
    sources=dict(reg['source_hashes'])
    for name in ['tools/mft_identity_ledger_evaluation.py','tools/run_mft_identity_ledger.py',
                 'tests/test_mft_identity_ledger_controls.py',str(SPEC.relative_to(ROOT))]:sources[name]=sha256(ROOT/name)
    banks={s:bank_audit(work/'tracklet_permanence_20261127_r2','dev',s,peer_registered())for s in LENGTHS['dev']}
    files=[SPEC,base/'CPU_controls_r1.json',train/'fit_complete.json',train/'fit_r1.execution.json',work/'archive_reasoning_evaluation_20261127/complete_Dev.json']
    v=dict(status='REGISTERED_ALL16_FULL_DEV_LEDGER_CASES_BEFORE_NEW_DEV',UTC=utc(),source_hashes=sources,
        artifact_hashes={str(p):sha256(p)for p in files},model_sha256=fit['model_sha256'],normal_cases=policies(),
        Dev_frames=4834,Dev_banks=banks,parent_best_Dev_HOTA=chosen['combined47']['HOTA'],
        parent_best_Dev_case=chosen['case'],Train_only_model_fit=True,Test_used_for_registration_or_selection=False,
        historical_Test_previously_exposed=True,goal_met=False,supported_innovations=0)
    assert not(base/'pipeline_registered.json').exists();frozen_json(base/'pipeline_registered.json',v)
    print(json.dumps(dict(status=v['status'],sources=len(sources),parent_Dev_HOTA=v['parent_best_Dev_HOTA'])),flush=True)


def registered():
    Train_registered();plan,protocol,work,base,train=locations();r=read(base/'pipeline_registered.json')
    assert r['normal_cases']==policies()
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in r['artifact_hashes'].items():assert sha256(n)==h,n
    return r


def maker(case,models,thresholds,work,old_models,old_thresholds):
    speed=threshold_values(work)['Train_unknown_speed_upper']
    if case['kind']=='reference_native':return CandidateTeacher(speed)
    if case['kind']=='reference_archive':
        old=next(c for c in archive_policies()if c['id']=='AR_RGB_q999_shape')
        return ArchiveTracker(old,old_models,old_thresholds,speed)
    return make_ledger(case,models,thresholds,speed)


def prefix(arrays,cases,models,thresholds,work,sequence,count,old_models,old_thresholds):
    trackers={c['id']:maker(c,models,thresholds,work,old_models,old_thresholds)for c in cases};result={c['id']:[]for c in cases}
    for i in range(count):
        lo,hi=map(int,arrays['offsets'][i:i+2])
        for c in cases:result[c['id']].extend(trackers[c['id']].update(sequence,i+1,arrays['detections'][lo:hi],arrays['rgb'][lo:hi],arrays['ir'][lo:hi]))
    return {k:''.join(map(mot_line,v))for k,v in result.items()}


def engineer():
    reg=registered();_,fit,models,thresholds=fitted();plan,protocol,work,base,train=locations()
    _,_,old_models,old_thresholds=archive_fitted();cases=policies();result={}
    controls=[dict(matched(c,'critic'),id='MISSING_'+c['id'],intervention='missing')for c in cases if c['kind']=='RGB']
    for sequence,q in Train_registered()['qualified_Train'].items():
        arrays=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);count=64;cut=32
        full=prefix(arrays,cases+controls,models,thresholds,work,sequence,count,old_models,old_thresholds)
        past=prefix(arrays,cases+controls,models,thresholds,work,sequence,cut,old_models,old_thresholds)
        assert all(full[k].startswith(past[k])for k in full)
        for c in cases:
            if c['kind']=='RGB':assert full[c['id']]==full['MISSING_'+c['id']]
        at=int(arrays['offsets'][cut]);arrays['detections'][at:,:4]+=777
        for cue in ['rgb','ir']:arrays[cue][at:]*=-1
        future=prefix(arrays,cases+controls,models,thresholds,work,sequence,count,old_models,old_thresholds)
        assert all(future[k].startswith(past[k])for k in future)
        result[sequence]=dict(frames=count,prefix_frames=cut,all16_cases_and7_missing_controls_causal_prefix_exact=True,
            all7_missing_IR_MOT_exact_RGB=True,bank_sha256=q['bank_sha256'])
        guard_storage(work)
    assert sha256(fit['model_file'])==fit['model_sha256'];registered()
    frozen_json(base/'real_Train_engineering.json',dict(status='COMPLETE_FITTED_LEDGER_ALL6_REAL_TRAIN_ENGINEERING',UTC=utc(),
        rows=result,sequences=6,source_hashes=reg['source_hashes'],model_sha256=fit['model_sha256'],new_accuracy=False))


def split_cases(split,stage):
    plan,protocol,work,base,train=locations()
    if split=='test':
        f=read(base/'Test_selection_frozen.json');assert f['Test_read_for_selection']is False
        assert f['complete_Dev_sha256']==sha256(base/'complete_Dev.json');return f['cases']
    assert split=='dev'
    if stage=='normal':return policies()
    assert stage=='intervention';return read(base/'Test_selection_frozen.json')['Dev_interventions']


def replay(split,stage,sequence):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''and sequence in LENGTHS[split]
    reg=registered();_,fit,models,thresholds=fitted();plan,protocol,work,base,train=locations()
    _,_,old_models,old_thresholds=archive_fitted();cases=split_cases(split,stage);label=split if stage=='normal'else split+'_intervention'
    peer=work/'tracklet_permanence_20261127_r2';bank=bank_audit(peer,split,sequence,peer_registered())
    meta=read(peer/'cache'/split/(sequence+'.json'))
    with np.load(meta['path'],allow_pickle=False)as data:arrays={k:data[k].copy()for k in data.files}
    writer=FusionMemoryStreamWriter(base/'streams'/label,sequence,[c['id']for c in cases]);started=time.monotonic()
    trackers={c['id']:maker(c,models,thresholds,work,old_models,old_thresholds)for c in cases}
    try:
        for i,f in enumerate(arrays['frame_ids']):
            lo,hi=map(int,arrays['offsets'][i:i+2]);d=arrays['detections'][lo:hi]
            rows={c['id']:trackers[c['id']].update(sequence,int(f),d,arrays['rgb'][lo:hi],arrays['ir'][lo:hi])for c in cases}
            # All ledger cases preserve the same detector/physical box stream.
            raw=[rows[c['id']]for c in cases if c['kind']in ['RGB','critic']and not c['shape']]
            if len(raw)>1:
                def geometry(v):return b''.join(np.r_[f,box,score].astype(np.float64).tobytes()for f,box,identity,score in v)
                assert all(geometry(v)==geometry(raw[0])for v in raw)
            writer.append(int(f),rows,{c['id']:d for c in cases})
            if int(f)%250==0 or int(f)==LENGTHS[split][sequence]:
                guard_storage(work);atomic_json(base/'progress'/label/(sequence+'.json'),dict(frame=int(f),frames=LENGTHS[split][sequence],seconds=time.monotonic()-started))
        streams=writer.close(LENGTHS[split][sequence])
    finally:writer.abort()
    if stage=='normal':
        old=read(work/'archive_reasoning_evaluation_20261127/streams'/split/'sequence_manifests'/(sequence+'.json'))
        for own,previous in [('IL_native_control','AR_native_zero'),('IL_archive_control','AR_RGB_q999_shape')]:
            if own in streams:assert streams[own]['track_sha256']==old['streams'][previous]['track_sha256']
    else:
        chosen=read(base/'Test_selection_frozen.json')['chosen_case'];rgb=matched(chosen,'RGB')['id']
        normal=read(base/'streams/dev/sequence_manifests'/(sequence+'.json'))
        assert streams['IL_control_missing']['track_sha256']==normal['streams'][rgb]['track_sha256']
    assert all(v['detection_packet_sha256']==bank['packet_sha256']for v in streams.values())
    assert sha256(fit['model_file'])==fit['model_sha256'];registered()
    frozen_json(base/'streams'/label/'sequence_manifests'/(sequence+'.json'),dict(status='COMPLETE_CAUSAL_LEDGER_FULL_SEQUENCE',
        UTC=utc(),split=split,stage=stage,sequence=sequence,frames=LENGTHS[split][sequence],cases=cases,streams=streams,bank=bank,
        source_versions=reg['source_hashes'],model_sha256=fit['model_sha256'],statistics={k:t.statistics for k,t in trackers.items()},
        same_detections_and_physical_geometry=True,GT_read=False,past_outputs_rewritten=False,model_bytes_unchanged=True,
        original_references_or_missing_RGB_MOT_exact=True,cached_CPU_replay_seconds=time.monotonic()-started,neural_and_image_cost_excluded=True))


def evaluate(split,stage,case_id):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();_,fit,models,thresholds=fitted();plan,protocol,work,base,train=locations()
    case=next(c for c in split_cases(split,stage)if c['id']==case_id);label=split if stage=='normal'else split+'_intervention';coverage={}
    for sequence,frames in LENGTHS[split].items():
        m=read(base/'streams'/label/'sequence_manifests'/(sequence+'.json'))
        assert m['frames']==frames and m['source_versions']==reg['source_hashes']and m['model_sha256']==fit['model_sha256']
        s=m['streams'][case_id];assert sha256(s['track_path'])==s['track_sha256'];validate_mot(s['track_path'],frames,s['track_rows']);coverage[sequence]=s
    output=base/'evaluation'/label/case_id;assert not output.exists();output.mkdir(parents=True)
    values=run_official_tracks(base/'streams'/label/case_id/'track_results',output,plan,split);registered()
    frozen_json(output/'metrics.json',dict(status='COMPLETE_OFFICIAL_LEDGER_ALL47',case=case,frames=sum(LENGTHS[split].values()),
        coverage=coverage,source_versions=reg['source_hashes'],model_sha256=fit['model_sha256'],**values,goal_met=False,supported_innovations=0))


def phase(queue,split,stage='normal'):
    plan,protocol,work,base,train=locations();cases=split_cases(split,stage);label=split if stage=='normal'else split+'_intervention'
    cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))];extra=['--split',split,'--stage',stage]
    queue.jobs(PREFIX+'_'+label.upper()+'_REPLAY',[dict(id=s,command=cmd+['replay',*extra,'--sequence',s])for s in LENGTHS[split]],False,
        lambda j:read(base/'streams'/label/'sequence_manifests'/(j['id']+'.json'))['original_references_or_missing_RGB_MOT_exact'])
    queue.jobs(PREFIX+'_'+label.upper()+'_OFFICIAL',[dict(id=c['id'],command=cmd+['evaluate',*extra,'--case',c['id']])for c in cases],False,
        lambda j:read(base/'evaluation'/label/j['id']/'metrics.json')['frames']==sum(LENGTHS[split].values()))
    return [read(base/'evaluation'/label/c['id']/'metrics.json')for c in cases]


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();plan,protocol,work,base,train=locations()
    with(base/'state.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(base/'state.json').exists()
        state=dict(status='STARTING_LEDGER_FULL_DEV',controller_pid=os.getpid(),phases={},seed=0,new_GPU_work=False,goal_met=False,supported_innovations=0)
        queue=Queue(work,base,state,plan['data_root'],registered);queue.save();cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))]
        try:
            queue.jobs(PREFIX+'_ENGINEERING',[dict(id='all6',command=cmd+['engineer'])],False,
                lambda j:read(base/'real_Train_engineering.json')['sequences']==6)
            rows=phase(queue,'dev');old=read(work/'archive_reasoning_evaluation_20261127/complete_Dev.json')
            for own,previous in [('IL_native_control','AR_native_zero'),('IL_archive_control','AR_RGB_q999_shape')]:
                a=next(r for r in rows if r['case']['id']==own);b=next(r for r in old['rows']if r['case']['id']==previous)
                assert a['combined47']==b['combined47']and a['per_sequence47']==b['per_sequence47']
            chosen=min(rows,key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),r['case']['id']))
            gain=chosen['combined47']['HOTA']-reg['parent_best_Dev_HOTA']
            frozen_json(base/'complete_Dev.json',dict(status='COMPLETE_ALL16_NORMAL_LEDGER_FULL_DEV',UTC=utc(),rows=rows,chosen_case=chosen['case'],
                gain_over_parent_best_Dev_pp=gain,parent_best_Dev_HOTA=reg['parent_best_Dev_HOTA'],Test_read_for_selection=False,
                references_all47_and_MOT_exact=True,historical_Test_previously_exposed=True,goal_met=False,supported_innovations=0))
            if gain<=1e-8:
                state.update(status='COMPLETE_LEDGER_NO_DEV_GAIN_NO_NEW_TEST',Dev_gain_pp=gain);queue.save();return
            case=chosen['case'];opposite=matched(case,'critic'if case['kind']=='RGB'else'RGB')
            frozen_json(base/'Test_selection_frozen.json',dict(status='FROZEN_LEDGER_FULL_DEV_SELECTION',UTC=utc(),
                cases=policies()[:2]+[case,opposite],chosen_case=case,matched_identity_control=opposite,
                Dev_interventions=interventions(case),complete_Dev_sha256=sha256(base/'complete_Dev.json'),
                Test_read_for_selection=False,historical_Test_previously_exposed=True))
            controls=phase(queue,'dev','intervention');rgb=next(r for r in rows if r['case']==matched(case,'RGB'))
            assert controls[0]['combined47']==rgb['combined47']and controls[0]['per_sequence47']==rgb['per_sequence47']
            frozen_json(base/'complete_Dev_interventions.json',dict(status='COMPLETE_FROZEN_LEDGER_DEV_CONTROLS',UTC=utc(),rows=controls,
                missing_IR_MOT_and_all47_exact_RGB=True,no_reselection=True,Test_read_for_selection=False))
            # Independent full-Train fit verification must finish before Test.
            while True:
                ae=read(train/'audit_r1.execution.json')
                if ae['actual_child_exit_code']is not None:break
                live=identity(ae['identity']['PID']);assert live is not None,'Preserve unknown audit exit instead of claiming success'
                assert all(live[k]==ae['identity'][k]for k in ['PID','start_ticks','raw_argv_sha256'])
                state.update(status='WAITING_ACTUAL_INDEPENDENT_LEDGER_TRAIN_AUDIT',last_wait_UTC=utc());queue.save()
                registered();guard_storage(work);time.sleep(5)
            audit=read(train/'independent_fit_audit.json')
            assert ae['actual_child_exit_code']==0 and ae['source_unchanged']and identity(ae['identity']['PID'])is None
            assert audit['all6_Train_replayed']and audit['model_sha256']==read(train/'fit_complete.json')['model_sha256']
            test=phase(queue,'test');selected=next(r for r in test if r['case']==case)
            csv=work/'native_pyramid_evaluation/test/NP_source_canonical_RGB/trackeval_output/pedestrian_detailed.csv'
            assert sha256(csv)=='c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
            schema,_=official_schema(plan['evaluation']['TrackEval_root'])
            original=official_csv47(csv,schema,'test',read(csv.parent.parent/'metrics.json')['trackeval'])
            delta=selected['combined47']['HOTA']-original['combined47']['HOTA']
            frozen_json(base/'full_Test_results.json',dict(status='COMPLETE_DEV_FROZEN_LEDGER_ALL7_TEST',UTC=utc(),rows=test,selected_case=case,
                original_RGB_baseline=original,HOTA_gain_over_original_RGB_pp=delta,target_plus5_met=delta>=5,target_plus10_met=delta>=10,
                Test_used_for_selection=False,historical_Test_previously_exposed=True,goal_met=False,supported_innovations=0))
            state.update(status='COMPLETE_LEDGER_FULL_TEST_REQUIRES_ASSESSMENT',HOTA_gain_over_original_RGB_pp=delta);queue.save()
        except Exception as error:
            state.update(status='FAILED_PRESERVE_LEDGER_EVALUATION_ATTEMPT',error=repr(error));queue.save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','run','engineer','replay','evaluate'])
    p.add_argument('--split',choices=['dev','test']);p.add_argument('--stage',choices=['normal','intervention'],default='normal');p.add_argument('--sequence');p.add_argument('--case');a=p.parse_args()
    if a.mode=='register':register()
    elif a.mode=='run':run()
    elif a.mode=='engineer':engineer()
    elif a.mode=='replay':replay(a.split,a.stage,a.sequence)
    else:evaluate(a.split,a.stage,a.case)


if __name__=='__main__':main()
