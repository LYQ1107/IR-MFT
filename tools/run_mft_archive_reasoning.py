#!/usr/bin/env python3
"""Register archive controls, wait for actual fits, then select on full Dev."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.fit_mft_archive_reasoning import registered as Train_registered,locations as Train_locations
from tools.fit_mft_candidate_risk import threshold_values
from tools.mft_archive_reasoning import ArchiveLogistic
from tools.mft_archive_evaluation import ArchiveTracker,policies,matched_case,interventions
from tools.mft_candidate_risk_teacher import CandidateTeacher
from tools.mft_pose_continuity import PoseContinuity,policies as pose_policies
from tools.run_mft_selective_protection import registered as selective_registered
from tools.run_mft_candidate_risk import identity_score_hash
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.run_mft_candidate_train_bank import atomic_json
from tools.run_mft_native_memory_priority_v3 import identity
from tools.run_pseudo_ir_comparison import sha256
from tools.task_utility_queue import Queue
from tools.fusion_memory_observation import LENGTHS
from tools.fusion_memory_streams import FusionMemoryStreamWriter
from tools.fusion_memory_evaluation import run_official_tracks,validate_mot,official_csv47
from tools.export_perceptual_metrics47 import official_schema
from tools.audit_mft_tracklet_r2 import bank_audit
from tools.run_mft_tracklet_permanence_r2 import registered as peer_registered
from tools.mft_candidate_risk_data import load_arrays
from tools.native_pyramid_stream_tracks import mot_line

SPEC=ROOT/'configs/perceptual_archive_evaluation_20261127.json'
PREFIX='ARCHIVE_REASONING'


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def locations():
    plan,protocol,work,train,risk=Train_locations()
    return plan,protocol,work,work/'archive_reasoning_evaluation_20261127',train,work/'selective_protection_20261127'


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    train_reg=Train_registered();selective=selective_registered();plan,protocol,work,base,train,sp=locations()
    proof=read(base/'CPU_regressions_r1.json');assert proof['actual_tool_exit_code']==0 and proof['tests_passed']==3
    sources=dict(train_reg['source_hashes'])
    for n,h in selective['source_hashes'].items():
        assert n not in sources or sources[n]==h;sources[n]=h
    for n,h in proof['source_hashes'].items():assert sha256(ROOT/n)==h,n;sources[n]=h
    for n in ['tools/run_mft_archive_reasoning.py',str(SPEC.relative_to(ROOT))]:sources[n]=sha256(ROOT/n)
    spec=read(SPEC);assert spec['complete_Dev_cases']==policies()and len(policies())==11
    files=[SPEC,base/'CPU_regressions_r1.json',base/'CPU_regressions_r1.stdout.txt',
        train/'fit_registered.json',sp/'pipeline_registered.json']
    v=dict(status='REGISTERED_ARCHIVE_FULL_DEV_CONTROLS_BEFORE_NEW_DEV',UTC=utc(),source_hashes=sources,
        artifact_hashes={str(p):sha256(p)for p in files},normal_cases=policies(),
        specification_sha256=sha256(SPEC),Train_fit_identity=read(train/'fit_r1.execution.json')['identity'],
        selective_controller_identity=read(sp/'controller.execution.json')['identity'],
        archive_fit_complete_at_registration=(train/'fit_complete.json').exists(),
        new_Dev_complete_at_registration=(base/'complete_Dev.json').exists(),
        current_risk_or_selective_Test_scores_read_for_choice=False,
        parent_Dev_gate='Strictly improve maximum of frozen pose,complete risk and complete selective Dev best HOTA',
        fixed_seed=0,new_GPU_work=False,historical_Test_previously_exposed=True,goal_met=False,supported_innovations=0)
    assert not(base/'pipeline_registered.json').exists();frozen_json(base/'pipeline_registered.json',v)
    print(json.dumps(dict(status=v['status'],cases=11,sources=len(sources))),flush=True)


def registered():
    Train_registered();selective_registered();plan,protocol,work,base,train,sp=locations()
    r=read(base/'pipeline_registered.json');assert r['normal_cases']==policies()and r['specification_sha256']==sha256(SPEC)
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in r['artifact_hashes'].items():assert sha256(n)==h,n
    return r


def fitted():
    reg=registered();plan,protocol,work,base,train,sp=locations()
    e=read(train/'fit_r1.execution.json');assert e['identity']==reg['Train_fit_identity']
    assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(e['identity']['PID'])is None
    f=read(train/'fit_complete.json');assert f['Train_frames']==27251 and f['Train_sequences']==6
    assert f['Dev_rows']==f['Test_rows']==0 and f['all_teacher_MOT_exact']and not f['truncated_pairs']
    assert sha256(f['model_file'])==f['model_sha256']and f['registration_sha256']==sha256(train/'fit_registered.json')
    private=read(f['model_file']);models={k:ArchiveLogistic.from_dict(v)for k,v in private['models'].items()}
    assert private['thresholds']==f['thresholds']and len(private['history'])==6
    return reg,f,models,f['thresholds']


def eval_register():
    reg,fit,models,thresholds=fitted();plan,protocol,work,base,train,sp=locations()
    e=read(sp/'controller.execution.json')
    assert e['identity']==reg['selective_controller_identity']and e['actual_child_exit_code']==0 and e['source_unchanged']
    assert identity(e['identity']['PID'])is None
    refs=[];paths=[]
    for folder in ['pose_continuity_20261127','candidate_risk_20261127','selective_protection_20261127']:
        p=work/folder/'complete_Dev.json';value=read(p);assert value['Test_read_for_selection']is False
        chosen=min(value['rows'],key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),r['case']['id']))
        refs.append(dict(study=folder,case=chosen['case'],combined47=chosen['combined47']));paths.append(p)
    gate=max(r['combined47']['HOTA']for r in refs)
    banks={s:bank_audit(work/'tracklet_permanence_20261127_r2','dev',s,peer_registered())for s in LENGTHS['dev']}
    paths.extend([train/'fit_complete.json',train/'fit_r1.execution.json',sp/'controller.execution.json'])
    value=dict(status='REGISTERED_FROZEN_ARCHIVE_MODELS_AND_COMPLETE_PRIOR_DEV_GATE',UTC=utc(),
        pipeline_registration_sha256=sha256(base/'pipeline_registered.json'),model_sha256=fit['model_sha256'],
        artifact_hashes={str(p):sha256(p)for p in paths},normal_cases=policies(),Dev_banks=banks,
        parent_complete_Dev_references=refs,parent_best_Dev_HOTA=gate,Dev_frames=4834,
        Train_model_coefficients_and_normalization_frozen=True,new_Dev_labels_used_for_fit=False,
        Test_scores_used=False,historical_Test_previously_exposed=True)
    assert not(base/'evaluation_registered.json').exists();frozen_json(base/'evaluation_registered.json',value)
    print(json.dumps(dict(status=value['status'],parent_Dev_HOTA=gate)),flush=True)


def ready():
    reg,fit,models,thresholds=fitted();plan,protocol,work,base,train,sp=locations();ev=read(base/'evaluation_registered.json')
    assert ev['pipeline_registration_sha256']==sha256(base/'pipeline_registered.json')and ev['model_sha256']==fit['model_sha256']
    for n,h in ev['artifact_hashes'].items():assert sha256(n)==h,n
    return reg,fit,models,thresholds,ev


def make_tracker(case,models,thresholds,work):
    speed=threshold_values(work)['Train_unknown_speed_upper']
    if case['kind']=='reference_native':return CandidateTeacher(speed)
    if case['kind']in ['reference_late','reference_pose']:
        motion=read(work/'pose_continuity_20261127/Train_motion_statistics.json')
        return PoseContinuity(pose_policies()[1 if case['kind']=='reference_late'else 5],threshold_values(work),
            motion['equal_video_RMS_acceleration_sigma'])
    return ArchiveTracker(case,models,thresholds,speed)


def prefix_rows(arrays,cases,models,thresholds,work,sequence,count):
    trackers={c['id']:make_tracker(c,models,thresholds,work)for c in cases};rows={c['id']:[]for c in cases}
    for i,f in enumerate(arrays['frame_ids'][:count]):
        lo,hi=map(int,arrays['offsets'][i:i+2])
        for c in cases:rows[c['id']].extend(trackers[c['id']].update(sequence,int(f),arrays['detections'][lo:hi],
            arrays['rgb'][lo:hi],arrays['ir'][lo:hi]))
    return {k:''.join(map(mot_line,v))for k,v in rows.items()}


def engineer():
    reg,fit,models,thresholds,ev=ready();plan,protocol,work,base,train,sp=locations();rows={}
    for sequence,q in Train_registered()['qualified_Train'].items():
        arrays=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);count=min(64,q['frames']);cut=32
        full=prefix_rows(arrays,policies(),models,thresholds,work,sequence,count)
        prefix=prefix_rows(arrays,policies(),models,thresholds,work,sequence,cut)
        assert all(full[k].startswith(prefix[k])for k in full)
        at=int(arrays['offsets'][cut]);arrays['detections'][at:,:4]+=777
        for cue in ['rgb','ir']:arrays[cue][at:]*=-1
        future=prefix_rows(arrays,policies(),models,thresholds,work,sequence,count)
        assert all(future[k].startswith(prefix[k])for k in future)
        rows[sequence]=dict(frames=count,prefix_frames=cut,all11_cases_causal_prefix_exact=True,bank_sha256=q['bank_sha256'])
        guard_storage(work)
    assert sha256(fit['model_file'])==fit['model_sha256'];registered()
    frozen_json(base/'real_Train_engineering.json',dict(status='COMPLETE_FITTED_ARCHIVE_ALL6_REAL_TRAIN_ENGINEERING',
        UTC=utc(),rows=rows,sequences=6,model_sha256=fit['model_sha256'],model_bytes_unchanged=True,new_accuracy=False))


def split_cases(split,stage):
    plan,protocol,work,base,train,sp=locations()
    if split=='test':
        f=read(base/'Test_selection_frozen.json');assert f['Test_read_for_selection']is False
        assert f['complete_Dev_sha256']==sha256(base/'complete_Dev.json');return f['cases']
    assert split=='dev'
    if stage=='normal':return policies()
    assert stage=='intervention';return read(base/'Test_selection_frozen.json')['Dev_interventions']


def replay(split,stage,sequence):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''and sequence in LENGTHS[split]
    reg,fit,models,thresholds,ev=ready();plan,protocol,work,base,train,sp=locations();cases=split_cases(split,stage)
    peer=work/'tracklet_permanence_20261127_r2';bank=bank_audit(peer,split,sequence,peer_registered())
    meta=read(peer/'cache'/split/(sequence+'.json'))
    with np.load(meta['path'],allow_pickle=False)as data:arrays={k:data[k].copy()for k in data.files}
    label=split if stage=='normal'else split+'_intervention';count=32
    prefix=prefix_rows(arrays,cases,models,thresholds,work,sequence,count)
    at=int(arrays['offsets'][count]);future=arrays['detections'][at:].copy();arrays['detections'][at:,:4]+=777
    for cue in ['rgb','ir']:arrays[cue][at:]*=-1
    assert prefix_rows(arrays,cases,models,thresholds,work,sequence,count)==prefix
    arrays['detections'][at:]=future
    for cue in ['rgb','ir']:arrays[cue][at:]*=-1
    writer=FusionMemoryStreamWriter(base/'streams'/label,sequence,[c['id']for c in cases])
    trackers={c['id']:make_tracker(c,models,thresholds,work)for c in cases};started=time.monotonic()
    try:
        for i,f in enumerate(arrays['frame_ids']):
            lo,hi=map(int,arrays['offsets'][i:i+2]);d=arrays['detections'][lo:hi]
            rows={c['id']:trackers[c['id']].update(sequence,int(f),d,arrays['rgb'][lo:hi],arrays['ir'][lo:hi])for c in cases}
            writer.append(int(f),rows,{c['id']:d for c in cases})
            if int(f)%100==0 or int(f)==LENGTHS[split][sequence]:
                guard_storage(work);atomic_json(base/'progress'/label/(sequence+'.json'),dict(frame=int(f),
                    frames=LENGTHS[split][sequence],seconds=time.monotonic()-started,new_GPU_work=False))
        streams=writer.close(LENGTHS[split][sequence])
    finally:writer.abort()
    if stage=='normal':
        pose=read(work/'pose_continuity_20261127/streams'/split/'sequence_manifests'/(sequence+'.json'))
        for own,old in [('AR_native_zero','PC_native_zero'),('AR_fixed_RGB_late','PC_late_only'),
                ('AR_fixed_RGB_late_shape','PC_shape_anchor_late')]:
            if own in streams:assert streams[own]['track_sha256']==pose['streams'][old]['track_sha256']
    else:
        freeze=read(base/'Test_selection_frozen.json');rgb=matched_case(freeze['chosen_case'],'RGB')['id']
        normal=read(base/'streams/dev/sequence_manifests'/(sequence+'.json'))
        assert streams['AR_intervention_missing']['track_sha256']==normal['streams'][rgb]['track_sha256']
    shapes=[]
    for c in cases:
        if c['kind']in ['RGB','paired']and c['shape']and not c.get('intervention'):
            filtered=next((p for p in cases if p['kind']==c['kind']and p['quantile']==c['quantile']and not p['shape']),None)
            if filtered is not None:
                a=identity_score_hash(streams[c['id']]['track_path']);b=identity_score_hash(streams[filtered['id']]['track_path'])
                assert a==b;shapes.append(dict(current_shape=c['id'],filtered=filtered['id'],identity_score_order_hash=a))
    assert all(v['detection_packet_sha256']==bank['packet_sha256']for v in streams.values())
    assert sha256(fit['model_file'])==fit['model_sha256'];registered()
    frozen_json(base/'streams'/label/'sequence_manifests'/(sequence+'.json'),dict(
        status='COMPLETE_CAUSAL_ARCHIVE_FULL_SEQUENCE',UTC=utc(),split=split,stage=stage,sequence=sequence,
        frames=LENGTHS[split][sequence],cases=cases,streams=streams,bank=bank,source_versions=reg['source_hashes'],
        model_sha256=fit['model_sha256'],statistics={k:t.statistics for k,t in trackers.items()},
        original_references_or_RGB_missing_IR_MOT_exact=True,shape_identity_score_order_checks=shapes,
        future_suffix_prefix_exact=True,current_frame_inputs_only=True,GT_read=False,previous_outputs_rewritten=False,
        model_bytes_unchanged=True,compact_feature_cache_read=True,
        cached_CPU_replay_seconds=time.monotonic()-started,neural_and_image_cost_excluded=True))


def evaluate(split,stage,case_id):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg,fit,models,thresholds,ev=ready();plan,protocol,work,base,train,sp=locations()
    case=next(c for c in split_cases(split,stage)if c['id']==case_id);label=split if stage=='normal'else split+'_intervention';coverage={}
    for sequence,frames in LENGTHS[split].items():
        m=read(base/'streams'/label/'sequence_manifests'/(sequence+'.json'))
        assert m['frames']==frames and m['source_versions']==reg['source_hashes']and m['model_sha256']==fit['model_sha256']
        assert m['original_references_or_RGB_missing_IR_MOT_exact']and m['future_suffix_prefix_exact']
        s=m['streams'][case_id];assert sha256(s['track_path'])==s['track_sha256'];validate_mot(s['track_path'],frames,s['track_rows'])
        coverage[sequence]=s
    output=base/'evaluation'/label/case_id;assert not output.exists();output.mkdir(parents=True)
    values=run_official_tracks(base/'streams'/label/case_id/'track_results',output,plan,split);registered()
    frozen_json(output/'metrics.json',dict(status='COMPLETE_OFFICIAL_ARCHIVE_ALL47',case=case,
        frames=sum(LENGTHS[split].values()),coverage=coverage,source_versions=reg['source_hashes'],
        model_sha256=fit['model_sha256'],**values,goal_met=False,supported_innovations=0))


def run_phase(queue,split,stage='normal'):
    plan,protocol,work,base,train,sp=locations();cases=split_cases(split,stage);label=split if stage=='normal'else split+'_intervention'
    cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))];extra=['--split',split,'--stage',stage]
    queue.jobs(PREFIX+'_'+label.upper()+'_REPLAY',[dict(id=s,command=cmd+['replay',*extra,'--sequence',s])for s in LENGTHS[split]],
        False,lambda j:read(base/'streams'/label/'sequence_manifests'/(j['id']+'.json'))['original_references_or_RGB_missing_IR_MOT_exact'])
    queue.jobs(PREFIX+'_'+label.upper()+'_OFFICIAL',[dict(id=c['id'],command=cmd+['evaluate',*extra,'--case',c['id']])for c in cases],
        False,lambda j:read(base/'evaluation'/label/j['id']/'metrics.json')['frames']==sum(LENGTHS[split].values()))
    return [read(base/'evaluation'/label/c['id']/'metrics.json')for c in cases]


def wait_predecessors(queue):
    reg=registered();plan,protocol,work,base,train,sp=locations()
    while True:
        registered();guard_storage(work);pending=[]
        for name,path,expected in [('archive_Train_fit',train/'fit_r1.execution.json',reg['Train_fit_identity']),
                ('selective_controller',sp/'controller.execution.json',reg['selective_controller_identity'])]:
            e=read(path);assert e['identity']==expected
            if e['actual_child_exit_code']is not None:
                assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(expected['PID'])is None
            else:
                live=identity(expected['PID'])
                if live is not None:assert all(live[k]==expected[k]for k in ['PID','start_ticks','raw_argv_sha256'])
                pending.append(dict(name=name,live=live))
        if not pending:return
        queue.state.update(status='WAITING_ACTUAL_ARCHIVE_FIT_AND_COMPLETE_SELECTIVE_STUDY',pending=pending,last_wait_UTC=utc());queue.save();time.sleep(15)


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();plan,protocol,work,base,train,sp=locations()
    with(base/'state.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(base/'state.json').exists()
        state=dict(status='STARTING_ARCHIVE_FULL_DEV',controller_pid=os.getpid(),phases={},
            new_GPU_work=False,random_seed=0,goal_met=False,supported_innovations=0)
        queue=Queue(work,base,state,plan['data_root'],registered);queue.save();cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))]
        try:
            wait_predecessors(queue)
            queue.jobs(PREFIX+'_EVAL_REGISTER',[dict(id='all6',command=cmd+['eval-register'])],False,
                lambda j:len(read(base/'evaluation_registered.json')['normal_cases'])==11)
            queue.jobs(PREFIX+'_ENGINEERING',[dict(id='all6',command=cmd+['engineer'])],False,
                lambda j:read(base/'real_Train_engineering.json')['sequences']==6)
            values=run_phase(queue,'dev');ev=read(base/'evaluation_registered.json')
            pose=read(work/'pose_continuity_20261127/complete_Dev.json')
            for r,index in zip(values[:3],[0,1,5]):
                old=pose['rows'][index];assert r['combined47']==old['combined47']and r['per_sequence47']==old['per_sequence47']
            chosen=min(values,key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),r['case']['id']))
            gain=chosen['combined47']['HOTA']-ev['parent_best_Dev_HOTA']
            frozen_json(base/'complete_Dev.json',dict(status='COMPLETE_ALL11_ARCHIVE_NORMAL_DEV',UTC=utc(),rows=values,
                chosen_case=chosen['case'],parent_best_Dev_HOTA=ev['parent_best_Dev_HOTA'],gain_over_parent_best_Dev_pp=gain,
                Test_read_for_selection=False,current_parent_Test_scores_used=False,all3_reference_MOT_and_all47_exact=True,
                model_sha256=read(train/'fit_complete.json')['model_sha256'],goal_met=False,supported_innovations=0))
            if gain<=1e-8:
                state.update(status='COMPLETE_ARCHIVE_NO_DEV_GAIN_NO_NEW_TEST',Dev_gain_pp=gain);queue.save();return
            assert chosen['case']['kind']in ['RGB','paired']
            opposite=matched_case(chosen['case'],'paired'if chosen['case']['kind']=='RGB'else'RGB')
            cases=[policies()[0],policies()[2],chosen['case'],opposite]
            frozen_json(base/'Test_selection_frozen.json',dict(status='FROZEN_ARCHIVE_FULL_DEV_SELECTION_AND_CONTROLS',UTC=utc(),
                cases=cases,chosen_case=chosen['case'],matched_identity_control=opposite,
                Dev_interventions=interventions(chosen['case']),complete_Dev_sha256=sha256(base/'complete_Dev.json'),
                Test_read_for_selection=False,current_parent_Test_scores_used=False,historical_Test_previously_exposed=True))
            control=run_phase(queue,'dev','intervention')
            rgb=next(r for r in values if r['case']==matched_case(chosen['case'],'RGB'))
            assert control[0]['combined47']==rgb['combined47']and control[0]['per_sequence47']==rgb['per_sequence47']
            frozen_json(base/'complete_Dev_interventions.json',dict(status='COMPLETE_FROZEN_ARCHIVE_DEV_INTERVENTIONS',UTC=utc(),
                rows=control,matched_RGB_case=rgb['case'],missing_IR_all47_and_MOT_exact_RGB=True,
                no_reselection_after_controls=True,Test_read_for_selection=False))
            test=run_phase(queue,'test');selected=next(r for r in test if r['case']==chosen['case'])
            csv=work/'native_pyramid_evaluation/test/NP_source_canonical_RGB/trackeval_output/pedestrian_detailed.csv'
            assert sha256(csv)=='c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
            schema,_=official_schema(plan['evaluation']['TrackEval_root']);original=official_csv47(csv,schema,'test',read(csv.parent.parent/'metrics.json')['trackeval'])
            assert abs(original['combined47']['HOTA']-29.528970687393972)<1e-10
            delta=selected['combined47']['HOTA']-original['combined47']['HOTA']
            frozen_json(base/'full_Test_results.json',dict(status='COMPLETE_DEV_FROZEN_ARCHIVE_FULL7_TEST',UTC=utc(),rows=test,
                original_RGB_baseline=original,selected_case=chosen['case'],matched_identity_control=opposite,
                HOTA_gain_over_original_RGB_pp=delta,target_plus5_met=delta>=5,target_plus10_met=delta>=10,
                Test_used_for_selection=False,historical_Test_previously_exposed=True,goal_met=False,supported_innovations=0))
            state.update(status='COMPLETE_ARCHIVE_FULL_TEST_REQUIRES_SCIENTIFIC_ASSESSMENT',HOTA_gain_over_original_RGB_pp=delta);queue.save()
        except Exception as error:
            state.update(status='FAILED_PRESERVE_ARCHIVE_EVALUATION_ATTEMPT',error=repr(error));queue.save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['register','run','eval-register','engineer','replay','evaluate'])
    p.add_argument('--split',choices=['dev','test']);p.add_argument('--stage',choices=['normal','intervention'],default='normal')
    p.add_argument('--sequence');p.add_argument('--case');a=p.parse_args()
    if a.mode=='register':register()
    elif a.mode=='run':run()
    elif a.mode=='eval-register':eval_register()
    elif a.mode=='engineer':engineer()
    elif a.mode=='replay':replay(a.split,a.stage,a.sequence)
    else:evaluate(a.split,a.stage,a.case)


if __name__=='__main__':main()
