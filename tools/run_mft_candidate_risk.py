#!/usr/bin/env python3
"""Wait for all real Train banks; fit once, freeze full-Dev choice, then Test."""
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
from tools.fit_mft_candidate_risk import locations,require_planning,threshold_values,SPEC
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_candidate_train_bank import registered as bank_registered,atomic_json
from tools.run_mft_native_memory_priority_v3 import identity
from tools.task_utility_queue import Queue
from tools.fusion_memory_observation import LENGTHS
from tools.fusion_memory_streams import FusionMemoryStreamWriter
from tools.fusion_memory_evaluation import run_official_tracks,validate_mot
from tools.audit_mft_tracklet_r2 import bank_audit
from tools.run_mft_tracklet_permanence_r2 import registered as peer_registered
from tools.mft_candidate_risk_association import RiskAssociation,policies
from tools.mft_candidate_risk_model import RiskLogistic
from tools.mft_candidate_risk_teacher import CandidateTeacher
from tools.mft_pose_continuity import PoseContinuity,policies as pose_policies
from tools.native_pyramid_stream_tracks import mot_line

PREFIX='CANDIDATE_RISK'


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    parent=bank_registered();plan,protocol,work,base=locations()
    spec=read(SPEC);assert spec['complete_Dev_cases']==policies()
    proof=read(base/'CPU_regressions_r1.json')
    assert proof['actual_tool_exit_code']==0 and proof['tests_passed']==22
    sources=dict(parent['source_hashes'])
    for name,digest in proof['source_hashes'].items():
        assert sha256(ROOT/name)==digest,name;sources[name]=digest
    for name in ['tools/run_mft_candidate_risk.py','tools/fit_mft_candidate_risk.py',str(SPEC.relative_to(ROOT))]:
        assert name in proof['source_hashes']
    paths=[base/'CPU_regressions_r1.json',base/'CPU_regressions_r1.stdout.txt',
        work/'candidate_train_bank_20261127/pipeline_registered.json',
        work/'tracklet_permanence_20261127_r2/Train_thresholds.json',
        work/'pose_continuity_20261127/Train_motion_statistics.json',
        work/'pose_continuity_20261127/complete_Dev.json']
    guard_storage(work)
    value=dict(status='REGISTERED_CAUSAL_RISK_PLAN_BEFORE_FIT_OR_NEW_DEV',UTC=utc(),
        source_hashes=sources,artifact_hashes={str(p):sha256(p)for p in paths},
        specification_sha256=sha256(SPEC),normal_cases=policies(),fixed_seed=0,
        fixed_original_RGB_Test_HOTA=29.528970687393972,minimum_target_Test_HOTA=34.52897068739397,
        stretch_target_Test_HOTA=39.52897068739397,historical_Test_previously_exposed=True,
        this_stage_designed_after_prior_Test_review=True,prior_pose_frozen_Test_HOTA=29.955069130617183,
        train_bank_must_close_before_engineering_or_fit=True,all6_Train_required=True,
        no_new_NN_training=True,new_GPU_work=False,no_candidate_pair_disk_dump=True,
        maximum_cards=2,GPU_work_owner='Existing unchanged candidate Train bank GPU4/5',
        new_tracking_accuracy=False,goal_met=False,supported_innovations=0)
    assert not(base/'pipeline_registered.json').exists();frozen_json(base/'pipeline_registered.json',value)
    print(json.dumps(dict(status=value['status'],sources=len(sources),cases=len(policies()))),flush=True)


def require_fitted():
    reg=require_planning();plan,protocol,work,base=locations()
    state=read(base/'state.json');job=state['phases'][PREFIX+'_CPU_FIT']['fit']
    assert job['status']=='COMPLETE'and job['exit_code']==0 and identity(job['pid'])is None
    fit=read(base/'fit_complete.json');assert fit['Dev_rows']==fit['Test_rows']==0
    assert fit['complete_passes']==8 and fit['optimizer_passes']==6 and fit['early_weights_retained']
    assert sha256(fit['model_file'])==fit['model_sha256']
    value=read(fit['model_file'])
    assert value['source_registration_sha256']==sha256(base/'fit_registered.json')
    models={k:RiskLogistic.from_dict(v)for k,v in value['models'].items()}
    return reg,fit,models,value['thresholds']


def split_cases(split):
    assert split in ['dev','test']
    plan,protocol,work,base=locations()
    if split=='dev':return policies()
    freeze=read(base/'Test_selection_frozen.json')
    assert freeze['Test_read_for_selection']is False
    assert freeze['complete_Dev_sha256']==sha256(base/'complete_Dev.json')
    return freeze['cases']


def make_tracker(case,models,calibration,work):
    thresholds=threshold_values(work)
    if case['kind']=='reference_native':return CandidateTeacher(thresholds['Train_unknown_speed_upper'])
    if case['kind']=='reference_pose_late':
        motion=read(work/'pose_continuity_20261127/Train_motion_statistics.json')
        return PoseContinuity(pose_policies()[5],thresholds,motion['equal_video_RMS_acceleration_sigma'])
    return RiskAssociation(case,models,calibration,thresholds['Train_unknown_speed_upper'])


def prefix_rows(arrays,cases,models,calibration,work,sequence,count,missing_IR=False):
    trackers={c['id']:make_tracker(c,models,calibration,work)for c in cases};rows={c['id']:[]for c in cases}
    for i,f in enumerate(arrays['frame_ids'][:count]):
        lo,hi=map(int,arrays['offsets'][i:i+2])
        for c in cases:
            rows[c['id']].extend(trackers[c['id']].update(sequence,int(f),arrays['detections'][lo:hi],
                arrays['rgb'][lo:hi],None if missing_IR else arrays['ir'][lo:hi]))
    return {k:''.join(map(mot_line,v))for k,v in rows.items()}


def identity_score_hash(path):
    result=hashlib.sha256()
    for line in Path(path).read_text().splitlines(keepends=True):
        fields=line.split(',');result.update(','.join([*fields[:2],*fields[6:]]).encode())
    return result.hexdigest()


def replay(split,sequence):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''and sequence in LENGTHS[split]
    reg,fit,models,calibration=require_fitted();plan,protocol,work,base=locations()
    cases=split_cases(split);peer=work/'tracklet_permanence_20261127_r2'
    bank=bank_audit(peer,split,sequence,peer_registered())
    meta=read(peer/'cache'/split/(sequence+'.json'))
    with np.load(meta['path'],allow_pickle=False)as data:arrays={k:data[k].copy()for k in data.files}
    count=min(32,LENGTHS[split][sequence]);prefix=prefix_rows(arrays,cases,models,calibration,work,sequence,count)
    cut=int(arrays['offsets'][count]);future=arrays['detections'][cut:].copy()
    arrays['detections'][cut:,:4]+=777
    for cue in ['rgb','ir']:arrays[cue][cut:]*=-1
    assert prefix_rows(arrays,cases,models,calibration,work,sequence,count)==prefix
    arrays['detections'][cut:]=future
    for cue in ['rgb','ir']:arrays[cue][cut:]*=-1
    # This is a prefix engineering check, not an extra Dev/Test accuracy arm.
    fallback=prefix_rows(arrays,[policies()[2],policies()[3]],models,calibration,work,sequence,count,True)
    assert fallback['CR_RGB_risk']==fallback['CR_paired_risk']
    writer=FusionMemoryStreamWriter(base/'streams'/split,sequence,[c['id']for c in cases])
    trackers={c['id']:make_tracker(c,models,calibration,work)for c in cases};started=time.monotonic()
    try:
        for i,f in enumerate(arrays['frame_ids']):
            lo,hi=map(int,arrays['offsets'][i:i+2]);d=arrays['detections'][lo:hi]
            rows={c['id']:trackers[c['id']].update(sequence,int(f),d,arrays['rgb'][lo:hi],arrays['ir'][lo:hi])for c in cases}
            writer.append(int(f),rows,{c['id']:d for c in cases})
            if int(f)%100==0 or int(f)==LENGTHS[split][sequence]:
                guard_storage(work);atomic_json(base/'progress'/split/(sequence+'.json'),
                    dict(frame=int(f),frames=LENGTHS[split][sequence],CPU_only=True,seconds=time.monotonic()-started))
        streams=writer.close(LENGTHS[split][sequence])
    finally:writer.abort()
    pose=read(work/'pose_continuity_20261127/streams'/split/'sequence_manifests'/(sequence+'.json'))
    for ours,old in [('CR_native_zero','PC_native_zero'),('CR_pose_late_control','PC_shape_anchor_late')]:
        assert streams[ours]['track_sha256']==pose['streams'][old]['track_sha256']
    assert all(s['detection_packet_sha256']==bank['packet_sha256']for s in streams.values())
    shape_proof=None
    if split=='dev':
        a=identity_score_hash(streams['CR_paired_risk']['track_path'])
        b=identity_score_hash(streams['CR_paired_risk_shape']['track_path']);assert a==b
        shape_proof=dict(filtered_hash=a,current_shape_hash=b,frame_ID_score_order_exact=True,
            interpretation='Changed geometry can change GT matching and AssA even with identical internal identity decisions')
    require_planning()
    frozen_json(base/'streams'/split/'sequence_manifests'/(sequence+'.json'),dict(
        status='COMPLETE_CAUSAL_RISK_FULL_SEQUENCE_REPLAY',UTC=utc(),split=split,sequence=sequence,
        frames=LENGTHS[split][sequence],cases=cases,streams=streams,bank=bank,
        source_versions=reg['source_hashes'],model_sha256=fit['model_sha256'],
        statistics={k:t.statistics for k,t in trackers.items()},
        native_and_frozen_pose_control_MOT_exact=True,future_suffix_prefix_MOT_exact=True,
        missing_IR_prefix_MOT_exact_RGB_model_fallback=True,shape_identity_score_order_proof=shape_proof,
        only_current_frame_inputs=True,GT_read=False,previous_outputs_rewritten=False,
        compact_feature_cache_read=True,writer_local_false_cache_flag_scope='Only MOT is written; whole workflow reads existing compact NPZ',
        CPU_replay_seconds=time.monotonic()-started,neural_and_image_cost_excluded_from_replay=True))


def evaluate(split,case_id):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg,fit,models,calibration=require_fitted();plan,protocol,work,base=locations()
    case=next(c for c in split_cases(split)if c['id']==case_id);coverage={}
    for sequence,frames in LENGTHS[split].items():
        row=read(base/'streams'/split/'sequence_manifests'/(sequence+'.json'))
        assert row['frames']==frames and row['source_versions']==reg['source_hashes']and row['model_sha256']==fit['model_sha256']
        assert row['native_and_frozen_pose_control_MOT_exact']and row['future_suffix_prefix_MOT_exact']
        stream=row['streams'][case_id];assert sha256(stream['track_path'])==stream['track_sha256']
        validate_mot(stream['track_path'],frames,stream['track_rows']);coverage[sequence]=stream
    output=base/'evaluation'/split/case_id;assert not output.exists();output.mkdir(parents=True)
    values=run_official_tracks(base/'streams'/split/case_id/'track_results',output,plan,split)
    require_planning()
    frozen_json(output/'metrics.json',dict(status='COMPLETE_OFFICIAL_ALL_METRICS_CANDIDATE_RISK',case=case,
        frames=sum(LENGTHS[split].values()),coverage=coverage,source_versions=reg['source_hashes'],
        model_sha256=fit['model_sha256'],**values,goal_met=False,supported_innovations=0))


def run_phase(queue,split):
    plan,protocol,work,base=locations();cases=split_cases(split)
    cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))]
    queue.jobs(PREFIX+'_'+split.upper()+'_REPLAY',[dict(id=s,command=cmd+['replay','--split',split,'--sequence',s])
        for s in LENGTHS[split]],False,
        lambda j:read(base/'streams'/split/'sequence_manifests'/(j['id']+'.json'))['native_and_frozen_pose_control_MOT_exact'])
    queue.jobs(PREFIX+'_'+split.upper()+'_OFFICIAL',[dict(id=c['id'],command=cmd+['evaluate','--split',split,'--case',c['id']])
        for c in cases],False,lambda j:read(base/'evaluation'/split/j['id']/'metrics.json')['frames']==sum(LENGTHS[split].values()))
    return [read(base/'evaluation'/split/c['id']/'metrics.json')for c in cases]


def wait_Train(queue):
    plan,protocol,work,base=locations();bank=work/'candidate_train_bank_20261127'
    expected=read(bank/'controller.execution.json')['identity']
    queue.state.update(status='WAITING_EXISTING_TWO_GPU_ALL6_TRAIN_BANK',Train_GPU_owner=expected);queue.save()
    while True:
        require_planning();guard_storage(work)
        execution=read(bank/'controller.execution.json');assert execution['identity']==expected
        if execution['actual_child_exit_code']is not None:
            assert execution['actual_child_exit_code']==0 and execution['source_unchanged']
            assert identity(expected['PID'])is None
            from tools.mft_candidate_risk_data import qualify_Train
            qualify_Train(work,protocol);return
        live=identity(expected['PID'])
        if live is not None:assert all(live[k]==expected[k]for k in ['PID','start_ticks','raw_argv_sha256'])
        queue.state['last_Train_wait_UTC']=utc();queue.save();time.sleep(15)


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=require_planning();plan,protocol,work,base=locations()
    with(base/'state.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(base/'state.json').exists()
        state=dict(status='STARTING_CAUSAL_RISK_TRAIN_THEN_DEV',phases={},controller_pid=os.getpid(),
            data_root=plan['data_root'],new_GPU_work=False,fixed_seed=0,goal_met=False,supported_innovations=0)
        queue=Queue(work,base,state,plan['data_root'],require_planning);queue.save()
        fit_cmd=[sys.executable,'-u','tools/fit_mft_candidate_risk.py']
        try:
            wait_Train(queue)
            queue.jobs(PREFIX+'_CPU_ENGINEERING',[dict(id='all6',command=fit_cmd+['engineer'])],False,
                lambda j:read(base/'CPU_real_bank_engineering.json')['sequences']==6)
            queue.jobs(PREFIX+'_CPU_FIT_REGISTER',[dict(id='register',command=fit_cmd+['register-fit'])],False,
                lambda j:read(base/'fit_registered.json')['Train_frames']==27251)
            queue.jobs(PREFIX+'_CPU_FIT',[dict(id='fit',command=fit_cmd+['fit'])],False,
                lambda j:read(base/'fit_complete.json')['complete_passes']==8)
            values=run_phase(queue,'dev')
            pose=read(work/'pose_continuity_20261127/complete_Dev.json')
            for row,old in [(values[0],'PC_native_zero'),(values[1],'PC_shape_anchor_late')]:
                reference=next(r for r in pose['rows']if r['case']['id']==old)
                assert row['combined47']==reference['combined47']and row['per_sequence47']==reference['per_sequence47']
            chosen=min(values,key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),r['case']['id']))
            gain=chosen['combined47']['HOTA']-values[1]['combined47']['HOTA']
            frozen_json(base/'complete_Dev.json',dict(status='COMPLETE_ALL6_CANDIDATE_RISK_FULL_DEV',UTC=utc(),
                rows=values,chosen_case=chosen['case'],gain_over_frozen_pose_control_pp=gain,
                Test_read_for_selection=False,reference_all47_and_MOT_exact=True,goal_met=False,supported_innovations=0))
            if gain<=1e-8:
                state.update(status='COMPLETE_CANDIDATE_RISK_NO_DEV_GAIN_NO_NEW_TEST',Dev_gain_pp=gain);queue.save();return
            frozen_json(base/'Test_selection_frozen.json',dict(status='FROZEN_RISK_FROM_COMPLETE_DEV_BEFORE_NEW_TEST',UTC=utc(),
                cases=[policies()[0],policies()[1],chosen['case']],chosen_case=chosen['case'],
                complete_Dev_sha256=sha256(base/'complete_Dev.json'),Test_read_for_selection=False,
                historical_Test_previously_exposed=True,prior_pose_Test_previously_reviewed_by_agent=True))
            test=run_phase(queue,'test');selected=next(r for r in test if r['case']==chosen['case'])
            from tools.export_perceptual_metrics47 import official_schema
            from tools.fusion_memory_evaluation import official_csv47
            csv=work/'native_pyramid_evaluation/test/NP_source_canonical_RGB/trackeval_output/pedestrian_detailed.csv'
            assert sha256(csv)=='c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
            baseline=read(csv.parent.parent/'metrics.json');schema,_=official_schema(plan['evaluation']['TrackEval_root'])
            original=official_csv47(csv,schema,'test',baseline['trackeval'])
            assert abs(original['combined47']['HOTA']-29.528970687393972)<1e-10
            delta=selected['combined47']['HOTA']-original['combined47']['HOTA']
            frozen_json(base/'full_Test_results.json',dict(status='COMPLETE_DEV_FROZEN_RISK_ALL7_TEST',UTC=utc(),
                rows=test,original_RGB_baseline=original,selected_case=chosen['case'],
                HOTA_gain_over_original_RGB_pp=delta,target_plus5_met=delta>=5,target_plus10_met=delta>=10,
                Test_used_for_selection=False,historical_Test_previously_exposed=True,
                goal_met=False,supported_innovations=0))
            state.update(status='COMPLETE_CANDIDATE_RISK_FULL_TEST_REQUIRES_SCIENTIFIC_ASSESSMENT',
                HOTA_gain_over_original_RGB_pp=delta);queue.save()
        except Exception as error:
            state.update(status='FAILED_PRESERVE_CANDIDATE_RISK_ATTEMPT',error=repr(error));queue.save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','run','replay','evaluate'])
    p.add_argument('--split',choices=['dev','test']);p.add_argument('--sequence');p.add_argument('--case');args=p.parse_args()
    if args.mode=='register':register()
    elif args.mode=='run':run()
    elif args.mode=='replay':replay(args.split,args.sequence)
    else:evaluate(args.split,args.case)


if __name__=='__main__':main()
