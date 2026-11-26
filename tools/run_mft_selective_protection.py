#!/usr/bin/env python3
"""Reuse fitted risk coefficients; calibrate selective geometry protection on Train."""
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
from tools.fit_mft_candidate_risk import locations as parent_locations,threshold_values
from tools.run_mft_candidate_risk import require_fitted as parent_fitted,make_tracker as parent_tracker
from tools.mft_candidate_risk_association import policies as parent_policies,RiskAssociation
from tools.mft_selective_protection import SelectiveProtection,policies
from tools.mft_candidate_risk_data import qualify_Train,load_arrays,stream_video
from tools.run_mft_dev_candidate_graph_diagnosis import registered as diagnosis_registered
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity
from tools.run_mft_candidate_train_bank import atomic_json
from tools.task_utility_queue import Queue
from tools.fusion_memory_observation import LENGTHS
from tools.fusion_memory_streams import FusionMemoryStreamWriter
from tools.fusion_memory_evaluation import run_official_tracks,validate_mot
from tools.audit_mft_tracklet_r2 import bank_audit
from tools.run_mft_tracklet_permanence_r2 import registered as peer_registered
from tools.native_pyramid_stream_tracks import mot_line

SPEC=ROOT/'configs/perceptual_selective_protection_20261127.json'
PREFIX='SELECTIVE_PROTECTION'


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def locations():
    plan,protocol,work,parent=parent_locations()
    return plan,protocol,work,work/'selective_protection_20261127',parent


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    upstream=diagnosis_registered();plan,protocol,work,base,parent=locations();spec=read(SPEC)
    assert spec['complete_Dev_cases']==policies()
    proof=read(base/'CPU_regressions_r1.json');assert proof['actual_tool_exit_code']==0 and proof['tests_passed']==5
    sources=dict(upstream['source_hashes'])
    for n,h in proof['source_hashes'].items():assert sha256(ROOT/n)==h,n;sources[n]=h
    for n in ['tools/run_mft_selective_protection.py','tools/mft_selective_protection.py',str(SPEC.relative_to(ROOT))]:assert n in sources
    files=[base/'CPU_regressions_r1.json',base/'CPU_regressions_r1.stdout.txt',SPEC,
        ROOT/'configs/perceptual_candidate_recoverability_hypotheses_20261127.json',
        parent/'pipeline_registered.json',work/'dev_candidate_graph_diagnosis_20261127/complete.json']
    guard_storage(work)
    value=dict(status='REGISTERED_SELECTIVE_PROTECTION_BEFORE_CALIBRATION_OR_NEW_DEV',UTC=utc(),
        source_hashes=sources,artifact_hashes={str(p):sha256(p)for p in files},
        specification_sha256=sha256(SPEC),normal_cases=policies(),
        parent_controller_identity=read(parent/'controller.execution.json')['identity'],
        parent_new_Dev_exists_at_registration=(parent/'complete_Dev.json').exists(),
        parent_new_Test_exists_at_registration=(parent/'full_Test_results.json').exists(),
        current_parent_Test_accuracy_read_for_choice=False,new_optimizer_steps=0,new_NN_work=False,
        fixed_seed=0,historical_Test_previously_exposed=True,goal_met=False,supported_innovations=0)
    assert not(base/'pipeline_registered.json').exists();frozen_json(base/'pipeline_registered.json',value)
    print(json.dumps(dict(status=value['status'],sources=len(sources),cases=8)),flush=True)


def registered():
    plan,protocol,work,base,parent=locations();reg=read(base/'pipeline_registered.json')
    assert reg['specification_sha256']==sha256(SPEC)and reg['normal_cases']==policies()
    for n,h in reg['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in reg['artifact_hashes'].items():assert sha256(n)==h,n
    return reg


def parent_ready():
    reg=registered();plan,protocol,work,base,parent=locations();e=read(parent/'controller.execution.json')
    assert e['identity']==reg['parent_controller_identity']and e['actual_child_exit_code']==0 and e['source_unchanged']
    assert identity(e['identity']['PID'])is None
    preg,fit,models,thresholds=parent_fitted()
    return fit,models,thresholds


def register_calibration():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    fit,models,thresholds=parent_ready();plan,protocol,work,base,parent=locations()
    qualified=qualify_Train(work,protocol)
    value=dict(status='REGISTERED_FROZEN_PARENT_MODELS_ALL6_TRAIN_BEFORE_STRICT_CALIBRATION',UTC=utc(),
        qualified_Train=qualified,parent_model_sha256=fit['model_sha256'],parent_rescue_thresholds=thresholds,
        parent_fit_complete_sha256=sha256(parent/'fit_complete.json'),
        parent_Train_statistics_sha256=sha256(parent/'Train_feature_statistics.json'),
        pipeline_registration_sha256=sha256(base/'pipeline_registered.json'),quantile=.999,margin=.05,
        Train_frames=27251,Dev_rows=0,Test_rows=0,new_optimizer_steps=0,new_model_fits=0)
    assert not(base/'calibration_registered.json').exists();frozen_json(base/'calibration_registered.json',value)
    print(json.dumps(dict(status=value['status'],Train_frames=27251)),flush=True)


def calibrate():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    fit,models,thresholds=parent_ready();plan,protocol,work,base,parent=locations()
    reg=read(base/'calibration_registered.json')
    assert reg['pipeline_registration_sha256']==sha256(base/'pipeline_registered.json')
    assert reg['parent_model_sha256']==fit['model_sha256']
    qualified=qualify_Train(work,protocol);assert qualified==reg['qualified_Train']
    statistics=read(parent/'Train_feature_statistics.json')
    assert sha256(parent/'Train_feature_statistics.json')==reg['parent_Train_statistics_sha256']
    private=read(fit['model_file']);scores={k:[]for k in models};sizes={k:0 for k in models};coverage={}
    started=time.monotonic();assert not(base/'strict_Train_thresholds.json').exists()
    for sequence,q in qualified.items():
        counts=dict(known=0,positive=0,negative=0)
        def sink(batch):
            if batch.frame<=int(.75*q['frames']):return
            counts['known']+=len(batch.y);counts['positive']+=int(batch.y.sum());counts['negative']+=int((batch.y==0).sum())
            for k,m in models.items():
                wrong=m.probabilities(batch.x)[batch.y==0].copy();sizes[k]+=wrong.nbytes
                assert sizes[k]<=67108864,'Preserve failure; never drop calibration edges'
                scores[k].append(wrong)
        def progress(frame):
            guard_storage(work);atomic_json(base/'progress/calibration.json',dict(sequence=sequence,frame=frame,
                frames=q['frames'],seconds=time.monotonic()-started,new_optimizer_steps=0))
        reference=stream_video(sequence,q,threshold_values(work)['Train_unknown_speed_upper'],sink,progress=progress)
        assert reference==statistics['full_teacher_references'][sequence]
        assert counts==private['calibration'][sequence]['counts']
        coverage[sequence]=dict(frames=q['frames'],proposals=reference['proposal_rows'],late_counts=counts,
            teacher_MOT_sha256=reference['original_teacher_MOT_sha256'],bank_sha256=q['bank_sha256'],GT_sha256=q['GT_sha256'])
    strict={};rechecked={}
    for k in models:
        values=np.concatenate(scores[k]);assert len(values)>0 and np.isfinite(values).all()
        strict[k]=float(np.quantile(values,.999));rechecked[k]=float(np.quantile(values,.99))
        assert rechecked[k]==thresholds[k]and strict[k]>=thresholds[k]
    assert sha256(fit['model_file'])==fit['model_sha256'];registered()
    frozen_json(base/'strict_Train_thresholds.json',dict(status='COMPLETE_ALL6_TRAIN_STRICT_OVERRIDE_CALIBRATION',UTC=utc(),
        thresholds=strict,parent_rescue_thresholds=rechecked,coverage=coverage,frames=27251,sequences=6,
        source_registration_sha256=sha256(base/'calibration_registered.json'),parent_model_sha256=fit['model_sha256'],
        parent_model_bytes_unchanged=True,quantile=.999,margin=.05,new_optimizer_steps=0,new_model_fits=0,
        Dev_rows=0,Test_rows=0,candidate_pair_disk_dump=False,seconds=time.monotonic()-started))
    print(json.dumps(dict(status='COMPLETE_ALL6_TRAIN_STRICT_OVERRIDE_CALIBRATION',thresholds=strict,
        seconds=time.monotonic()-started)),flush=True)


def calibrated_models():
    fit,models,thresholds=parent_ready();plan,protocol,work,base,parent=locations()
    state=read(base/'state.json');job=state['phases'][PREFIX+'_CALIBRATE']['all6']
    assert job['status']=='COMPLETE'and job['exit_code']==0 and identity(job['pid'])is None
    value=read(base/'strict_Train_thresholds.json');assert value['frames']==27251 and value['sequences']==6
    assert value['parent_model_sha256']==fit['model_sha256']and value['Dev_rows']==value['Test_rows']==0
    return fit,models,thresholds,value['thresholds']


def make_tracker(case,models,thresholds,strict,work):
    if case in policies():return SelectiveProtection(case,models,thresholds,strict,threshold_values(work)['Train_unknown_speed_upper'])
    assert case in parent_policies()[:2]
    return parent_tracker(case,models,thresholds,work)


def prefix_rows(arrays,cases,models,thresholds,strict,work,sequence,count,missing_IR=False):
    trackers={c['id']:make_tracker(c,models,thresholds,strict,work)for c in cases};rows={c['id']:[]for c in cases}
    for i,f in enumerate(arrays['frame_ids'][:count]):
        lo,hi=map(int,arrays['offsets'][i:i+2])
        for c in cases:rows[c['id']].extend(trackers[c['id']].update(sequence,int(f),arrays['detections'][lo:hi],
            arrays['rgb'][lo:hi],None if missing_IR else arrays['ir'][lo:hi]))
    return {k:''.join(map(mot_line,v))for k,v in rows.items()}


def engineer():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    fit,models,thresholds,strict=calibrated_models();plan,protocol,work,base,parent=locations()
    qualified=qualify_Train(work,protocol);proofs={}
    for sequence,q in qualified.items():
        arrays=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);count=min(64,q['frames']);cut=min(32,count)
        full=prefix_rows(arrays,policies(),models,thresholds,strict,work,sequence,count)
        prefix=prefix_rows(arrays,policies(),models,thresholds,strict,work,sequence,cut)
        for c in policies():assert full[c['id']].startswith(prefix[c['id']])
        offset=int(arrays['offsets'][cut]);future=arrays['detections'][offset:].copy()
        arrays['detections'][offset:,:4]+=777
        for cue in ['rgb','ir']:arrays[cue][offset:]*=-1
        assert prefix_rows(arrays,policies(),models,thresholds,strict,work,sequence,cut)==prefix
        arrays['detections'][offset:]=future
        for cue in ['rgb','ir']:arrays[cue][offset:]*=-1
        for c in policies()[:4]:
            source=next(p for p in parent_policies()if p['id']==c['source_case'])
            tracker=RiskAssociation(source,models,thresholds,threshold_values(work)['Train_unknown_speed_upper']);reference=[]
            for i,f in enumerate(arrays['frame_ids'][:count]):
                lo,hi=map(int,arrays['offsets'][i:i+2]);reference.extend(tracker.update(sequence,int(f),
                    arrays['detections'][lo:hi],arrays['rgb'][lo:hi],arrays['ir'][lo:hi]))
            assert ''.join(map(mot_line,reference))==full[c['id']]
        missing=prefix_rows(arrays,policies()[4:6],models,thresholds,strict,work,sequence,count,True)
        assert missing['SP_RGB_selective']==missing['SP_paired_selective']
        proofs[sequence]=dict(frames=count,all4_static_controls_MOT_exact=True,
            prefix_and_future_suffix_corruption_exact=True,missing_IR_MOT_exact_RGB_fallback=True,
            bank_sha256=q['bank_sha256']);guard_storage(work)
    assert sha256(fit['model_file'])==fit['model_sha256'];registered()
    frozen_json(base/'real_Train_engineering.json',dict(status='COMPLETE_ALL6_TRAIN_SELECTIVE_PROTECTION_ENGINEERING',UTC=utc(),
        sequences=6,frames=sum(v['frames']for v in proofs.values()),rows=proofs,parent_model_bytes_unchanged=True,
        parent_model_sha256=fit['model_sha256'],strict_threshold_sha256=sha256(base/'strict_Train_thresholds.json'),
        new_optimizer_steps=0,new_GPU_work=False,new_accuracy=False))
    print(json.dumps(dict(status='COMPLETE_ALL6_TRAIN_SELECTIVE_PROTECTION_ENGINEERING',sequences=6)),flush=True)


def register_evaluation():
    fit,models,thresholds,strict=calibrated_models();plan,protocol,work,base,parent=locations()
    state=read(base/'state.json');j=state['phases'][PREFIX+'_ENGINEERING']['all6']
    assert j['status']=='COMPLETE'and j['exit_code']==0 and identity(j['pid'])is None
    eng=read(base/'real_Train_engineering.json');assert eng['sequences']==6 and eng['parent_model_bytes_unchanged']
    dev=read(parent/'complete_Dev.json');assert dev['Test_read_for_selection']is False
    best=min(dev['rows'],key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),r['case']['id']))
    assert best['case']==dev['chosen_case']
    value=dict(status='REGISTERED_SELECTIVE_FULL_DEV_AFTER_TRAIN_CALIBRATION_AND_ENGINEERING',UTC=utc(),
        strict_thresholds_sha256=sha256(base/'strict_Train_thresholds.json'),
        engineering_sha256=sha256(base/'real_Train_engineering.json'),parent_model_sha256=fit['model_sha256'],
        parent_complete_Dev_sha256=sha256(parent/'complete_Dev.json'),parent_best_Dev_case=best['case'],
        parent_best_Dev_metrics={k:best['combined47'][k]for k in ['HOTA','AssA','IDF1']},
        normal_cases=policies(),current_parent_Test_score_used=False,
        pipeline_registration_sha256=sha256(base/'pipeline_registered.json'))
    assert not(base/'evaluation_registered.json').exists();frozen_json(base/'evaluation_registered.json',value)
    print(json.dumps(dict(status=value['status'],cases=8,Dev_gate_HOTA=best['combined47']['HOTA'])),flush=True)


def require_evaluation():
    reg=registered();fit,models,thresholds,strict=calibrated_models();plan,protocol,work,base,parent=locations()
    value=read(base/'evaluation_registered.json')
    assert value['pipeline_registration_sha256']==sha256(base/'pipeline_registered.json')
    assert value['strict_thresholds_sha256']==sha256(base/'strict_Train_thresholds.json')
    assert value['engineering_sha256']==sha256(base/'real_Train_engineering.json')
    assert value['parent_complete_Dev_sha256']==sha256(parent/'complete_Dev.json')
    assert value['parent_model_sha256']==fit['model_sha256']and value['normal_cases']==policies()
    return reg,fit,models,thresholds,strict,value


def split_cases(split):
    plan,protocol,work,base,parent=locations()
    if split=='dev':return policies()
    assert split=='test';freeze=read(base/'Test_selection_frozen.json')
    assert freeze['Test_read_for_selection']is False
    assert freeze['complete_Dev_sha256']==sha256(base/'complete_Dev.json')
    return freeze['cases']


def replay(split,sequence):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''and sequence in LENGTHS[split]
    reg,fit,models,thresholds,strict,ev=require_evaluation();plan,protocol,work,base,parent=locations()
    cases=split_cases(split);peer=work/'tracklet_permanence_20261127_r2'
    bank=bank_audit(peer,split,sequence,peer_registered());meta=read(peer/'cache'/split/(sequence+'.json'))
    with np.load(meta['path'],allow_pickle=False)as data:arrays={k:data[k].copy()for k in data.files}
    count=min(32,LENGTHS[split][sequence]);expected=prefix_rows(arrays,cases,models,thresholds,strict,work,sequence,count)
    offset=int(arrays['offsets'][count]);future=arrays['detections'][offset:].copy()
    arrays['detections'][offset:,:4]+=777
    for cue in ['rgb','ir']:arrays[cue][offset:]*=-1
    assert prefix_rows(arrays,cases,models,thresholds,strict,work,sequence,count)==expected
    arrays['detections'][offset:]=future
    for cue in ['rgb','ir']:arrays[cue][offset:]*=-1
    missing=prefix_rows(arrays,policies()[4:6],models,thresholds,strict,work,sequence,count,True)
    assert missing['SP_RGB_selective']==missing['SP_paired_selective']
    writer=FusionMemoryStreamWriter(base/'streams'/split,sequence,[c['id']for c in cases])
    trackers={c['id']:make_tracker(c,models,thresholds,strict,work)for c in cases};started=time.monotonic()
    try:
        for i,f in enumerate(arrays['frame_ids']):
            lo,hi=map(int,arrays['offsets'][i:i+2]);d=arrays['detections'][lo:hi]
            rows={c['id']:trackers[c['id']].update(sequence,int(f),d,arrays['rgb'][lo:hi],arrays['ir'][lo:hi])for c in cases}
            writer.append(int(f),rows,{c['id']:d for c in cases})
            if int(f)%100==0 or int(f)==LENGTHS[split][sequence]:
                guard_storage(work);atomic_json(base/'progress'/split/(sequence+'.json'),
                    dict(frame=int(f),frames=LENGTHS[split][sequence],seconds=time.monotonic()-started,CPU_only=True))
        streams=writer.close(LENGTHS[split][sequence])
    finally:writer.abort()
    if split=='dev':
        old=read(parent/'streams/dev/sequence_manifests'/(sequence+'.json'))
        for c in policies()[:4]:assert streams[c['id']]['track_sha256']==old['streams'][c['source_case']]['track_sha256']
    else:
        old=read(work/'pose_continuity_20261127/streams/test/sequence_manifests'/(sequence+'.json'))
        for own,source in [('CR_native_zero','PC_native_zero'),('CR_pose_late_control','PC_shape_anchor_late')]:
            assert streams[own]['track_sha256']==old['streams'][source]['track_sha256']
    assert all(s['detection_packet_sha256']==bank['packet_sha256']for s in streams.values())
    assert sha256(fit['model_file'])==fit['model_sha256'];registered()
    frozen_json(base/'streams'/split/'sequence_manifests'/(sequence+'.json'),dict(
        status='COMPLETE_CAUSAL_SELECTIVE_PROTECTION_FULL_SEQUENCE',UTC=utc(),split=split,sequence=sequence,
        frames=LENGTHS[split][sequence],cases=cases,streams=streams,bank=bank,
        source_versions=reg['source_hashes'],parent_model_sha256=fit['model_sha256'],
        strict_thresholds_sha256=sha256(base/'strict_Train_thresholds.json'),
        static_controls_or_Test_references_MOT_exact=True,future_suffix_prefix_MOT_exact=True,
        missing_IR_filtered_prefix_MOT_exact_RGB_fallback=True,
        statistics={k:t.statistics for k,t in trackers.items()},only_current_frame_inputs=True,
        GT_read=False,previous_outputs_rewritten=False,model_bytes_unchanged=True,
        compact_feature_cache_read=True,writer_local_false_cache_flag_scope='Writer creates only MOT; workflow reads compact NPZ',
        CPU_replay_seconds=time.monotonic()-started,encoder_detector_image_cost_excluded=True))


def evaluate(split,case_id):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg,fit,models,thresholds,strict,ev=require_evaluation();plan,protocol,work,base,parent=locations()
    case=next(c for c in split_cases(split)if c['id']==case_id);coverage={}
    for sequence,frames in LENGTHS[split].items():
        value=read(base/'streams'/split/'sequence_manifests'/(sequence+'.json'))
        assert value['frames']==frames and value['source_versions']==reg['source_hashes']
        assert value['parent_model_sha256']==fit['model_sha256']and value['model_bytes_unchanged']
        assert value['static_controls_or_Test_references_MOT_exact']and value['future_suffix_prefix_MOT_exact']
        s=value['streams'][case_id];assert sha256(s['track_path'])==s['track_sha256']
        validate_mot(s['track_path'],frames,s['track_rows']);coverage[sequence]=s
    output=base/'evaluation'/split/case_id;assert not output.exists();output.mkdir(parents=True)
    values=run_official_tracks(base/'streams'/split/case_id/'track_results',output,plan,split);registered()
    frozen_json(output/'metrics.json',dict(status='COMPLETE_OFFICIAL_SELECTIVE_PROTECTION_ALL_METRICS',
        case=case,frames=sum(LENGTHS[split].values()),coverage=coverage,source_versions=reg['source_hashes'],
        parent_model_sha256=fit['model_sha256'],strict_thresholds_sha256=sha256(base/'strict_Train_thresholds.json'),
        **values,goal_met=False,supported_innovations=0))


def run_phase(queue,split):
    plan,protocol,work,base,parent=locations();cases=split_cases(split)
    cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))]
    queue.jobs(PREFIX+'_'+split.upper()+'_REPLAY',[dict(id=s,command=cmd+['replay','--split',split,'--sequence',s])
        for s in LENGTHS[split]],False,
        lambda j:read(base/'streams'/split/'sequence_manifests'/(j['id']+'.json'))['static_controls_or_Test_references_MOT_exact'])
    queue.jobs(PREFIX+'_'+split.upper()+'_OFFICIAL',[dict(id=c['id'],command=cmd+['evaluate','--split',split,'--case',c['id']])
        for c in cases],False,lambda j:read(base/'evaluation'/split/j['id']/'metrics.json')['frames']==sum(LENGTHS[split].values()))
    return [read(base/'evaluation'/split/c['id']/'metrics.json')for c in cases]


def wait_parent(queue):
    reg=registered();plan,protocol,work,base,parent=locations();expected=reg['parent_controller_identity']
    queue.state.update(status='WAITING_UNCHANGED_PARENT_RISK_COMPLETE',parent_identity=expected);queue.save()
    while True:
        registered();guard_storage(work);value=read(parent/'controller.execution.json');assert value['identity']==expected
        if value['actual_child_exit_code']is not None:
            assert value['actual_child_exit_code']==0 and value['source_unchanged']and identity(expected['PID'])is None
            parent_ready();return
        live=identity(expected['PID'])
        if live is not None:assert all(live[k]==expected[k]for k in ['PID','start_ticks','raw_argv_sha256'])
        queue.state['last_parent_wait_UTC']=utc();queue.save();time.sleep(15)


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();plan,protocol,work,base,parent=locations()
    with(base/'state.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(base/'state.json').exists()
        state=dict(status='STARTING_SELECTIVE_PROTECTION',controller_pid=os.getpid(),phases={},
            data_root=plan['data_root'],new_GPU_work=False,new_optimizer_steps=0,fixed_seed=0,goal_met=False,supported_innovations=0)
        queue=Queue(work,base,state,plan['data_root'],registered);queue.save()
        cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))]
        try:
            wait_parent(queue)
            queue.jobs(PREFIX+'_CAL_REGISTER',[dict(id='all6',command=cmd+['cal-register'])],False,
                lambda j:read(base/'calibration_registered.json')['Train_frames']==27251)
            queue.jobs(PREFIX+'_CALIBRATE',[dict(id='all6',command=cmd+['calibrate'])],False,
                lambda j:read(base/'strict_Train_thresholds.json')['frames']==27251)
            queue.jobs(PREFIX+'_ENGINEERING',[dict(id='all6',command=cmd+['engineer'])],False,
                lambda j:read(base/'real_Train_engineering.json')['sequences']==6)
            queue.jobs(PREFIX+'_EVAL_REGISTER',[dict(id='all6',command=cmd+['eval-register'])],False,
                lambda j:len(read(base/'evaluation_registered.json')['normal_cases'])==8)
            values=run_phase(queue,'dev');ev=read(base/'evaluation_registered.json')
            parent_dev=read(parent/'complete_Dev.json')
            for r in values[:4]:
                original=next(p for p in parent_dev['rows']if p['case']['id']==r['case']['source_case'])
                assert r['combined47']==original['combined47']and r['per_sequence47']==original['per_sequence47']
            chosen=min(values,key=lambda r:(*(-r['combined47'][k]for k in ['HOTA','AssA','IDF1']),r['case']['id']))
            gain=chosen['combined47']['HOTA']-ev['parent_best_Dev_metrics']['HOTA']
            frozen_json(base/'complete_Dev.json',dict(status='COMPLETE_ALL8_SELECTIVE_PROTECTION_FULL_DEV',UTC=utc(),
                rows=values,chosen_case=chosen['case'],gain_over_parent_best_Dev_pp=gain,
                parent_best_Dev_case=ev['parent_best_Dev_case'],parent_best_Dev_metrics=ev['parent_best_Dev_metrics'],
                Test_read_for_selection=False,current_parent_Test_score_used=False,
                all4_static_controls_all47_and_MOT_exact=True,goal_met=False,supported_innovations=0))
            if gain<=1e-8:
                state.update(status='COMPLETE_SELECTIVE_PROTECTION_NO_DEV_GAIN_NO_NEW_TEST',Dev_gain_pp=gain);queue.save();return
            frozen_json(base/'Test_selection_frozen.json',dict(status='FROZEN_SELECTIVE_PROTECTION_FROM_COMPLETE_DEV',UTC=utc(),
                cases=[*parent_policies()[:2],chosen['case']],chosen_case=chosen['case'],
                complete_Dev_sha256=sha256(base/'complete_Dev.json'),Test_read_for_selection=False,
                current_parent_Test_score_used=False,historical_Test_previously_exposed=True))
            test=run_phase(queue,'test');selected=next(r for r in test if r['case']==chosen['case'])
            from tools.export_perceptual_metrics47 import official_schema
            from tools.fusion_memory_evaluation import official_csv47
            csv=work/'native_pyramid_evaluation/test/NP_source_canonical_RGB/trackeval_output/pedestrian_detailed.csv'
            assert sha256(csv)=='c63ae33bbb01f14009da3dd3f834b017907df02958fe5200c68f20f38dc6fc6a'
            summary=read(csv.parent.parent/'metrics.json');schema,_=official_schema(plan['evaluation']['TrackEval_root'])
            original=official_csv47(csv,schema,'test',summary['trackeval'])
            assert abs(original['combined47']['HOTA']-29.528970687393972)<1e-10
            delta=selected['combined47']['HOTA']-original['combined47']['HOTA']
            frozen_json(base/'full_Test_results.json',dict(status='COMPLETE_DEV_FROZEN_SELECTIVE_PROTECTION_FULL7_TEST',UTC=utc(),
                rows=test,original_RGB_baseline=original,selected_case=chosen['case'],HOTA_gain_over_original_RGB_pp=delta,
                target_plus5_met=delta>=5,target_plus10_met=delta>=10,Test_used_for_selection=False,
                historical_Test_previously_exposed=True,goal_met=False,supported_innovations=0))
            state.update(status='COMPLETE_SELECTIVE_PROTECTION_FULL_TEST_REQUIRES_SCIENTIFIC_ASSESSMENT',
                HOTA_gain_over_original_RGB_pp=delta);queue.save()
        except Exception as error:
            state.update(status='FAILED_PRESERVE_SELECTIVE_PROTECTION_ATTEMPT',error=repr(error));queue.save();raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['register','run','cal-register','calibrate','engineer','eval-register','replay','evaluate'])
    p.add_argument('--split',choices=['dev','test']);p.add_argument('--sequence');p.add_argument('--case');args=p.parse_args()
    if args.mode=='register':register()
    elif args.mode=='run':run()
    elif args.mode=='cal-register':register_calibration()
    elif args.mode=='calibrate':calibrate()
    elif args.mode=='engineer':engineer()
    elif args.mode=='eval-register':register_evaluation()
    elif args.mode=='replay':replay(args.split,args.sequence)
    else:evaluate(args.split,args.case)


if __name__=='__main__':main()
