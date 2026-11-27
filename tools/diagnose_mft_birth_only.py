#!/usr/bin/env python3
"""Engineer a birth-only tracker port and diagnose all6Train opportunities."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.mft_birth_only import (BirthOnlyQuality,BirthOnlyTeacher,birth_mask,
    policies,ORIGINAL_BYTE_TRACKER_SHA256)
from tools.mft_candidate_risk_teacher import CandidateTeacher
from tools.mft_proposal_quality import ProposalConfirmation,current_features
from tools.mft_proposal_quality_evaluation import SharedConfirmation,make_tracker,matched
from tools.run_mft_proposal_quality import registered as quality_registered,fitted,locations as quality_locations
from tools.run_mft_archive_reasoning import fitted as archive_fitted
from tools.fit_mft_candidate_risk import threshold_values
from tools.mft_candidate_risk_data import load_arrays,load_GT
from tools.mft_candidate_risk_labels import current_detection_labels
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.run_mft_candidate_train_bank import atomic_json
from tools.run_mft_native_memory_priority_v3 import identity
from tools.native_pyramid_stream_tracks import mot_line
from tools.run_pseudo_ir_comparison import sha256
from tools.fit_mft_proposal_quality import utc
from yolox.tracker import matching

SPEC=ROOT/'configs/perceptual_birth_only_20261127.json'
SOURCES=['tools/mft_birth_only.py','tools/diagnose_mft_birth_only.py',
    'tests/test_mft_birth_only.py',str(SPEC.relative_to(ROOT))]


def locations():
    plan,protocol,work,peer,train=quality_locations()
    return plan,protocol,work,work/'birth_only_20261127',train


def sources():
    reg=quality_registered();result=dict(reg['source_hashes'])
    for n in SOURCES:result[n]=sha256(ROOT/n)
    assert sha256(ROOT/'yolox/tracker/byte_tracker.py')==ORIGINAL_BYTE_TRACKER_SHA256
    return result


def state_equal(a,b):
    assert a.core.frame_id==b.core.frame_id
    for name in ['tracked_stracks','lost_stracks','removed_stracks']:
        x,y=getattr(a.core,name),getattr(b.core,name)
        assert len(x)==len(y)
        for p,q in zip(x,y):
            for k in ['track_id','state','is_activated','frame_id','start_frame','score','accepted','memory_frame','measurement_frame','proposal_index']:
                assert getattr(p,k)==getattr(q,k),(name,k)
            for k in ['mean','covariance','rgb','ir','current_measurement','last_box']:
                u,v=getattr(p,k),getattr(q,k)
                if u is None or v is None:assert u is v
                else:np.testing.assert_array_equal(u,v)


def engineer():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    source=sources();reg,fit,views,heads,thresholds=fitted()
    plan,protocol,work,base,train=locations();speed=threshold_values(work)['Train_unknown_speed_upper']
    _,_,old_models,old_thresholds=archive_fitted();rows={};cases=policies()
    missing=[dict(c,id='MISSING_'+c['id'],intervention='missing')for c in cases if c['kind']=='paired']
    def replay(sequence,a,count,future=False):
        shared=SharedConfirmation();trackers={c['id']:BirthOnlyQuality(c,views,heads,thresholds,speed,old_models,old_thresholds,shared)for c in cases+missing}
        plain=CandidateTeacher(speed)
        previous=make_tracker(dict(id='PQ_archive_control',kind='reference_archive'),views,heads,thresholds,speed,old_models,old_thresholds)
        values={k:[]for k in trackers}
        for i in range(count):
            lo,hi=map(int,a['offsets'][i:i+2]);d=a['detections'][lo:hi].copy();r=a['rgb'][lo:hi].copy();ir=a['ir'][lo:hi].copy()
            if future and i>=32:d[:,:4]+=777;r*=-1;ir*=-1
            originals=[x.copy()for x in [d,r,ir]]
            for k,t in trackers.items():values[k].extend(t.update(sequence,i+1,d,r,ir))
            raw=plain.update(sequence,i+1,d,r,ir);old=previous.update(sequence,i+1,d,r,ir)
            assert ''.join(map(mot_line,raw))==''.join(map(mot_line,[x for x in values['BO_native_zero']if x[0]==i+1]))
            assert ''.join(map(mot_line,old))==''.join(map(mot_line,[x for x in values['BO_archive_zero']if x[0]==i+1]))
            state_equal(plain,trackers['BO_native_zero'].teacher)
            for x,y in zip([d,r,ir],originals):np.testing.assert_array_equal(x,y)
        return {k:{'full':''.join(map(mot_line,v)),'prefix32':''.join(mot_line(x)for x in v if x[0]<=32)}for k,v in values.items()}
    for sequence,q in reg['qualified_Train'].items():
        a=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);normal=replay(sequence,a,64);future=replay(sequence,a,64,True)
        assert all(normal[k]['prefix32']==future[k]['prefix32']for k in normal)
        for c in cases:
            if c['kind']=='paired':
                rgb=next(x['id']for x in cases if x['kind']=='RGB'and x['quantile']==c['quantile']and x['confirmation']==c['confirmation'])
                assert normal['MISSING_'+c['id']]['full']==normal[rgb]['full']
        rows[sequence]=dict(frames=64,zero_gate_native_full_state_and_MOT_exact=True,zero_gate_archive_shape_MOT_exact=True,
            all20_cases_future_suffix_prefix32_MOT_exact=True,all4_qualityIR_missing_controls_exact_RGB=True,
            input_boxes_scores_descriptors_unchanged=True,bank_sha256=q['bank_sha256'])
        guard_storage(work);print(json.dumps(dict(sequence=sequence,status='ENGINEERING_PASSED')),flush=True)
    assert sources()==source and sha256(fit['model_file'])==fit['model_sha256']
    frozen_json(base/'real_Train_engineering.json',dict(status='COMPLETE_ALL6_BIRTH_ONLY_REAL_TRAIN_ENGINEERING',UTC=utc(),
        rows=rows,sequences=6,source_hashes=source,model_sha256=fit['model_sha256'],GT_labels_used=False,
        new_accuracy=False,goal_met=False,supported_innovations=0))


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    source=sources();plan,protocol,work,base,train=locations();reg,fit,views,heads,thresholds=fitted()
    e=read(base/'engineering_r1.execution.json');p=read(base/'real_Train_engineering.json');u=read(base/'CPU_regressions.json')
    assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(e['identity']['PID'])is None
    assert p['source_hashes']==source and p['sequences']==6
    assert u['actual_tool_exit_code']==0 and u['tests_passed']==3
    for n,h in u['source_hashes'].items():assert sha256(ROOT/n)==h
    paths=[SPEC,base/'CPU_regressions.json',base/'engineering_r1.execution.json',base/'real_Train_engineering.json']
    frozen_json(base/'registered.json',dict(status='REGISTERED_ALL6_TRAIN_ZERO_GATE_BIRTH_OPPORTUNITY_DIAGNOSIS',UTC=utc(),
        source_hashes=source,artifact_hashes={str(p):sha256(p)for p in paths},qualified_Train=reg['qualified_Train'],
        model_sha256=fit['model_sha256'],cases=policies(),Train_frames=27251,seed=0,
        Dev_rows=0,Test_rows=0,new_GPU_work=False,new_fits=False,new_disk_feature_dump=False,goal_met=False,supported_innovations=0))


def registered():
    plan,protocol,work,base,train=locations();reg=read(base/'registered.json')
    assert sources()==reg['source_hashes']
    for n,h in reg['artifact_hashes'].items():assert sha256(n)==h
    return reg


def counter():
    return dict(opportunities=0,known_positive=0,known_background=0,unknown=0,
        known_positive_uncovered_by_zero_output_local_IoU05=0,near_existing_measured_proposal_IoU05=0)


def diagnose():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();_,fit,views,heads,thresholds=fitted();plan,protocol,work,base,train=locations()
    speed=threshold_values(work)['Train_unknown_speed_upper'];started=time.monotonic();coverage={}
    cases=[c for c in policies()if not c['kind'].startswith('reference_')]
    old=read(work/'proposal_confidence_diagnosis_20261127/complete_Train.json')
    for sequence,q in reg['qualified_Train'].items():
        a=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256'])
        plain=CandidateTeacher(speed);port=BirthOnlyTeacher(speed);confirmation=ProposalConfirmation()
        counts={phase:{c['id']:counter()for c in cases}for phase in ['early','late']}
        all_counts={phase:counter()for phase in ['early','late']};original=hashlib.sha256();mot=hashlib.sha256();label_packet=hashlib.sha256();output_rows=0
        for i,fid in enumerate(a['frame_ids']):
            frame=int(fid);lo,hi=map(int,a['offsets'][i:i+2]);left,right=map(int,bounds[i:i+2]);d=a['detections'][lo:hi];r=a['rgb'][lo:hi];ir=a['ir'][lo:hi]
            before=d.copy();streak=confirmation.update(sequence,frame,d)
            x=current_features(d,r,ir,views);pred={k:head.probabilities(x)for k,head in heads.items()}
            masks={}
            for c in cases:
                threshold=0. if c['kind']=='unconditional'else thresholds[c['kind']][c['quantile']]['confirmed'if c['confirmation']else'all']
                p=np.ones(len(d))if c['kind']=='unconditional'else pred[c['kind']]
                masks[c['id']]=(d[:,4]>.6)&(d[:,4]<.7)&(p>threshold)&(streak>=c['confirmation'])
            # Both inference paths finish before any current GT label is made.
            raw=plain.update(sequence,frame,d,r,ir);rows=port.update(sequence,frame,d,r,ir)
            text=''.join(map(mot_line,raw));assert text==''.join(map(mot_line,rows));mot.update(text.encode());output_rows+=len(rows)
            np.testing.assert_array_equal(d,before)
            if frame%100==0 or frame==q['frames']:state_equal(plain,port)
            labels=current_detection_labels(d,gt[left:right],frame=frame,split='train')
            label_packet.update(np.asarray([frame,len(d)],np.int64).tobytes()+d.tobytes()+labels.tobytes())
            original.update(np.asarray([frame,len(d)],dtype='<i8').tobytes()+np.asarray(d,dtype='<f4').tobytes())
            phase='early'if frame<=int(.75*q['frames'])else'late'
            candidates=[e['proposal_index']for e in port.core.last_birth_events if not e['original_allowed']]
            assigned=list(plain.current_proposal_assignments(frame).values())
            measured=d[assigned,:4]if assigned else np.empty((0,4))
            out=np.asarray([np.r_[v[1][:2],v[1][:2]+v[1][2:]]for v in rows],np.float32).reshape(-1,4)
            current_gt=gt[left:right]
            for j in candidates:
                target=int(labels[j]);uncovered=False
                if target>0:
                    found=current_gt[(current_gt[:,1]==target)&(current_gt[:,6]>0)&(current_gt[:,7]==1)]
                    assert len(found)==1
                    box=found[:,2:6].copy();box[:,2:]+=box[:,:2]
                    uncovered=not len(out)or matching.ious(box,out).max()<.5
                nearby=bool(len(measured)and matching.ious(d[j:j+1,:4],measured).max()>=.5)
                def add(c):
                    c['opportunities']+=1;c['known_positive'if target>0 else'known_background'if target==0 else'unknown']+=1
                    c['known_positive_uncovered_by_zero_output_local_IoU05']+=int(uncovered)
                    c['near_existing_measured_proposal_IoU05']+=int(nearby)
                add(all_counts[phase])
                for c in cases:
                    if masks[c['id']][j]:add(counts[phase][c['id']])
            if frame%1000==0 or frame==q['frames']:
                guard_storage(work);atomic_json(base/'progress.json',dict(status='ALL_TRAIN_BIRTH_OPPORTUNITY_DIAGNOSIS',sequence=sequence,
                    frame=frame,frames=q['frames'],completed_sequences=len(coverage),seconds=time.monotonic()-started))
        prior=old['coverage'][sequence]
        assert label_packet.hexdigest()==prior['current_Train_label_packet_sha256']and output_rows==prior['ordinary_output_rows']
        assert port.statistics['extra_qualified_births']==0
        coverage[sequence]=dict(frames=q['frames'],proposals=len(a['detections']),ordinary_output_rows=output_rows,
            all_original_birth_opportunities=all_counts,by_case=counts,original_detector_packet_sha256=original.hexdigest(),
            zero_gate_MOT_sha256=mot.hexdigest(),current_Train_label_packet_sha256=label_packet.hexdigest(),
            original_teacher_and_birth_only_zero_gate_all_MOT_exact=True,zero_gate_full_state_every100_frames_and_end_exact=True,
            bank_sha256=q['bank_sha256'],GT_sha256=q['GT_sha256'])
        print(json.dumps(dict(sequence=sequence,all_opportunities=all_counts)),flush=True)
    assert registered()==reg and sha256(fit['model_file'])==fit['model_sha256']
    frozen_json(base/'complete_Train_diagnosis.json',dict(status='COMPLETE_ALL27251_TRAIN_BIRTH_ONLY_OPPORTUNITY_DIAGNOSIS',UTC=utc(),
        coverage=coverage,Train_frames=sum(v['frames']for v in coverage.values()),proposals=sum(v['proposals']for v in coverage.values()),
        model_sha256=fit['model_sha256'],source_hashes=reg['source_hashes'],registration_sha256=sha256(base/'registered.json'),
        original_scores_boxes_and_zero_gate_outputs_unchanged=True,GT_labels_only_after_inference=True,
        unknown_not_background=True,local_opportunities_not_counterfactual_HOTA_or_recoverability_guarantee=True,
        new_fits=False,new_GPU_work=False,Dev_rows=0,Test_rows=0,new_feature_disk_dump=False,
        seconds=time.monotonic()-started,goal_met=False,supported_innovations=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['engineer','register','diagnose']);args=p.parse_args()
    {'engineer':engineer,'register':register,'diagnose':diagnose}[args.mode]()
