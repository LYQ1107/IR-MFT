#!/usr/bin/env python3
"""Full-Train potential of output-only causal short-gap predictions.

No virtual box is fed to the tracker or emitted as a new formal result. The
current Kalman prediction is a GT-free candidate; only a separate Train label
join measures whether it could cover a currently missed annotated object.
"""
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
from yolox.tracker import matching
from yolox.tracker.basetrack import TrackState
from tools.fit_mft_candidate_risk import locations as parent_locations,threshold_values
from tools.mft_candidate_risk_data import qualify_Train,load_arrays,load_GT
from tools.mft_candidate_risk_teacher import CandidateTeacher,observations
from tools.mft_candidate_risk_labels import current_detection_labels
from tools.mft_identity_ledger_labels import OriginLabels
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_stream_tracks import mot_line
from tools.native_pyramid_training_loop import guard_storage
from tools.run_mft_candidate_train_bank import atomic_json
from tools.run_pseudo_ir_comparison import sha256

SPEC=ROOT/'configs/perceptual_causal_gap_completion_diagnosis_20261127.json'
FEATURES=('log_gap','log_track_age','log_reliable_count','last_score','template_quality',
    'vx_body','vy_body','speed_body','dx_from_last_measurement','dy_from_last_measurement',
    'log_center_covariance_body','nearby_proposal_count','nearby_max_score',
    'max_current_proposal_IoU','max_visible_output_IoU','max_nearby_RGB_cosine','max_nearby_IR_cosine',
    'IR_available','cue_disagreement','log_template_age')
REASONS=('positive_uncovered','negative_duplicate_visible','negative_origin_absent','negative_bad_geometry',
    'unknown_origin','unknown_ignored','unknown_IoU_band')


def locations():
    plan,protocol,work,parent=parent_locations();return plan,protocol,work,work/'causal_gap_completion_20261127',parent


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan,protocol,work,base,parent=locations();q=qualify_Train(work,protocol)
    sources=dict(read(work/'identity_ledger_20261127/fit_registered.json')['source_hashes'])
    for n in ['tools/diagnose_mft_causal_gap_completion.py',str(SPEC.relative_to(ROOT))]:sources[n]=sha256(ROOT/n)
    for n,h in sources.items():assert sha256(ROOT/n)==h,n
    v=dict(status='REGISTERED_ALL6_TRAIN_CAUSAL_GAP_COMPLETION_DIAGNOSIS_BEFORE_COLLECTION',source_hashes=sources,
        qualified_Train=q,specification_sha256=sha256(SPEC),Train_frames=27251,Dev_rows=0,Test_rows=0,seed=0,
        all_actual_proposals_delivered=True,teacher_physical_and_output_MOT_unchanged=True,
        GT_free_virtual_candidate_generation=True,Train_only_labels=True,new_tracker=False,new_models=False,
        pair_or_feature_disk_dump=False,new_GPU_work=False,historical_Test_previously_exposed=True,
        goal_met=False,supported_innovations=0)
    assert not(base/'registered.json').exists();frozen_json(base/'registered.json',v)
    print(json.dumps(dict(status=v['status'],sources=len(sources))),flush=True)


def registered():
    plan,protocol,work,base,parent=locations();r=read(base/'registered.json')
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    assert r['specification_sha256']==sha256(SPEC)
    return r


def band(gap):return '1'if gap==1 else'2_3'if gap<=3 else'4_5'if gap<=5 else'6_10'if gap<=10 else'11_25'


def candidate(track,frame,d,obs,visible):
    gap=frame-int(track.frame_id);assert 1<=gap<=25
    box=track.tlbr.copy();size=np.maximum(box[2:]-box[:2],1.);diagonal=float(np.linalg.norm(size))
    center=(box[:2]+box[2:])/2;last=(track.last_box[:2]+track.last_box[2:])/2
    current_center=(d[:,:2]+d[:,2:4])/2;distance=np.linalg.norm(current_center-center,axis=1)/diagonal
    near=np.flatnonzero(distance<=2);velocity=np.asarray(track.mean[4:6],np.float64)/diagonal
    rgb=max((float(track.rgb@obs[j]['rgb'])for j in near),default=0.)
    available=track.ir is not None and len(near)>0
    ir=max((float(track.ir@obs[j]['ir'])for j in near),default=0.)if available else 0.
    iou=matching.ious(box[None],d[:,:4]);other=matching.ious(box[None],visible)
    x=np.asarray([np.log1p(gap),np.log1p(frame-track.start_frame),np.log1p(track.accepted),track.score,track.memory_quality,
        *velocity,float(np.linalg.norm(velocity)),*((center-last)/diagonal),
        np.log1p(max(float(np.trace(track.covariance[:2,:2])),0)/diagonal**2),len(near),
        float(d[near,4].max())if len(near)else 0.,float(iou.max())if iou.size else 0.,
        float(other.max())if other.size else 0.,rgb,ir,float(available),abs(rgb-ir)if available else 0.,
        np.log1p(frame-track.memory_frame)],np.float32)
    assert x.shape==(len(FEATURES),)and np.isfinite(x).all()
    return box,x


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();plan,protocol,work,base,parent=locations();started=time.monotonic();coverage={}
    reference=read(parent/'Train_feature_statistics.json')['full_teacher_references']
    for sequence,q in reg['qualified_Train'].items():
        arrays=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256'])
        teacher=CandidateTeacher(threshold_values(work)['Train_unknown_speed_upper']);teacher.reset(sequence)
        origins=OriginLabels();seed_count={};digest=hashlib.sha256();features=hashlib.sha256()
        buckets={p:{b:{r:0 for r in REASONS}for b in ['1','2_3','4_5','6_10','11_25']}for p in ['early','late']}
        stats=dict(frames=0,proposals=0,valid_GT_rows=0,original_output_rows=0,virtual_candidates=0,
            unique_uncovered_GT_with_positive_candidate=0,frames_with_positive_candidate=0)
        for i,f in enumerate(arrays['frame_ids']):
            frame=int(f);lo,hi=map(int,arrays['offsets'][i:i+2]);left,right=map(int,bounds[i:i+2])
            d=arrays['detections'][lo:hi];current_GT=gt[left:right];obs=observations(d,arrays['rgb'][lo:hi],arrays['ir'][lo:hi])
            rows=teacher.update(sequence,frame,d,arrays['rgb'][lo:hi],arrays['ir'][lo:hi])
            for row in rows:digest.update(mot_line(row).encode())
            visible=np.asarray([np.r_[box[:2],box[:2]+box[2:]]for f,box,identity,score in rows],np.float64).reshape(-1,4)
            emitted=np.asarray([np.r_[box[:2],box[:2]+box[2:],score]for f,box,identity,score in rows],np.float64).reshape(-1,5)
            output_labels=current_detection_labels(emitted,current_GT,frame=frame,split='train')
            covered={int(v)for v in output_labels if v>0};positives=set();valid=(current_GT[:,6]>0)&(current_GT[:,7]==1)
            gboxes=current_GT[:,2:6].copy();gboxes[:,2:]+=gboxes[:,:2]
            part='early'if frame<=int(.75*q['frames'])else'late'
            for track in teacher.core.lost_stracks:
                gap=frame-int(track.frame_id)
                if track.state!=TrackState.Lost or not 1<=gap<=25 or not track.is_activated or track.accepted<3 or track.rgb is None:continue
                if track.tlwh[2]*track.tlwh[3]<=100:continue
                internal=int(track.track_id);assert internal not in {r[2]for r in rows}
                box,x=candidate(track,frame,d,obs,visible);origin=origins.origin(internal,frame)
                features.update(np.asarray([frame,internal],np.int64).tobytes()+x.tobytes())
                overlap=matching.ious(box[None],gboxes)[0]
                if origin is None:reason='unknown_origin'
                elif (~valid).any()and float(overlap[~valid].max())>=.2:reason='unknown_ignored'
                else:
                    match=np.flatnonzero(valid&(current_GT[:,1]==origin))
                    assert len(match)<=1
                    if not len(match):reason='negative_origin_absent'
                    elif origin in covered:reason='negative_duplicate_visible'
                    elif overlap[match[0]]>=.5:reason='positive_uncovered';positives.add(origin)
                    elif overlap[match[0]]<.2:reason='negative_bad_geometry'
                    else:reason='unknown_IoU_band'
                buckets[part][band(gap)][reason]+=1;stats['virtual_candidates']+=1
            assignments=teacher.current_proposal_assignments(frame);events=[]
            for internal,index in assignments.items():
                if obs[index]['reliable']and seed_count.get(internal,0)<3:
                    events.append((internal,index));seed_count[internal]=seed_count.get(internal,0)+1
            labels=current_detection_labels(d,current_GT,frame=frame,split='train');origins.commit(frame,events,labels)
            stats['frames']+=1;stats['proposals']+=len(d);stats['valid_GT_rows']+=int(valid.sum());stats['original_output_rows']+=len(rows)
            stats['unique_uncovered_GT_with_positive_candidate']+=len(positives);stats['frames_with_positive_candidate']+=int(bool(positives))
            if frame%1000==0 or frame==q['frames']:
                guard_storage(work);atomic_json(base/'progress.json',dict(sequence=sequence,frame=frame,frames=q['frames'],seconds=time.monotonic()-started))
        assert stats['frames']==q['frames']and stats['proposals']==reference[sequence]['proposal_rows']
        assert digest.hexdigest()==reference[sequence]['original_teacher_MOT_sha256']and origins.frame==q['frames']
        coverage[sequence]=dict(statistics=stats,by_phase_and_gap=buckets,original_teacher_MOT_exact=True,
            original_teacher_MOT_sha256=digest.hexdigest(),GT_free_feature_packet_sha256=features.hexdigest(),bank_sha256=q['bank_sha256'],GT_sha256=q['GT_sha256'])
        print(json.dumps(dict(sequence=sequence,statistics=stats,by_phase_and_gap=buckets)),flush=True)
    registered();frozen_json(base/'complete_Train.json',dict(status='COMPLETE_ALL6_TRAIN_CAUSAL_GAP_COMPLETION_CANDIDATE_DIAGNOSIS',
        coverage=coverage,Train_frames=27251,unique_Train_proposals=sum(r['statistics']['proposals']for r in coverage.values()),
        all_original_teacher_MOT_exact=True,all_proposals_delivered=True,feature_names=list(FEATURES),new_tracker=False,new_models=False,
        no_predicted_box_emitted_or_fed_back=True,pair_or_feature_disk_dump=False,Dev_rows=0,Test_rows=0,
        local_IoU_05_coverage_proxy_not_a_global_HOTA_upper_bound=True,duplicate_visible_targets_explicitly_negative=True,
        unknown_origin_ignore_or_IoU_band_not_negative=True,seed=0,new_GPU_work=False,seconds=time.monotonic()-started,
        historical_Test_previously_exposed=True,goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_ALL6_TRAIN_CAUSAL_GAP_COMPLETION_CANDIDATE_DIAGNOSIS',frames=27251)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','run']);a=p.parse_args();register()if a.mode=='register'else run()
