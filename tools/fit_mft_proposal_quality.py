#!/usr/bin/env python3
"""All-Train binary foreground evidence and confirmation-specific calibration."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.mft_proposal_quality import (ProposalConfirmation,ForegroundView,QualityHead,
    FEATURE_NAMES,IR_FEATURES,prepare_kind,geometry_features,current_features)
from tools.mft_candidate_risk_teacher import normalized_descriptors
from tools.mft_candidate_risk_data import qualify_Train,load_arrays,load_GT
from tools.mft_candidate_risk_labels import current_detection_labels
from tools.mft_candidate_risk_model import NewtonPass,class_video_weights
from tools.fit_mft_candidate_risk import locations as parent_locations
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.run_mft_candidate_train_bank import atomic_json
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity

SPEC=ROOT/'configs/perceptual_proposal_quality_20261127.json'
SOURCES=['tools/mft_proposal_quality.py','tools/fit_mft_proposal_quality.py',
    'tests/test_mft_proposal_quality.py',str(SPEC.relative_to(ROOT))]


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def locations():
    plan,protocol,work,parent=parent_locations();return plan,protocol,work,work/'proposal_quality_20261127'


def sources():
    plan,protocol,work,base=locations();r=read(work/'proposal_confidence_diagnosis_20261127/registered.json')
    result={**r['source_hashes'],**{n:sha256(ROOT/n)for n in SOURCES}}
    for n,h in result.items():assert sha256(ROOT/n)==h,n
    return result


def engineer():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan,protocol,work,base=locations();s=sources();q=qualify_Train(work,protocol);rows={}
    direction=np.zeros(512);direction[0]=1;views={k:ForegroundView(np.zeros(512),direction)for k in ['RGB','IR']}
    for sequence,v in q.items():
        a=load_arrays(v['bank_path'],v['frames'],v['bank_sha256'])
        def replay(future=False):
            temporal=ProposalConfirmation();packets=[]
            for i in range(64):
                lo,hi=map(int,a['offsets'][i:i+2]);d=a['detections'][lo:hi].copy();r=a['rgb'][lo:hi].copy();ir=a['ir'][lo:hi].copy()
                if future and i>=32:d[:,:4]+=777;r*=-1;ir*=-1
                before=[x.copy()for x in [d,r,ir]];streak=temporal.update(sequence,i+1,d);x=current_features(d,r,ir,views)
                packets.append(hashlib.sha256(d.tobytes()+x.tobytes()+streak.tobytes()).hexdigest())
                for actual,original in zip([d,r,ir],before):np.testing.assert_array_equal(actual,original)
            return packets
        normal=replay();changed=replay(True);assert normal[:32]==changed[:32]
        rows[sequence]=dict(frames=64,prefix_frames=32,GT_labels_not_parsed_or_used=True,future_suffix_prefix_features_and_confirmation_exact=True,
            input_arrays_unchanged=True,bank_sha256=v['bank_sha256'])
        guard_storage(work)
    assert sources()==s
    frozen_json(base/'real_Train_engineering.json',dict(status='COMPLETE_ALL6_REAL_TRAIN_QUALITY_ENGINEERING',UTC=utc(),
        rows=rows,sequences=6,frames=384,source_hashes=s,qualified_Train=q,new_quality_models=False,new_accuracy=False))
    print(json.dumps(dict(status='COMPLETE_ALL6_REAL_TRAIN_QUALITY_ENGINEERING',frames=384)),flush=True)


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan,protocol,work,base=locations();s=sources();proof=read(base/'CPU_regressions_r2.json')
    assert proof['actual_tool_exit_code']==0 and proof['tests_passed']==5
    for n,h in proof['source_hashes'].items():assert sha256(ROOT/n)==h,n
    e=read(base/'engineering_r1.execution.json');p=read(base/'real_Train_engineering.json')
    assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(e['identity']['PID'])is None
    assert p['source_hashes']==s and p['frames']==384 and p['sequences']==6
    paths=[SPEC,base/'CPU_regressions_r2.json',base/'real_Train_engineering.json',base/'engineering_r1.execution.json',
        work/'proposal_confidence_diagnosis_20261127/complete_Train.json']
    v=dict(status='REGISTERED_ALL6_TRAIN_PROPOSAL_QUALITY_FIT_BEFORE_COLLECTION',UTC=utc(),source_hashes=s,
        artifact_hashes={str(p):sha256(p)for p in paths},qualified_Train=qualify_Train(work,protocol),
        specification_sha256=sha256(SPEC),feature_names=list(FEATURE_NAMES),Train_frames=27251,Dev_rows=0,Test_rows=0,
        seed=0,private_input_RAM_cap_bytes=2*2**30,descriptor_or_pair_disk_dump=False,GT_filters_candidate_graph=False,
        new_GPU_work=False,historical_Test_previously_exposed=True,goal_met=False,supported_innovations=0)
    assert not(base/'fit_registered.json').exists();frozen_json(base/'fit_registered.json',v)
    print(json.dumps(dict(status=v['status'],sources=len(s),Train_frames=27251)),flush=True)


def registered():
    plan,protocol,work,base=locations();r=read(base/'fit_registered.json')
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in r['artifact_hashes'].items():assert sha256(n)==h,n
    return r


def fit_view(cache,coverage,cue,ridge_fraction):
    mean={0:np.zeros(512),1:np.zeros(512)};weight={0:0.,1:0.};within=np.zeros((512,512));groups=[]
    for sequence,a in cache.items():
        early=a['early'];known=a['known'];y=a['y'];classes=[c for c in [0,1]if (early&known&(y==c)).any()]
        assert classes
        for c in classes:
            indices=np.flatnonzero(early&known&(y==c));x=a[cue][indices].astype(np.float64)
            m=x.mean(axis=0);w=1/(len(cache)*len(classes));mean[c]+=w*m;weight[c]+=w
            residual=x-m;within+=w*(residual.T@residual)/len(x)
            groups.append(dict(sequence=sequence,target=c,rows=len(x),group_weight=w))
    assert weight[0]>0 and weight[1]>0
    mu0,mu1=mean[0]/weight[0],mean[1]/weight[1]
    ridge=max(ridge_fraction*float(np.trace(within))/512,1e-8)
    direction=np.linalg.solve(within+ridge*np.eye(512),mu1-mu0);norm=float(np.linalg.norm(direction));assert norm>1e-9
    direction/=norm
    model=ForegroundView((mu1+mu0)/2,direction)
    return model,dict(cue=cue,groups=groups,within_trace=float(np.trace(within)),absolute_ridge=ridge,
        ridge_fraction=ridge_fraction,raw_solution_norm=norm,normalized_direction_norm=float(np.linalg.norm(direction)),
        input_dimensions=512,task='Known foreground versus known background;not identity discrimination')


def fit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();plan,protocol,work,base=locations();spec=read(SPEC);started=time.monotonic();cache={};coverage={};payload=0
    assert not(base/'fit_complete.json').exists()and not(base/'private_models.json').exists()
    old=read(work/'proposal_confidence_diagnosis_20261127/complete_Train.json')
    for sequence,q in reg['qualified_Train'].items():
        a=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256'])
        n=len(a['detections']);data=dict(rgb=normalized_descriptors(a['rgb'],n),ir=normalized_descriptors(a['ir'],n),
            geometry=np.empty((n,7),np.float32),y=np.empty(n,np.float32),known=np.empty(n,bool),
            early=np.empty(n,bool),streak=np.empty(n,np.int32),weak=(a['detections'][:,4]>.3)&(a['detections'][:,4]<=.7))
        confirmation=ProposalConfirmation();digest=hashlib.sha256();counts={p:dict(proposals=0,known_positive=0,known_negative=0,unknown=0,
            weak_known_positive=0,weak_known_negative=0,confirmed_weak_positive=0,confirmed_weak_negative=0)for p in ['early','late']}
        for i,f in enumerate(a['frame_ids']):
            frame=int(f);lo,hi=map(int,a['offsets'][i:i+2]);left,right=map(int,bounds[i:i+2]);d=a['detections'][lo:hi]
            labels=current_detection_labels(d,gt[left:right],frame=frame,split='train')
            geometry=geometry_features(d);streak=confirmation.update(sequence,frame,d)
            early=frame<=int(.75*q['frames']);data['geometry'][lo:hi]=geometry;data['streak'][lo:hi]=streak
            data['known'][lo:hi]=labels>=0;data['y'][lo:hi]=labels>0;data['early'][lo:hi]=early
            c=counts['early'if early else'late'];weak=data['weak'][lo:hi]
            c['proposals']+=len(d);c['known_positive']+=int((labels>0).sum());c['known_negative']+=int((labels==0).sum());c['unknown']+=int((labels<0).sum())
            for key,mask in [('weak_known_positive',weak&(labels>0)),('weak_known_negative',weak&(labels==0)),
                ('confirmed_weak_positive',weak&(labels>0)&(streak>=3)),('confirmed_weak_negative',weak&(labels==0)&(streak>=3))]:c[key]+=int(mask.sum())
            digest.update(np.asarray([frame,len(d)],np.int64).tobytes()+d.tobytes()+geometry.tobytes()+streak.tobytes()+labels.tobytes())
            if frame%1000==0 or frame==q['frames']:
                guard_storage(work);atomic_json(base/'progress.json',dict(status='COLLECT_ALL_TRAIN_QUALITY_AND_CONFIRMATION',UTC=utc(),
                    sequence=sequence,frame=frame,frames=q['frames'],completed_payload_bytes=payload,seconds=time.monotonic()-started))
        assert sum(c['proposals']for c in counts.values())==n
        prior=old['coverage'][sequence];assert n==prior['proposals']and sum(c['known_positive']for c in counts.values())==prior['positive_proposal_rows']
        payload+=sum(v.nbytes for v in data.values());assert payload<=reg['private_input_RAM_cap_bytes']
        cache[sequence]=data;coverage[sequence]=dict(frames=q['frames'],proposals=n,counts=counts,
            all_Train_proposals_delivered=True,GT_free_confirmation_and_feature_packet_sha256=digest.hexdigest(),
            bank_sha256=q['bank_sha256'],GT_sha256=q['GT_sha256'])
        print(json.dumps(dict(sequence=sequence,counts=counts,RAM_payload_bytes=payload)),flush=True)
        del a,gt
    frozen_json(base/'Train_coverage.json',dict(status='COMPLETE_ALL27251_TRAIN_QUALITY_CANDIDATES',UTC=utc(),coverage=coverage,
        Train_frames=27251,proposals=sum(r['proposals']for r in coverage.values()),private_RAM_payload_bytes=payload,
        descriptor_or_pair_disk_dump=False,unknown_not_background=True,Dev_rows=0,Test_rows=0,
        registration_sha256=sha256(base/'fit_registered.json')))
    views={};view_details={}
    for key,cue in [('RGB','rgb'),('IR','ir')]:
        views[key],view_details[key]=fit_view(cache,coverage,cue,spec['foreground_view']['ridge_fraction'])
        print(json.dumps(dict(view=key,details=view_details[key])),flush=True)
    for sequence,a in cache.items():
        r=views['RGB'].scores(a['rgb']);i=views['IR'].scores(a['ir'])
        a['x']=np.column_stack([a['geometry'],r,i,np.ones(len(r)),abs(r-i)]).astype(np.float32)
        del a['rgb'],a['ir'],a['geometry']
    heads={};early_count=sum(int((a['early']&a['known']).sum())for a in cache.values())
    for kind in ['RGB','paired','duplicatedRGB']:
        total=np.zeros(len(FEATURE_NAMES));squares=total.copy()
        for a in cache.values():
            x=prepare_kind(a['x'][a['early']&a['known']],kind);total+=x.sum(axis=0);squares+=(x*x).sum(axis=0)
        mean=total/early_count;scale=np.maximum(np.sqrt(np.maximum(squares/early_count-mean*mean,0)),1e-4)
        heads[kind]=QualityHead(mean,scale,kind)
    def accumulate(head):
        acc=NewtonPass(head,regularization=spec['quality_head']['L2'])
        for s,a in cache.items():
            keep=a['early']&a['known'];x,y=a['x'][keep],a['y'][keep];c=coverage[s]['counts']['early']
            for lo in range(0,len(y),4096):
                yy=y[lo:lo+4096];acc.add(x[lo:lo+4096],yy,class_video_weights(yy,c['known_positive'],c['known_negative'],len(cache)))
        return acc
    history=[]
    for index in range(spec['quality_head']['fixed_Newton_updates']):
        row=dict(pass_index=index+1,heads={k:accumulate(m).step()for k,m in heads.items()});history.append(row)
        atomic_json(base/'fit_history.json',dict(rows=history));print(json.dumps(row),flush=True)
    final={}
    for k,m in heads.items():
        loss,g,h=accumulate(m).objective();final[k]=dict(loss=loss,gradient_norm=float(np.linalg.norm(g)),
            next_Newton_step_norm=float(np.linalg.norm(np.linalg.solve(h+1e-8*np.eye(len(g)),g))))
    thresholds={};calibration={}
    for k,m in heads.items():
        thresholds[k]={name:{}for name in spec['calibration']['wrong_quantiles']};calibration[k]={}
        for condition in ['all','confirmed']:
            wrong=[];right=[]
            for a in cache.values():
                mask=(~a['early'])&a['known']&a['weak']
                if condition=='confirmed':mask&=a['streak']>=3
                p=m.probabilities(a['x'][mask]);y=a['y'][mask]
                wrong.append(p[y==0]);right.append(p[y==1])
            negative=np.concatenate(wrong);positive=np.concatenate(right)
            for name,q in spec['calibration']['wrong_quantiles'].items():
                thresholds[k][name][condition]=float(np.quantile(negative,q))if len(negative)else 1.
            calibration[k][condition]=dict(known_negative=len(negative),known_positive=len(positive),
                passing_positive={name:int((positive>values[condition]).sum())for name,values in thresholds[k].items()},
                passing_negative={name:int((negative>values[condition]).sum())for name,values in thresholds[k].items()},
                no_negative_threshold1_disables_promotion=len(negative)==0)
    assert not heads['RGB'].coefficients[list(IR_FEATURES)].any()
    private=dict(status='PRIVATE_FOREGROUND_VIEWS_AND_QUALITY_HEADS_NOT_FOR_GIT',UTC=utc(),views={k:v.to_dict()for k,v in views.items()},
        heads={k:v.to_dict()for k,v in heads.items()},thresholds=thresholds,registration_sha256=sha256(base/'fit_registered.json'),seed=0,Train_only=True)
    registered();guard_storage(work);frozen_json(base/'private_models.json',private)
    frozen_json(base/'fit_complete.json',dict(status='COMPLETE_ALL6_TRAIN_FOREGROUND_QUALITY_AND_CONFIRMATION_FIT',UTC=utc(),
        model_file=str(base/'private_models.json'),model_sha256=sha256(base/'private_models.json'),view_fit_details=view_details,
        thresholds=thresholds,late_calibration=calibration,final_diagnostics=final,history=history,
        Train_frames=27251,Train_sequences=6,proposals=sum(r['proposals']for r in coverage.values()),
        early_known_rows=early_count,private_RAM_payload_bytes=payload,descriptor_or_pair_disk_dump=False,
        registration_sha256=sha256(base/'fit_registered.json'),coverage_sha256=sha256(base/'Train_coverage.json'),
        old_identity_or_archive_weights_used_for_quality=False,Dev_rows=0,Test_rows=0,seed=0,new_GPU_work=False,
        gradient_converged=all(v['gradient_norm']<1e-5 for v in final.values()),seconds=time.monotonic()-started,
        no_tracking_accuracy_yet=True,score_quantile_not_a_tracking_error_probability_bound=True,goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_ALL6_TRAIN_FOREGROUND_QUALITY_AND_CONFIRMATION_FIT',thresholds=thresholds,final_diagnostics=final)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['engineer','register','fit']);a=p.parse_args()
    {'engineer':engineer,'register':register,'fit':fit}[a.mode]()
