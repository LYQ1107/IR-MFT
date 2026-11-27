#!/usr/bin/env python3
"""Read-only full-Train audit of conditional birth collection and exact fit."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.fit_mft_birth_conditioned_quality import registered,locations,SPEC,utc
from tools.run_mft_proposal_quality import fitted as quality_fitted
from tools.mft_proposal_quality import ProposalConfirmation,QualityHead,prepare_kind,current_features,IR_FEATURES
from tools.mft_birth_only import BirthOnlyTeacher
from tools.mft_candidate_risk_data import load_arrays,load_GT
from tools.mft_candidate_risk_labels import current_detection_labels
from tools.fit_mft_candidate_risk import threshold_values
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.native_pyramid_stream_tracks import mot_line
from tools.run_mft_candidate_train_bank import atomic_json
from tools.run_mft_native_memory_priority_v3 import identity
from tools.run_pseudo_ir_comparison import sha256


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    started=time.monotonic();reg=registered();plan,protocol,work,base,train=locations();spec=read(SPEC)
    e=read(base/'fit_r1.execution.json');fit=read(base/'fit_complete.json');raw=read(fit['model_file']);coverage=read(base/'Train_coverage.json')['coverage']
    assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(e['identity']['PID'])is None and fit['gradient_converged']
    assert sha256(fit['model_file'])==fit['model_sha256']
    _,old,views,_,_=quality_fitted();assert raw['views']==read(old['model_file'])['views']
    heads={k:QualityHead.from_dict(v)for k,v in raw['heads'].items()}
    source={**reg['source_hashes'],'tools/audit_mft_birth_conditioned_quality.py':sha256(Path(__file__))}
    frozen_json(base/'independent_fit_audit_registered.json',dict(status='REGISTERED_READ_ONLY_CONDITIONAL_BIRTH_FIT_REPLAY',UTC=utc(),
        source_hashes=source,fit_model_sha256=fit['model_sha256'],fit_execution_sha256=sha256(base/'fit_r1.execution.json'),
        optimizer_updates=0,Dev_or_Test_rows=0))
    cache={};rebuilt={}
    for sequence,q in reg['qualified_Train'].items():
        a=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256'])
        teacher=BirthOnlyTeacher(threshold_values(work)['Train_unknown_speed_upper']);shadow=ProposalConfirmation()
        parts={k:[]for k in ['x','labels','streak','early']};feature_hash=hashlib.sha256();label_hash=hashlib.sha256();mot=hashlib.sha256()
        for index,fid in enumerate(a['frame_ids']):
            frame=int(fid);lo,hi=map(int,a['offsets'][index:index+2]);left,right=map(int,bounds[index:index+2]);d=a['detections'][lo:hi]
            x=current_features(d,a['rgb'][lo:hi],a['ir'][lo:hi],views);streak=shadow.update(sequence,frame,d)
            rows=teacher.update(sequence,frame,d,a['rgb'][lo:hi],a['ir'][lo:hi]);mot.update(''.join(map(mot_line,rows)).encode())
            chosen=np.array([v['proposal_index']for v in teacher.core.last_birth_events if not v['original_allowed']],np.int64)
            if len(chosen):chosen=chosen[(d[chosen,4]>.6)&(d[chosen,4]<.7)]
            labels=current_detection_labels(d,gt[left:right],frame=frame,split='train')
            feature_hash.update(np.asarray([frame,len(chosen)],np.int64).tobytes()+chosen.tobytes()+x[chosen].tobytes()+streak[chosen].tobytes())
            label_hash.update(np.asarray([frame,len(d)],np.int64).tobytes()+d.tobytes()+labels.tobytes())
            for k,v in [('x',x[chosen]),('labels',labels[chosen]),('streak',streak[chosen]),
                ('early',np.full(len(chosen),frame<=int(.75*q['frames']),bool))]:parts[k].append(v)
            if frame%1000==0 or frame==q['frames']:
                guard_storage(work);atomic_json(base/'independent_audit_progress.json',dict(sequence=sequence,frame=frame,frames=q['frames'],seconds=time.monotonic()-started))
        expected=coverage[sequence]
        assert feature_hash.hexdigest()==expected['conditional_feature_and_streak_packet_sha256']
        assert label_hash.hexdigest()==expected['current_Train_label_packet_sha256']and mot.hexdigest()==expected['original_zero_gate_MOT_sha256']
        cache[sequence]={k:np.concatenate(v)for k,v in parts.items()};rebuilt[sequence]=dict(frames=q['frames'],conditional_rows=len(cache[sequence]['labels']),
            conditional_features_streak_labels_and_zero_gate_MOT_exact=True)
    diagnostics={};calibration={};L2=spec['quality_head']['L2']
    for kind,m in heads.items():
        early=np.concatenate([prepare_kind(a['x'][a['early']&(a['labels']>=0)],kind)for a in cache.values()])
        expected_mean=early.mean(0);expected_scale=np.maximum(early.std(0),1e-4)
        if kind=='RGB':expected_mean[list(IR_FEATURES)]=0;expected_scale[list(IR_FEATURES)]=1
        mean_error=float(np.max(abs(expected_mean-m.mean)));scale_error=float(np.max(abs(expected_scale-m.scale)))
        assert mean_error<1e-10 and scale_error<3e-8
        loss=0.;gradient=np.zeros(12);hessian=np.zeros((12,12));total_weight=0.;rows=0
        for sequence,a in cache.items():
            keep=a['early']&(a['labels']>=0);y=(a['labels'][keep]>0).astype(np.float64)
            value=prepare_kind(a['x'][keep],kind);x=np.column_stack([np.clip((value-m.mean)/m.scale,-8,8),np.ones(len(value))])
            count=coverage[sequence]['counts']['early'];pos,neg=count['known_positive'],count['known_background'];classes=int(pos>0)+int(neg>0)
            weights=np.where(y==1,1/max(pos,1),1/max(neg,1))/(6*classes)
            z=x@m.coefficients;p=np.exp(-np.logaddexp(0,-z))
            loss+=float(weights@(np.logaddexp(0,z)-y*z));gradient+=x.T@(weights*(p-y))
            hessian+=x.T@((weights*p*(1-p))[:,None]*x);total_weight+=float(weights.sum());rows+=len(y)
        penalty=m.coefficients.copy();penalty[-1]=0;loss+=.5*L2*float(penalty@penalty);gradient+=L2*penalty
        hessian+=np.diag(np.r_[np.full(11,L2),0.]);g=float(np.linalg.norm(gradient));step=float(np.linalg.norm(np.linalg.solve(hessian+1e-8*np.eye(12),gradient)))
        expected=fit['final_diagnostics'][kind]
        assert abs(total_weight-1)<1e-7 and rows==fit['early_known_fit_rows']
        assert abs(loss-expected['weighted_objective'])<1e-10 and abs(g-expected['gradient_norm'])<1e-9
        assert abs(step-expected['next_Newton_step_norm'])<1e-7
        diagnostics[kind]=dict(weighted_objective=loss,gradient_norm=g,next_step_norm=step,
            independent_mean_max_error=mean_error,independent_scale_max_error=scale_error,weight_sum=total_weight)
        calibration[kind]={}
        for condition in ['all','confirmed']:
            probabilities=[];targets=[]
            for a in cache.values():
                keep=(~a['early'])&(a['labels']>=0)
                if condition=='confirmed':keep&=a['streak']>=3
                probabilities.append(m.probabilities(a['x'][keep]));targets.append(a['labels'][keep])
            p=np.concatenate(probabilities);labels=np.concatenate(targets);bad=np.sort(p[labels==0]);points={}
            for name,qvalue in [('q95',.95),('q99',.99),('q999',.999)]:
                position=qvalue*(len(bad)-1);left=int(np.floor(position));right=int(np.ceil(position));fraction=position-left
                threshold=float((1-fraction)*bad[left]+fraction*bad[right]);saved=raw['thresholds'][kind][name][condition]
                assert abs(threshold-saved)<2e-12
                passing=p>saved;expected=fit['late_calibration'][kind][condition]['operating_points'][name]
                assert int((passing&(labels>0)).sum())==expected['known_positive_passing']and int((passing&(labels==0)).sum())==expected['known_background_passing']
                points[name]=dict(independently_sorted_threshold=threshold,stored_threshold=saved,passing_counts_exact=True)
            calibration[kind][condition]=points
    assert sum(v['conditional_rows']for v in rebuilt.values())==18873
    assert sha256(fit['model_file'])==fit['model_sha256'];registered()
    for n,h in source.items():assert sha256(ROOT/n)==h
    frozen_json(base/'independent_fit_audit.json',dict(status='COMPLETE_READ_ONLY_ALL_TRAIN_CONDITIONAL_BIRTH_FIT_AUDIT',UTC=utc(),
        coverage=rebuilt,Train_frames=27251,conditional_rows=18873,head_diagnostics=diagnostics,independently_sorted_calibration=calibration,
        all_foreground_views_exact_parent=True,model_sha256=fit['model_sha256'],model_and_normalization_bytes_unchanged=True,
        optimizer_updates=0,all_labels_after_GT_free_features_and_original_tracker=True,source_hashes=source,
        Dev_rows=0,Test_rows=0,new_GPU_work=False,seconds=time.monotonic()-started,goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_READ_ONLY_ALL_TRAIN_CONDITIONAL_BIRTH_FIT_AUDIT',Train_frames=27251,rows=18873)),flush=True)


if __name__=='__main__':main()
