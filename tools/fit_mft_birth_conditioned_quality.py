#!/usr/bin/env python3
"""Refit identical quality heads on actual original-teacher birth decisions."""
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
from tools.run_mft_birth_only import registered as parent_registered,locations as parent_locations,utc
from tools.run_mft_proposal_quality import fitted as quality_fitted
from tools.mft_birth_only import BirthOnlyTeacher
from tools.mft_proposal_quality import ProposalConfirmation,QualityHead,prepare_kind,current_features,FEATURE_NAMES
from tools.mft_candidate_risk_model import NewtonPass,class_video_weights
from tools.mft_candidate_risk_data import load_arrays,load_GT
from tools.mft_candidate_risk_labels import current_detection_labels
from tools.fit_mft_candidate_risk import threshold_values
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.native_pyramid_stream_tracks import mot_line
from tools.run_mft_candidate_train_bank import atomic_json
from tools.run_mft_native_memory_priority_v3 import identity
from tools.run_pseudo_ir_comparison import sha256

SPEC=ROOT/'configs/perceptual_birth_conditioned_quality_20261127.json'


def locations():
    p,protocol,w,b,t=parent_locations();return p,protocol,w,w/'birth_conditioned_quality_20261127',t


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    source=parent_registered();q,old,views,heads,thresholds=quality_fitted();plan,protocol,work,base,train=locations()
    parent=work/'birth_only_20261127';e=read(parent/'diagnose_r1.execution.json');d=read(parent/'complete_Train_diagnosis.json')
    assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(e['identity']['PID'])is None
    assert d['Train_frames']==27251 and d['proposals']==249639
    sources=dict(source['source_hashes'])
    for n in ['tools/fit_mft_birth_conditioned_quality.py',str(SPEC.relative_to(ROOT))]:sources[n]=sha256(ROOT/n)
    paths=[SPEC,parent/'complete_Train_diagnosis.json',parent/'diagnose_r1.execution.json',train/'fit_complete.json']
    frozen_json(base/'registered.json',dict(status='REGISTERED_CONDITIONAL_BIRTH_POPULATION_FIT_BEFORE_COLLECTION',UTC=utc(),
        source_hashes=sources,artifact_hashes={str(p):sha256(p)for p in paths},qualified_Train=q['qualified_Train'],
        parent_quality_model_sha256=old['model_sha256'],original_features=list(FEATURE_NAMES),new_NN_training=False,
        Dev_rows=0,Test_rows=0,seed=0,goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='REGISTERED_CONDITIONAL_BIRTH_POPULATION_FIT_BEFORE_COLLECTION',sources=len(sources))),flush=True)


def registered():
    parent_registered();plan,protocol,work,base,train=locations();r=read(base/'registered.json')
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in r['artifact_hashes'].items():assert sha256(n)==h,n
    return r


def fit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();_,old,views,_,_=quality_fitted();plan,protocol,work,base,train=locations();spec=read(SPEC)
    assert not(base/'private_models.json').exists()and not(base/'fit_complete.json').exists()
    started=time.monotonic();prior=read(work/'birth_only_20261127/complete_Train_diagnosis.json');cache={};coverage={};payload=0
    for sequence,q in reg['qualified_Train'].items():
        a=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256'])
        teacher=BirthOnlyTeacher(threshold_values(work)['Train_unknown_speed_upper']);shadow=ProposalConfirmation()
        parts={k:[]for k in ['x','labels','streak','early']};mot=hashlib.sha256();labels_packet=hashlib.sha256();features=hashlib.sha256()
        counts={phase:dict(opportunities=0,known_positive=0,known_background=0,unknown=0)for phase in ['early','late']}
        for index,fid in enumerate(a['frame_ids']):
            frame=int(fid);lo,hi=map(int,a['offsets'][index:index+2]);left,right=map(int,bounds[index:index+2])
            d=a['detections'][lo:hi];rgb=a['rgb'][lo:hi];ir=a['ir'][lo:hi]
            x=current_features(d,rgb,ir,views);streak=shadow.update(sequence,frame,d)
            rows=teacher.update(sequence,frame,d,rgb,ir);mot.update(''.join(map(mot_line,rows)).encode())
            candidates=np.asarray([e['proposal_index']for e in teacher.core.last_birth_events if not e['original_allowed']],np.int64)
            if len(candidates):candidates=candidates[(d[candidates,4]>.6)&(d[candidates,4]<.7)]
            # Current labels only follow all GT-free features and original tracking.
            labels=current_detection_labels(d,gt[left:right],frame=frame,split='train')
            labels_packet.update(np.asarray([frame,len(d)],np.int64).tobytes()+d.tobytes()+labels.tobytes())
            features.update(np.asarray([frame,len(candidates)],np.int64).tobytes()+candidates.tobytes()+x[candidates].tobytes()+streak[candidates].tobytes())
            early=frame<=int(.75*q['frames']);chosen=labels[candidates];c=counts['early'if early else'late']
            c['opportunities']+=len(candidates);c['known_positive']+=int((chosen>0).sum());c['known_background']+=int((chosen==0).sum());c['unknown']+=int((chosen<0).sum())
            parts['x'].append(x[candidates]);parts['labels'].append(chosen);parts['streak'].append(streak[candidates]);parts['early'].append(np.full(len(candidates),early,bool))
            if frame%1000==0 or frame==q['frames']:
                guard_storage(work);atomic_json(base/'progress.json',dict(status='COLLECT_ALL_TRAIN_CONDITIONAL_BIRTH_POPULATION',
                    sequence=sequence,frame=frame,frames=q['frames'],completed_sequences=len(coverage),seconds=time.monotonic()-started))
        previous=prior['coverage'][sequence]
        assert mot.hexdigest()==previous['zero_gate_MOT_sha256']and labels_packet.hexdigest()==previous['current_Train_label_packet_sha256']
        for phase,c in counts.items():
            old_count=previous['by_case'][phase]['BO_unconditional_c0']
            assert all(c[k]==old_count[k]for k in c)
        data={k:np.concatenate(v)for k,v in parts.items()};payload+=sum(v.nbytes for v in data.values());assert payload<=spec['private_RAM_cap_bytes']
        cache[sequence]=data;coverage[sequence]=dict(frames=q['frames'],all_proposals=len(a['detections']),counts=counts,
            original_zero_gate_MOT_sha256=mot.hexdigest(),current_Train_label_packet_sha256=labels_packet.hexdigest(),
            conditional_feature_and_streak_packet_sha256=features.hexdigest(),bank_sha256=q['bank_sha256'],GT_sha256=q['GT_sha256'])
        print(json.dumps(dict(sequence=sequence,counts=counts)),flush=True)
    assert sum(v['counts'][phase]['opportunities']for v in coverage.values()for phase in ['early','late'])==spec['expected_birth_rows']
    frozen_json(base/'Train_coverage.json',dict(status='COMPLETE_ALL27251_TRAIN_CONDITIONAL_OPPORTUNITY_COLLECTION',UTC=utc(),
        coverage=coverage,Train_frames=27251,all_proposals=249639,conditional_rows=18873,RAM_payload_bytes=payload,
        no_feature_disk_bank=True,unknown_not_background=True,new_Dev_or_Test=False))
    known_total=sum(int((a['early']&(a['labels']>=0)).sum())for a in cache.values());heads={}
    for kind in spec['fit_kinds']:
        total=np.zeros(11);squares=total.copy()
        for a in cache.values():
            x=prepare_kind(a['x'][a['early']&(a['labels']>=0)],kind);total+=x.sum(0);squares+=(x*x).sum(0)
        mean=total/known_total;scale=np.maximum(np.sqrt(np.maximum(squares/known_total-mean*mean,0)),1e-4)
        heads[kind]=QualityHead(mean,scale,kind)
    def accumulate(model):
        acc=NewtonPass(model,regularization=spec['quality_head']['L2'])
        for sequence,a in cache.items():
            keep=a['early']&(a['labels']>=0);x=a['x'][keep];y=(a['labels'][keep]>0).astype(np.float64);c=coverage[sequence]['counts']['early']
            for lo in range(0,len(y),4096):acc.add(x[lo:lo+4096],y[lo:lo+4096],class_video_weights(y[lo:lo+4096],c['known_positive'],c['known_background'],6))
        return acc
    history=[];config=spec['quality_head']
    for index in range(config['maximum_Newton_updates']):
        row=dict(pass_index=index+1,heads={k:accumulate(m).step()for k,m in heads.items()});history.append(row)
        atomic_json(base/'fit_history.json',dict(rows=history))
        if index+1>=config['fixed_Newton_updates']and(index+1-config['fixed_Newton_updates'])%config['extension_block']==0:
            norms={k:float(np.linalg.norm(accumulate(m).objective()[1]))for k,m in heads.items()}
            if max(norms.values())<=config['extend_only_if_Train_gradient_norm_exceeds']:break
    final={}
    for k,m in heads.items():
        loss,g,h=accumulate(m).objective();final[k]=dict(weighted_objective=loss,gradient_norm=float(np.linalg.norm(g)),
            next_Newton_step_norm=float(np.linalg.norm(np.linalg.solve(h+1e-8*np.eye(len(g)),g))))
    converged=all(v['gradient_norm']<=config['extend_only_if_Train_gradient_norm_exceeds']for v in final.values())
    assert converged,'Preserve unconverged fit rather than qualify Dev'
    thresholds={};calibration={}
    for k,m in heads.items():
        thresholds[k]={name:{}for name in ['q95','q99','q999']};calibration[k]={}
        for condition in ['all','confirmed']:
            probability=[];labels=[]
            for a in cache.values():
                keep=(~a['early'])&(a['labels']>=0)
                if condition=='confirmed':keep&=a['streak']>=3
                probability.append(m.probabilities(a['x'][keep]));labels.append(a['labels'][keep])
            probability=np.concatenate(probability);labels=np.concatenate(labels);bad=probability[labels==0];assert len(bad)>0
            calibration[k][condition]=dict(known_positive=int((labels>0).sum()),known_background=len(bad),operating_points={})
            for name,q in [('q95',.95),('q99',.99),('q999',.999)]:
                threshold=float(np.quantile(bad,q));thresholds[k][name][condition]=threshold;selected=probability>threshold
                calibration[k][condition]['operating_points'][name]=dict(threshold=threshold,known_positive_passing=int((selected&(labels>0)).sum()),
                    known_background_passing=int((selected&(labels==0)).sum()))
    raw_parent=read(old['model_file']);model=dict(views=raw_parent['views'],heads={k:m.to_dict()for k,m in heads.items()},thresholds=thresholds,
        foreground_views_reused_from_model_sha256=old['model_sha256'],seed=0)
    frozen_json(base/'private_models.json',model);registered()
    frozen_json(base/'fit_complete.json',dict(status='COMPLETE_CONVERGED_TRAIN_CONDITIONAL_BIRTH_QUALITY',UTC=utc(),
        Train_frames=27251,all_proposals=249639,conditional_rows=18873,early_known_fit_rows=known_total,
        Newton_updates=len(history),final_diagnostics=final,gradient_converged=converged,late_calibration=calibration,thresholds=thresholds,
        source_model_sha256=old['model_sha256'],foreground_views_numerically_byte_reused=True,
        model_file=str(base/'private_models.json'),model_sha256=sha256(base/'private_models.json'),
        source_hashes=reg['source_hashes'],coverage_sha256=sha256(base/'Train_coverage.json'),registration_sha256=sha256(base/'registered.json'),
        new_NN_training=False,new_training_seeds=0,Dev_rows=0,Test_rows=0,no_tracking_accuracy_claim=True,
        seconds=time.monotonic()-started,goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_CONVERGED_TRAIN_CONDITIONAL_BIRTH_QUALITY',updates=len(history),diagnostics=final)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','fit']);args=p.parse_args()
    {'register':register,'fit':fit}[args.mode]()
