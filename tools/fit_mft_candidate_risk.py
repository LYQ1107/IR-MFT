#!/usr/bin/env python3
"""All-Train streaming risk fitting, after bank closure and CPU qualification."""
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
from tools.native_pyramid_experiment import context,read,frozen_json
from tools.run_pseudo_ir_comparison import sha256
from tools.native_pyramid_training_loop import guard_storage
from tools.native_pyramid_stream_tracks import IndependentByteTrack,mot_line
from tools.mft_candidate_risk_data import qualify_Train,load_arrays,load_GT,stream_video
from tools.mft_candidate_risk_teacher import CandidateTeacher
from tools.mft_candidate_risk_training_stream import TrainEdgeCollector
from tools.mft_candidate_risk_model import RiskLogistic,NewtonPass,class_video_weights
from tools.mft_candidate_risk_features import FEATURE_NAMES
from tools.run_mft_candidate_train_bank import atomic_json

SPEC=ROOT/'configs/perceptual_candidate_risk_20261127.json'


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def locations():
    plan,study,protocol,calibration,work,checkpoint=context()
    return plan,protocol,work,work/'candidate_risk_20261127'


def require_planning():
    plan,protocol,work,base=locations()
    reg=read(base/'pipeline_registered.json')
    assert reg['specification_sha256']==sha256(SPEC)
    for name,digest in reg['source_hashes'].items():assert sha256(ROOT/name)==digest,name
    for name,digest in reg['artifact_hashes'].items():assert sha256(name)==digest,name
    return reg


def threshold_values(work):
    return read(work/'tracklet_permanence_20261127_r2/Train_thresholds.json')['thresholds']


def engineer():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=require_planning();plan,protocol,work,base=locations()
    qualified=qualify_Train(work,protocol);speed=threshold_values(work)['Train_unknown_speed_upper']
    results={}
    for sequence,q in qualified.items():
        arrays=load_arrays(q['bank_path'],q['frames'],q['bank_sha256'])
        gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256'])
        count=min(64,q['frames']);cut=min(32,count)
        def run(corrupt_suffix=False,rename_labels=False):
            seen=[];batches=[]
            teacher=CandidateTeacher(speed,lambda x,eligible,ids,indices,f:
                seen.append(hashlib.sha256(x.tobytes()+eligible.tobytes()).hexdigest()))
            collector=TrainEdgeCollector(speed,lambda b:batches.append(hashlib.sha256(b.x.tobytes()+b.y.tobytes()).hexdigest()))
            original=IndependentByteTrack();rows=[]
            for i in range(count):
                lo,hi=map(int,arrays['offsets'][i:i+2]);left,right=map(int,bounds[i:i+2]);frame=i+1
                d=arrays['detections'][lo:hi].copy();rgb=arrays['rgb'][lo:hi].copy();ir=arrays['ir'][lo:hi].copy()
                current_gt=gt[left:right].copy()
                if corrupt_suffix and frame>cut:d[:,:4]+=777;rgb*=-1;ir*=-1
                if rename_labels:current_gt[:,1]+=1000000
                own=teacher.update(sequence,frame,d,rgb,ir)
                taught=collector.step(sequence,frame,d,rgb,ir,current_gt,split='train')
                expected=original.update(d,frame)
                text=''.join(map(mot_line,own))
                assert text==''.join(map(mot_line,expected))==''.join(map(mot_line,taught))
                rows.append(text)
            return rows,seen,batches,collector.statistics
        normal=run();future=run(corrupt_suffix=True);renamed=run(rename_labels=True)
        assert normal[0][:cut]==future[0][:cut]and normal[1][:cut]==future[1][:cut]
        assert normal[:3]==renamed[:3]
        results[sequence]=dict(frames=count,original_teacher_MOT_exact=True,
            future_suffix_corruption_prefix_frames=cut,future_prefix_features_and_MOT_exact=True,
            GT_identity_rename_features_targets_and_MOT_exact=True,statistics=normal[3],
            bank_sha256=q['bank_sha256'],GT_sha256=q['GT_sha256'],
            MOT_sha256=hashlib.sha256(''.join(normal[0]).encode()).hexdigest())
        guard_storage(work)
    require_planning()
    frozen_json(base/'CPU_real_bank_engineering.json',dict(status='COMPLETE_ALL6_REAL_TRAIN_BANK_CPU_ENGINEERING',
        UTC=utc(),sequences=6,frames=sum(v['frames']for v in results.values()),rows=results,
        new_fitted_models=0,new_tracking_accuracy=False,GPU_work=False,
        source_registration_sha256=sha256(base/'pipeline_registered.json')))
    print(json.dumps(dict(status='COMPLETE_ALL6_REAL_TRAIN_BANK_CPU_ENGINEERING',sequences=6,
        frames=sum(v['frames']for v in results.values()))),flush=True)


def register_fit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=require_planning();plan,protocol,work,base=locations();qualified=qualify_Train(work,protocol)
    state=read(base/'state.json');jobs=state['phases']['CANDIDATE_RISK_CPU_ENGINEERING']
    assert set(jobs)=={'all6'}and jobs['all6']['status']=='COMPLETE'and jobs['all6']['exit_code']==0
    engineering=read(base/'CPU_real_bank_engineering.json')
    assert engineering['sequences']==6 and engineering['new_fitted_models']==0
    assert engineering['source_registration_sha256']==sha256(base/'pipeline_registered.json')
    value=dict(status='REGISTERED_AFTER_ALL6_TRAIN_AND_REAL_CPU_ENGINEERING_BEFORE_RISK_FIT',
        UTC=utc(),source_hashes=reg['source_hashes'],qualified_Train=qualified,
        pipeline_registration_sha256=sha256(base/'pipeline_registered.json'),
        engineering_sha256=sha256(base/'CPU_real_bank_engineering.json'),
        specification_sha256=sha256(SPEC),fit_parameters=read(SPEC)['fit_parameters'],
        complete_Dev_cases=read(SPEC)['complete_Dev_cases'],Train_frames=27251,
        Dev_or_Test_rows_for_fitting=0,fixed_seed=0,new_GPU_work=False,
        historical_Test_previously_exposed=True,goal_met=False,supported_innovations=0)
    assert not(base/'fit_registered.json').exists();frozen_json(base/'fit_registered.json',value)
    print(json.dumps(dict(status=value['status'],Train_frames=27251)),flush=True)


def require_fit():
    require_planning();plan,protocol,work,base=locations();value=read(base/'fit_registered.json')
    assert value['pipeline_registration_sha256']==sha256(base/'pipeline_registered.json')
    assert value['engineering_sha256']==sha256(base/'CPU_real_bank_engineering.json')
    assert value['qualified_Train']==qualify_Train(work,protocol)
    return value


def early_statistics(qualified,speed,base,progress):
    rows={};references={}
    for sequence,q in qualified.items():
        count=0;positive=0;sums=np.zeros(len(FEATURE_NAMES));squared=sums.copy();digest=hashlib.sha256()
        cut=int(.75*q['frames'])
        def sink(batch):
            nonlocal count,positive
            if batch.frame>cut:return
            x=batch.x.astype(np.float64);count+=len(x);positive+=int(batch.y.sum())
            sums[:]+=x.sum(axis=0);squared[:]+=(x*x).sum(axis=0)
            digest.update(np.asarray([batch.frame,len(x)],dtype='<i8').tobytes())
            digest.update(batch.x.tobytes());digest.update(batch.y.tobytes())
        references[sequence]=stream_video(sequence,q,speed,sink,progress=lambda f:progress('statistics',sequence,f))
        assert count>0,'Do not omit a Train video without usable known edges'
        rows[sequence]=dict(time_cut_frame=cut,count=count,positive=positive,negative=count-positive,
            mean=(sums/count).tolist(),second_moment=(squared/count).tolist(),feature_target_packet_sha256=digest.hexdigest())
    mean=np.mean([v['mean']for v in rows.values()],axis=0)
    second=np.mean([v['second_moment']for v in rows.values()],axis=0)
    scale=np.maximum(np.sqrt(np.maximum(second-mean*mean,0)),1e-4)
    result=dict(status='COMPLETE_ALL6_TRAIN_EARLY_FEATURE_STATISTICS',UTC=utc(),
        feature_names=list(FEATURE_NAMES),rows=rows,equal_video_mean=mean.tolist(),equal_video_scale=scale.tolist(),
        full_teacher_references=references,all_Train_frames=27251,Dev_rows=0,Test_rows=0,
        known_early_edges=sum(v['count']for v in rows.values()),no_candidate_pair_disk_dump=True)
    frozen_json(base/'Train_feature_statistics.json',result)
    return result


def fit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=require_fit();plan,protocol,work,base=locations();qualified=reg['qualified_Train']
    assert not(base/'Train_feature_statistics.json').exists()and not(base/'private_models.json').exists(),\
        'Preserve every actual fitting attempt'
    speed=threshold_values(work)['Train_unknown_speed_upper'];started=time.monotonic()
    def progress(phase,sequence,frame):
        guard_storage(work)
        atomic_json(base/'progress'/'fit.json',dict(phase=phase,sequence=sequence,frame=frame,
            frames=qualified[sequence]['frames'],seconds=time.monotonic()-started,
            GT_split='train',GPU_work=False,new_tracking_accuracy=False))
    stats=early_statistics(qualified,speed,base,progress)
    models={kind:RiskLogistic(stats['equal_video_mean'],stats['equal_video_scale'],kind)for kind in ['RGB','paired']}
    history=[]
    for epoch in range(6):
        updates={kind:NewtonPass(model,1e-3)for kind,model in models.items()}
        for sequence,q in qualified.items():
            row=stats['rows'][sequence];digest=hashlib.sha256()
            def sink(batch):
                if batch.frame>row['time_cut_frame']:return
                digest.update(np.asarray([batch.frame,len(batch.x)],dtype='<i8').tobytes())
                digest.update(batch.x.tobytes());digest.update(batch.y.tobytes())
                weights=class_video_weights(batch.y,row['positive'],row['negative'],len(qualified))
                for update in updates.values():update.add(batch.x,batch.y,weights)
            reference=stream_video(sequence,q,speed,sink,progress=lambda f:progress('Newton_'+str(epoch+1),sequence,f))
            assert reference==stats['full_teacher_references'][sequence]
            assert digest.hexdigest()==row['feature_target_packet_sha256']
        record=dict(pass_index=epoch+1,models={k:u.step()for k,u in updates.items()})
        history.append(record);print(json.dumps(record),flush=True)
        atomic_json(base/'fit_history.json',dict(status='RUNNING_SIX_FIXED_NEWTON_PASSES',rows=history))
    # Keep early-fit coefficients. The late quarter calibrates only, with no refit.
    scores={k:[]for k in models};score_bytes={k:0 for k in models};calibration={}
    for sequence,q in qualified.items():
        counts=dict(known=0,positive=0,negative=0);diagnostics={k:dict(Brier_sum=0.,positive_probability_sum=0.,negative_probability_sum=0.)for k in models}
        def sink(batch):
            if batch.frame<=stats['rows'][sequence]['time_cut_frame']:return
            counts['known']+=len(batch.y);counts['positive']+=int(batch.y.sum());counts['negative']+=int((batch.y==0).sum())
            for kind,model in models.items():
                p=model.probabilities(batch.x);wrong=p[batch.y==0].copy()
                score_bytes[kind]+=wrong.nbytes
                assert score_bytes[kind]<=67108864,'Preserve calibration failure; do not drop wrong edges to fit memory'
                scores[kind].append(wrong)
                d=diagnostics[kind];d['Brier_sum']+=float(((p-batch.y)**2).sum())
                d['positive_probability_sum']+=float(p[batch.y==1].sum())
                d['negative_probability_sum']+=float(wrong.sum())
        reference=stream_video(sequence,q,speed,sink,progress=lambda f:progress('late_Train_calibration',sequence,f))
        assert reference==stats['full_teacher_references'][sequence]
        calibration[sequence]=dict(counts=counts,models=diagnostics)
    thresholds={}
    for kind in models:
        values=np.concatenate(scores[kind]);assert len(values)>0 and np.isfinite(values).all()
        thresholds[kind]=float(np.quantile(values,.99))
    private=dict(status='COMPLETE_EARLY_TRAIN_MODELS_LATE_TRAIN_CALIBRATION',UTC=utc(),
        models={k:m.to_dict()for k,m in models.items()},thresholds=thresholds,fit_history=history,
        calibration=calibration,source_registration_sha256=sha256(base/'fit_registered.json'),
        train_statistics_sha256=sha256(base/'Train_feature_statistics.json'),
        Train_frames_per_complete_pass=27251,complete_passes=8,optimizer_passes=6,
        Dev_rows=0,Test_rows=0,GPU_work=False,coefficient_files_public=False,
        encoder_identity_exposure='Train descriptors use encoders already trained on these Train identities; calibration is not a new heldout-encoder validation')
    require_fit();frozen_json(base/'private_models.json',private)
    frozen_json(base/'fit_complete.json',dict(status='COMPLETE_TWO_TRAIN_RISK_MODELS_NO_NEW_ACCURACY',UTC=utc(),
        fixed_seed=0,models=['RGB','paired'],model_file=str(base/'private_models.json'),
        model_sha256=sha256(base/'private_models.json'),thresholds=thresholds,
        Train_frames_per_complete_pass=27251,complete_passes=8,optimizer_passes=6,
        unique_Train_proposals=sum(v['proposal_rows']for v in stats['full_teacher_references'].values()),
        known_early_edges=stats['known_early_edges'],calibration=calibration,
        source_registration_sha256=sha256(base/'fit_registered.json'),seconds=time.monotonic()-started,
        all_proposals_and_frames_passed_to_original_teacher=True,early_weights_retained=True,
        feature_target_packets_exact_across_passes=True,teacher_MOT_exact_across_passes=True,
        Train_feature_pair_disk_dump=False,Dev_rows=0,Test_rows=0,new_GPU_work=False,
        new_tracking_accuracy=False,goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_TWO_TRAIN_RISK_MODELS_NO_NEW_ACCURACY',thresholds=thresholds,
        seconds=time.monotonic()-started)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['engineer','register-fit','fit']);args=p.parse_args()
    {'engineer':engineer,'register-fit':register_fit,'fit':fit}[args.mode]()


if __name__=='__main__':main()
