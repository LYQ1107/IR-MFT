#!/usr/bin/env python3
"""Rebuild all Train archive supervision and independently check the final fit."""
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
from tools.fit_mft_archive_reasoning import locations,registered as fit_registered
from tools.fit_mft_candidate_risk import threshold_values
from tools.mft_archive_reasoning import ArchiveTrainCollector,ArchiveLogistic,FEATURE_NAMES,IR_FEATURES
from tools.mft_candidate_risk_data import load_arrays,load_GT
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.native_pyramid_stream_tracks import mot_line
from tools.run_mft_native_memory_priority_v3 import identity
from tools.run_mft_candidate_train_bank import atomic_json
from tools.run_pseudo_ir_comparison import sha256


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def batch_diagnostic(model,values,targets,weights):
    """Separate numerical path: no NewtonPass call and no coefficient update."""
    a=np.asarray(values,np.float64).copy();y=np.asarray(targets,np.float64);w=np.asarray(weights,np.float64)
    assert a.ndim==2 and a.shape[1]==len(FEATURE_NAMES)and y.shape==w.shape==(len(a),)
    assert np.logical_or(y==0,y==1).all()and np.isfinite(w).all()and(w>=0).all()
    if model.kind=='RGB':a[:,IR_FEATURES]=0
    z=np.column_stack([np.clip((a-model.mean)/model.scale,-8,8),np.ones(len(a))])
    logits=z@model.coefficients;p=np.exp(-np.logaddexp(0,-logits))
    np.testing.assert_allclose(p,model.probabilities(values),rtol=2e-15,atol=2e-16)
    return float(w@(np.logaddexp(0,logits)-y*logits)),z.T@(w*(p-y)),float(w.sum()),p


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    parent=fit_registered();plan,protocol,work,base,risk=locations()
    fit=read(base/'fit_complete.json');e=read(base/'fit_r1.execution.json')
    assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(e['identity']['PID'])is None
    assert sha256(fit['model_file'])==fit['model_sha256']
    proof=read(base/'independent_audit_CPU_regressions_r1.json')
    assert proof['actual_tool_exit_code']==0 and proof['tests_passed']==2
    for n,h in proof['source_hashes'].items():assert sha256(ROOT/n)==h,n
    paths=[base/'fit_registered.json',base/'fit_complete.json',base/'Train_archive_coverage.json',
        base/'fit_r1.execution.json',base/'independent_audit_CPU_regressions_r1.json']
    value=dict(status='REGISTERED_READ_ONLY_ALL6_TRAIN_ARCHIVE_FIT_AUDIT',UTC=utc(),
        source_hashes=proof['source_hashes'],artifact_hashes={str(p):sha256(p)for p in paths},
        model_sha256=fit['model_sha256'],new_optimizer_steps=0,new_models=0,new_GPU_work=False,
        Train_frames=27251,Dev_rows=0,Test_rows=0,no_feature_pair_disk_dump=True)
    assert not(base/'independent_audit_registered.json').exists();frozen_json(base/'independent_audit_registered.json',value)
    print(json.dumps(dict(status=value['status'],sources=len(value['source_hashes']))),flush=True)


def registered():
    fit_registered();plan,protocol,work,base,risk=locations();reg=read(base/'independent_audit_registered.json')
    for n,h in reg['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in reg['artifact_hashes'].items():assert sha256(n)==h,n
    return reg


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();preg=fit_registered();plan,protocol,work,base,risk=locations();started=time.monotonic()
    fit=read(base/'fit_complete.json');private=read(fit['model_file']);coverage=read(base/'Train_archive_coverage.json')
    assert sha256(fit['model_file'])==reg['model_sha256']==fit['model_sha256']
    models={k:ArchiveLogistic.from_dict(v)for k,v in private['models'].items()}
    coefficients={k:m.coefficients.copy()for k,m in models.items()}
    early_videos=fit['early_videos_used'];count=0;total=np.zeros(26,np.float64);squares=total.copy()
    losses={k:0. for k in models};gradients={k:np.zeros(27,np.float64)for k in models}
    weights={k:0. for k in models};wrong={k:[]for k in models};rows={};late_counts=dict(known=0,positive=0,negative=0)
    for sequence,q in preg['qualified_Train'].items():
        arrays=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256'])
        source=coverage['rows'][sequence];cut=source['early_cut'];counts={p:dict(known=0,positive=0,negative=0)for p in ['early','late']}
        digests={p:hashlib.sha256()for p in ['early','late']}
        def sink(batch):
            nonlocal count,total,squares
            part='early'if batch.frame<=cut else'late';c=counts[part]
            c['known']+=len(batch.y);c['positive']+=int(batch.y.sum());c['negative']+=int((batch.y==0).sum())
            digests[part].update(np.asarray([batch.frame],np.int64).tobytes()+batch.x.tobytes()+batch.y.tobytes())
            if part=='early':
                count+=len(batch.x);a=batch.x.astype(np.float64);total+=a.sum(axis=0);squares+=(a*a).sum(axis=0)
                ref=source['counts']['early'];positive,negative=ref['positive'],ref['negative']
                classes=int(positive>0)+int(negative>0);assert sequence in early_videos and classes>0
                w=np.where(batch.y==1,1/max(positive,1),1/max(negative,1))/(len(early_videos)*classes)
                for k,m in models.items():
                    loss,gradient,weight,p=batch_diagnostic(m,batch.x,batch.y,w)
                    losses[k]+=loss;gradients[k]+=gradient;weights[k]+=weight
            else:
                for k,m in models.items():wrong[k].append(m.probabilities(batch.x)[batch.y==0].copy())
        collector=ArchiveTrainCollector(threshold_values(work)['Train_unknown_speed_upper'],sink);digest=hashlib.sha256()
        for i,f in enumerate(arrays['frame_ids']):
            frame=int(f);lo,hi=map(int,arrays['offsets'][i:i+2]);left,right=map(int,bounds[i:i+2])
            output=collector.step(sequence,frame,arrays['detections'][lo:hi],arrays['rgb'][lo:hi],
                arrays['ir'][lo:hi],gt[left:right],split='train')
            for row in output:digest.update(mot_line(row).encode())
            if frame%500==0 or frame==q['frames']:
                guard_storage(work);atomic_json(base/'independent_audit_progress.json',dict(status='REBUILDING_ALL6_TRAIN_ARCHIVE_AUDIT',
                    UTC=utc(),sequence=sequence,frame=frame,frames=q['frames'],seconds=time.monotonic()-started,new_optimizer_steps=0))
        assert counts==source['counts']and collector.statistics==source['collector_statistics']
        assert collector.linker.statistics==source['archive_statistics']
        assert digest.hexdigest()==source['original_teacher_MOT_sha256']
        assert {p:d.hexdigest()for p,d in digests.items()}==source['feature_target_packet_sha256']
        for k in late_counts:late_counts[k]+=counts['late'][k]
        rows[sequence]=dict(frames=q['frames'],proposals=len(arrays['detections']),counts=counts,
            teacher_MOT_sha256=digest.hexdigest(),feature_target_packets_exact=True,original_teacher_MOT_exact=True)
    assert count==fit['early_known_edges']==12951
    mean=total/count;scale=np.maximum(np.sqrt(np.maximum(squares/count-mean**2,0)),1e-4)
    results={}
    for k,m in models.items():
        expected_mean=mean.copy();expected_scale=scale.copy()
        if k=='RGB':expected_mean[list(IR_FEATURES)]=0;expected_scale[list(IR_FEATURES)]=1
        np.testing.assert_allclose(m.mean,expected_mean,rtol=0,atol=2e-12)
        np.testing.assert_allclose(m.scale,expected_scale,rtol=0,atol=2e-12)
        c=m.coefficients.copy();c[-1]=0;loss=losses[k]+.0005*float(c@c);gradient=gradients[k]+.001*c
        original=fit['final_fit_diagnostics'][k]
        np.testing.assert_allclose([loss,np.linalg.norm(gradient),weights[k]],
            [original['weighted_objective_after_final_step'],original['gradient_norm'],original['weight']],rtol=1e-10,atol=2e-12)
        a=np.concatenate(wrong[k]);thresholds=dict(q99=float(np.quantile(a,.99)),q999=float(np.quantile(a,.999)))
        assert thresholds==fit['thresholds'][k]and late_counts==fit['late_calibration_counts'][k]
        np.testing.assert_array_equal(m.coefficients,coefficients[k])
        results[k]=dict(weighted_objective_after_final_step=loss,gradient_norm=float(np.linalg.norm(gradient)),
            weight=weights[k],thresholds=thresholds,late_calibration_counts=late_counts,
            normalization_rebuilt_and_exact_within_2e_minus12=True,coefficients_unchanged=True,
            independent_numerical_path=True)
    assert not models['RGB'].coefficients[list(IR_FEATURES)].any()
    assert sha256(fit['model_file'])==reg['model_sha256'];registered()
    frozen_json(base/'independent_Train_fit_audit.json',dict(status='COMPLETE_INDEPENDENT_ALL6_TRAIN_ARCHIVE_FIT_AUDIT',
        UTC=utc(),coverage=rows,models=results,Train_frames=27251,early_known_edges=count,
        model_sha256=reg['model_sha256'],registration_sha256=sha256(base/'independent_audit_registered.json'),
        new_optimizer_steps=0,new_models=0,model_bytes_unchanged=True,normalization_and_coefficients_published=False,
        feature_pair_disk_dump=False,Dev_rows=0,Test_rows=0,new_GPU_work=False,seconds=time.monotonic()-started,
        gradient_convergence_is_not_tracking_accuracy=True,new_accuracy=False,goal_met=False,supported_innovations=0))
    atomic_json(base/'independent_audit_progress.json',dict(status='COMPLETE_INDEPENDENT_ALL6_TRAIN_ARCHIVE_FIT_AUDIT',UTC=utc()))
    print(json.dumps(dict(status='COMPLETE_INDEPENDENT_ALL6_TRAIN_ARCHIVE_FIT_AUDIT',Train_frames=27251,
        gradient_norms={k:v['gradient_norm']for k,v in results.items()})),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','run']);a=p.parse_args()
    register()if a.mode=='register'else run()


if __name__=='__main__':main()
