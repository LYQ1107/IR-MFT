#!/usr/bin/env python3
"""Rebuild all Train confirmation packets and independently verify quality fit."""
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
from tools.fit_mft_proposal_quality import registered,locations,utc,SPEC
from tools.mft_proposal_quality import (ForegroundView,QualityHead,ProposalConfirmation,
    geometry_features,current_features,FEATURE_NAMES,IR_FEATURES)
from tools.mft_candidate_risk_teacher import normalized_descriptors
from tools.mft_candidate_risk_data import load_arrays,load_GT
from tools.mft_candidate_risk_labels import current_detection_labels
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.run_mft_candidate_train_bank import atomic_json
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity


def sorted_quantile(a,q):
    a=np.sort(np.asarray(a,np.float64));assert len(a)>0
    position=(len(a)-1)*q;lo=int(np.floor(position));hi=int(np.ceil(position))
    return float(a[lo]+(position-lo)*(a[hi]-a[lo]))


def prepare(x,kind):
    a=x.astype(np.float64).copy()
    if kind=='RGB':a[:,[8,9,10]]=0
    elif kind=='duplicatedRGB':a[:,8]=a[:,7];a[:,9]=1;a[:,10]=0
    else:assert kind=='paired'
    return a


def audit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();plan,protocol,work,base=locations();started=time.monotonic()
    e=read(base/'fit_r1.execution.json');assert e['actual_child_exit_code']==0 and e['source_unchanged']
    assert identity(e['identity']['PID'])is None
    f=read(base/'fit_complete.json');assert f['gradient_converged']and sha256(f['model_file'])==f['model_sha256']
    p=read(f['model_file']);views={k:ForegroundView.from_dict(v)for k,v in p['views'].items()}
    heads={k:QualityHead.from_dict(v)for k,v in p['heads'].items()};old=read(base/'Train_coverage.json')['coverage']
    sources={**reg['source_hashes'],'tools/audit_mft_proposal_quality_fit.py':sha256(Path(__file__))}
    paths=[base/'fit_complete.json',base/'fit_r1.execution.json',base/'Train_coverage.json']
    target=base/'independent_fit_audit_registered.json';assert not target.exists()
    frozen_json(target,dict(status='REGISTERED_READ_ONLY_FULL_TRAIN_QUALITY_FIT_REBUILD',UTC=utc(),source_hashes=sources,
        artifact_hashes={str(v):sha256(v)for v in paths},model_sha256=f['model_sha256'],new_optimizer_steps=0,Dev_rows=0,Test_rows=0))
    view_stats={cue:dict(mean={0:np.zeros(512),1:np.zeros(512)},weight={0:0.,1:0.},within=np.zeros((512,512)))for cue in views}
    head_stats={k:dict(rows=0,weight=0.,loss=0.,gradient=np.zeros(len(FEATURE_NAMES)+1),
        hessian=np.zeros((len(FEATURE_NAMES)+1,len(FEATURE_NAMES)+1)),total=np.zeros(len(FEATURE_NAMES)),squares=np.zeros(len(FEATURE_NAMES)))for k in heads}
    scores={k:{condition:{'negative':[],'positive':[]}for condition in ['all','confirmed']}for k in heads};coverage={}
    maximum_raw_vs_fit_restore_feature_error=0.
    for sequence,q in reg['qualified_Train'].items():
        a=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256'])
        n=len(a['detections']);rgb=normalized_descriptors(a['rgb'],n);ir=normalized_descriptors(a['ir'],n)
        binary=np.empty(n,np.float32);known=np.empty(n,bool);early=np.empty(n,bool);streaks=np.empty(n,np.int32)
        all_x=np.empty((n,len(FEATURE_NAMES)),np.float32);confirmation=ProposalConfirmation();signature=hashlib.sha256()
        counts={phase:dict(proposals=0,known_positive=0,known_negative=0,unknown=0,weak_known_positive=0,weak_known_negative=0,
            confirmed_weak_positive=0,confirmed_weak_negative=0)for phase in ['early','late']}
        for i,fid in enumerate(a['frame_ids']):
            frame=int(fid);lo,hi=map(int,a['offsets'][i:i+2]);left,right=map(int,bounds[i:i+2]);d=a['detections'][lo:hi]
            labels=current_detection_labels(d,gt[left:right],frame=frame,split='train');g=geometry_features(d)
            s=confirmation.update(sequence,frame,d);isearly=frame<=int(.75*q['frames'])
            binary[lo:hi]=labels>0;known[lo:hi]=labels>=0;early[lo:hi]=isearly;streaks[lo:hi]=s
            # Reproduce the fit's explicitFP16restore/unit step followed by
            # the view interface's normalization;measure single-restore inference.
            all_x[lo:hi]=current_features(d,rgb[lo:hi],ir[lo:hi],views)
            if frame<=64:
                raw=current_features(d,a['rgb'][lo:hi],a['ir'][lo:hi],views)
                if len(raw):maximum_raw_vs_fit_restore_feature_error=max(maximum_raw_vs_fit_restore_feature_error,float(np.max(np.abs(raw-all_x[lo:hi]))))
            signature.update(np.asarray([frame,len(d)],np.int64).tobytes()+d.tobytes()+g.tobytes()+s.tobytes()+labels.tobytes())
            c=counts['early'if isearly else'late'];weak=(d[:,4]>.3)&(d[:,4]<=.7)
            for key,mask in [('proposals',np.ones(len(d),bool)),('known_positive',labels>0),('known_negative',labels==0),('unknown',labels<0),
                ('weak_known_positive',weak&(labels>0)),('weak_known_negative',weak&(labels==0)),
                ('confirmed_weak_positive',weak&(labels>0)&(s>=3)),('confirmed_weak_negative',weak&(labels==0)&(s>=3))]:c[key]+=int(mask.sum())
            if frame%1000==0 or frame==q['frames']:
                guard_storage(work);atomic_json(base/'independent_audit_progress.json',dict(sequence=sequence,frame=frame,frames=q['frames'],seconds=time.monotonic()-started))
        expected=old[sequence];assert counts==expected['counts']
        assert signature.hexdigest()==expected['GT_free_confirmation_and_feature_packet_sha256']
        # Match the frozen fit's full-video GEMV grouping exactly, while the
        # frame-wise interface discrepancy was measured separately above.
        score_RGB=views['RGB'].scores(rgb);score_IR=views['IR'].scores(ir)
        all_x[:,7]=score_RGB;all_x[:,8]=score_IR;all_x[:,9]=1;all_x[:,10]=abs(score_RGB-score_IR)
        y=binary[early&known];classes=[c for c in [0,1]if (y==c).any()]
        for cue,values in [('RGB',rgb),('IR',ir)]:
            v=view_stats[cue]
            for c in classes:
                x=values[early&known&(binary==c)].astype(np.float64);mu=x.mean(axis=0);w=1/(len(old)*len(classes))
                v['mean'][c]+=w*mu;v['weight'][c]+=w
                # Independent uncentered-second-moment identity rather than
                # the producer's residual matrix multiplication.
                v['within']+=w*((x.T@x)/len(x)-np.outer(mu,mu))
        for k,m in heads.items():
            x=prepare(all_x[early&known],k);stats=head_stats[k];stats['rows']+=len(x)
            stats['total']+=x.sum(axis=0);stats['squares']+=(x*x).sum(axis=0)
            for lo in range(0,len(x),4096):
                xx=x[lo:lo+4096];yy=y[lo:lo+4096];c=counts['early']
                weights=np.where(yy>0,1/max(c['known_positive'],1),1/max(c['known_negative'],1))/(len(old)*len(classes))
                design=np.c_[np.clip((xx-m.mean)/m.scale,-8,8),np.ones(len(xx))]
                z=design@m.coefficients;prob=.5*(1+np.tanh(z/2))
                stats['loss']+=float(weights@(np.maximum(z,0)+np.log1p(np.exp(-abs(z)))-yy*z))
                stats['gradient']+=design.T@(weights*(prob-yy));stats['hessian']+=design.T@((weights*prob*(1-prob))[:,None]*design)
                stats['weight']+=float(weights.sum())
            for condition in ['all','confirmed']:
                use=(~early)&known&(a['detections'][:,4]>.3)&(a['detections'][:,4]<=.7)
                if condition=='confirmed':use&=streaks>=3
                xx=prepare(all_x[use],k);design=np.c_[np.clip((xx-m.mean)/m.scale,-8,8),np.ones(len(xx))]
                prob=.5*(1+np.tanh((design@m.coefficients)/2));yy=binary[use]
                scores[k][condition]['negative'].append(prob[yy==0]);scores[k][condition]['positive'].append(prob[yy==1])
        coverage[sequence]=dict(frames=q['frames'],proposals=n,counts=counts,
            confirmation_geometry_label_packet_sha256=signature.hexdigest(),all_packets_and_counts_exact=True)
        print(json.dumps(dict(sequence=sequence,all_packets_and_counts_exact=True)),flush=True)
    spec=read(SPEC);view_diagnostics={}
    for cue,v in view_stats.items():
        mu0,mu1=v['mean'][0]/v['weight'][0],v['mean'][1]/v['weight'][1]
        ridge=max(spec['foreground_view']['ridge_fraction']*float(np.trace(v['within']))/512,1e-8)
        direction=np.linalg.solve(v['within']+ridge*np.eye(512),mu1-mu0);direction/=np.linalg.norm(direction)
        np.testing.assert_allclose((mu0+mu1)/2,views[cue].center,rtol=1e-9,atol=1e-9)
        np.testing.assert_allclose(direction,views[cue].direction,rtol=1e-6,atol=1e-7)
        view_diagnostics[cue]=dict(center_max_abs_error=float(np.max(abs((mu0+mu1)/2-views[cue].center))),
            direction_max_abs_error=float(np.max(abs(direction-views[cue].direction))),ridge=ridge,
            uncentered_second_moments_used=True,coefficients_updated=False)
    head_diagnostics={};thresholds={}
    for k,m in heads.items():
        s=head_stats[k];assert s['rows']==f['early_known_rows']and abs(s['weight']-1)<1e-7
        mean=s['total']/s['rows'];scale=np.maximum(np.sqrt(np.maximum(s['squares']/s['rows']-mean*mean,0)),1e-4)
        if k=='RGB':mean[list(IR_FEATURES)]=0;scale[list(IR_FEATURES)]=1
        np.testing.assert_allclose(mean,m.mean,rtol=1e-8,atol=1e-8);np.testing.assert_allclose(scale,m.scale,rtol=1e-7,atol=1e-8)
        coeff=m.coefficients.copy();coeff[-1]=0;regul=spec['quality_head']['L2'];gradient=s['gradient']+regul*coeff
        h=s['hessian']+np.diag(np.r_[np.full(len(coeff)-1,regul),0.]);loss=s['loss']+.5*regul*float(coeff@coeff)
        head_diagnostics[k]=dict(loss=loss,gradient_norm=float(np.linalg.norm(gradient)),
            next_Newton_step_norm=float(np.linalg.norm(np.linalg.solve(h+1e-8*np.eye(len(gradient)),gradient))))
        for field,value in head_diagnostics[k].items():assert abs(value-f['final_diagnostics'][k][field])<1e-8,(k,field)
        thresholds[k]={name:{}for name in spec['calibration']['wrong_quantiles']}
        for condition,buckets in scores[k].items():
            bad=np.concatenate(buckets['negative']);good=np.concatenate(buckets['positive'])
            assert len(bad)==f['late_calibration'][k][condition]['known_negative']and len(good)==f['late_calibration'][k][condition]['known_positive']
            for name,q in spec['calibration']['wrong_quantiles'].items():
                value=sorted_quantile(bad,q)if len(bad)else 1.;thresholds[k][name][condition]=value
                assert abs(value-f['thresholds'][k][name][condition])<2e-12,(k,name,condition)
    assert maximum_raw_vs_fit_restore_feature_error<1e-5
    assert sha256(f['model_file'])==f['model_sha256'];registered()
    for n,h in sources.items():assert sha256(ROOT/n)==h,n
    frozen_json(base/'independent_fit_audit.json',dict(status='COMPLETE_READ_ONLY_FULL_TRAIN_QUALITY_FIT_REBUILD',UTC=utc(),
        all6_Train_replayed=True,Train_frames=27251,proposals=sum(r['proposals']for r in coverage.values()),coverage=coverage,
        all_confirmation_geometry_label_packets_exact=True,view_diagnostics=view_diagnostics,head_diagnostics=head_diagnostics,
        independently_sorted_late_thresholds=thresholds,coefficient_and_normalization_bytes_unchanged=True,
        model_sha256=f['model_sha256'],max_single_vs_double_FP32_restore_feature_error=maximum_raw_vs_fit_restore_feature_error,
        fit_and_inference_normalization_roundoff_disclosed=True,Dev_rows=0,Test_rows=0,new_optimizer_steps=0,new_GPU_work=False,
        registration_sha256=sha256(target),seconds=time.monotonic()-started,goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_READ_ONLY_FULL_TRAIN_QUALITY_FIT_REBUILD',thresholds=thresholds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['audit']);p.parse_args();audit()
