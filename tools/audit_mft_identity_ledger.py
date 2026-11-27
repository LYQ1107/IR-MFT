#!/usr/bin/env python3
"""Rebuild every Train packet; independently check ledger fit and quantiles."""
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
from tools.fit_mft_identity_ledger import registered,locations,utc,SPEC
from tools.mft_identity_ledger import LedgerLogistic,FEATURE_NAMES,IR_FEATURES
from tools.mft_identity_ledger_labels import LedgerCollector
from tools.mft_candidate_risk_data import load_arrays,load_GT
from tools.fit_mft_candidate_risk import threshold_values
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_stream_tracks import mot_line
from tools.native_pyramid_training_loop import guard_storage
from tools.run_mft_candidate_train_bank import atomic_json
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity


def sorted_quantile(values,q):
    a=np.sort(np.asarray(values,np.float64));assert len(a)>0 and np.isfinite(a).all()
    p=(len(a)-1)*q;lo=int(np.floor(p));hi=int(np.ceil(p))
    return float(a[lo]+(p-lo)*(a[hi]-a[lo]))


def audit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();plan,protocol,work,base,parent=locations();started=time.monotonic()
    e=read(base/'fit_r1.execution.json');assert e['actual_child_exit_code']==0 and e['source_unchanged']
    assert identity(e['identity']['PID'])is None
    fit=read(base/'fit_complete.json');model_hash=fit['model_sha256'];assert sha256(fit['model_file'])==model_hash
    private=read(fit['model_file']);models={k:LedgerLogistic.from_dict(v)for k,v in private['models'].items()}
    cov=read(base/'Train_coverage.json');reference=read(parent/'Train_feature_statistics.json')['full_teacher_references']
    source_hashes={**reg['source_hashes'],'tools/audit_mft_identity_ledger.py':sha256(Path(__file__))}
    files=[base/'fit_complete.json',base/'fit_r1.execution.json',base/'Train_coverage.json',SPEC]
    own=base/'independent_fit_audit_registered.json'
    assert not own.exists()
    frozen_json(own,dict(status='REGISTERED_INDEPENDENT_FULL_TRAIN_LEDGER_FIT_REBUILD',UTC=utc(),source_hashes=source_hashes,
        artifact_hashes={str(p):sha256(p)for p in files},model_sha256=model_hash,coefficients_updated=False,
        Dev_rows=0,Test_rows=0,historical_Test_previously_exposed=True))
    statistics={k:dict(loss=0.,weight=0.,rows=0,gradient=np.zeros(len(FEATURE_NAMES)+1),
        hessian=np.zeros((len(FEATURE_NAMES)+1,len(FEATURE_NAMES)+1)))for k in models}
    wrong={k:[]for k in models};right={k:[]for k in models};coverage={}
    n=0;total=np.zeros(len(FEATURE_NAMES));squares=total.copy();videos=fit['early_videos_used']
    for sequence,q in reg['qualified_Train'].items():
        arrays=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256'])
        expected=cov['rows'][sequence];counts={p:dict(known=0,positive=0,negative=0)for p in ['early','late']}
        packets={p:hashlib.sha256()for p in counts};cut=int(.75*q['frames'])
        def sink(s,f,x,y):
            nonlocal n,total,squares
            assert s==sequence
            part='early'if f<=cut else'late';c=counts[part];c['known']+=len(y);c['positive']+=int(y.sum());c['negative']+=int((y==0).sum())
            packets[part].update(np.asarray([f],np.int64).tobytes()+x.tobytes()+y.tobytes())
            if part=='early':
                n+=len(x);v=x.astype(np.float64);total+=v.sum(axis=0);squares+=(v*v).sum(axis=0)
            for k,m in models.items():
                # Independent logistic value/gradient/Hessian arithmetic, using
                # the frozen feature normalization but no optimizer helper.
                v=x.astype(np.float64).copy()
                if k=='RGB':v[:,IR_FEATURES]=0
                design=np.c_[np.clip((v-m.mean)/m.scale,-8,8),np.ones(len(v))]
                logits=design@m.coefficients;p=.5*(1+np.tanh(.5*logits))
                if part=='late':wrong[k].append(p[y==0]);right[k].append(p[y==1]);continue
                cc=expected['counts']['early'];classes=int(cc['positive']>0)+int(cc['negative']>0)
                weights=np.where(y>0,1/max(cc['positive'],1),1/max(cc['negative'],1))/(len(videos)*classes)
                a=statistics[k];a['rows']+=len(y);a['weight']+=float(weights.sum())
                a['loss']+=float(weights@(np.maximum(logits,0)+np.log1p(np.exp(-np.abs(logits)))-y*logits))
                a['gradient']+=design.T@(weights*(p-y))
                a['hessian']+=design.T@((weights*p*(1-p))[:,None]*design)
        collector=LedgerCollector(threshold_values(work)['Train_unknown_speed_upper'],sink);digest=hashlib.sha256()
        for i,f in enumerate(arrays['frame_ids']):
            frame=int(f);lo,hi=map(int,arrays['offsets'][i:i+2]);left,right=map(int,bounds[i:i+2])
            rows=collector.step(sequence,frame,arrays['detections'][lo:hi],arrays['rgb'][lo:hi],arrays['ir'][lo:hi],gt[left:right],split='train')
            for row in rows:digest.update(mot_line(row).encode())
            if frame%500==0 or frame==q['frames']:
                guard_storage(work);atomic_json(base/'independent_audit_progress.json',dict(sequence=sequence,frame=frame,frames=q['frames'],seconds=time.monotonic()-started))
        assert counts==expected['counts']
        assert {k:v.hexdigest()for k,v in packets.items()}==expected['feature_target_packet_sha256']
        assert digest.hexdigest()==expected['original_teacher_MOT_sha256']==reference[sequence]['original_teacher_MOT_sha256']
        assert collector.statistics==expected['collector_statistics']and collector.tracker.statistics==expected['ledger_statistics']
        assert sum(v is not None for v in collector.origins.labels.values())==expected['known_origin_roots']
        coverage[sequence]=dict(frames=q['frames'],proposals=len(arrays['detections']),counts=counts,
            original_teacher_MOT_sha256=digest.hexdigest(),feature_target_packet_sha256={k:v.hexdigest()for k,v in packets.items()},
            all_expected_counts_and_packets_exact=True)
        print(json.dumps(dict(sequence=sequence,frames=q['frames'],all_packets_exact=True)),flush=True)
    assert n==fit['early_known_edges'];mean=total/n;scale=np.maximum(np.sqrt(np.maximum(squares/n-mean*mean,0)),1e-4)
    np.testing.assert_allclose(models['paired'].mean,mean,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(models['paired'].scale,scale,rtol=1e-8,atol=1e-8)
    rgbmean=mean.copy();rgbscale=scale.copy();rgbmean[list(IR_FEATURES)]=0;rgbscale[list(IR_FEATURES)]=1
    np.testing.assert_allclose(models['RGB'].mean,rgbmean,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(models['RGB'].scale,rgbscale,rtol=1e-8,atol=1e-8)
    diagnostics={};thresholds={};spec=read(SPEC)
    for k,m in models.items():
        a=statistics[k];c=m.coefficients.copy();c[-1]=0;regularization=spec['fit']['L2']
        g=a['gradient']+regularization*c;h=a['hessian']+np.diag(np.r_[np.full(len(c)-1,regularization),0.])
        loss=a['loss']+.5*regularization*float(c@c)
        diagnostics[k]=dict(loss=loss,gradient_norm=float(np.linalg.norm(g)),
            next_undamped_Newton_step_norm=float(np.linalg.norm(np.linalg.solve(h+1e-8*np.eye(len(g)),g))))
        assert abs(a['weight']-1)<1e-7 and a['rows']==n
        for key,value in diagnostics[k].items():assert abs(value-fit['final_fit_diagnostics'][k][key])<1e-8,(k,key)
        bad=np.concatenate(wrong[k]);good=np.concatenate(right[k])
        thresholds[k]={name:sorted_quantile(bad,q)for name,q in spec['fit']['late_wrong_quantiles'].items()}
        thresholds[k]['veto']=sorted_quantile(good,.01)
        for q,value in thresholds[k].items():assert abs(value-fit['thresholds'][k][q])<2e-15,(k,q,value)
        assert len(bad)==fit['calibration'][k]['negative']and len(good)==fit['calibration'][k]['positive']
    assert sha256(fit['model_file'])==model_hash;registered()
    for n,h in source_hashes.items():assert sha256(ROOT/n)==h,n
    frozen_json(base/'independent_fit_audit.json',dict(status='COMPLETE_INDEPENDENT_FULL_TRAIN_LEDGER_FIT_REBUILD',UTC=utc(),
        all6_Train_replayed=True,Train_frames=27251,unique_Train_proposals=sum(r['proposals']for r in coverage.values()),
        coverage=coverage,all_feature_target_packets_and_teacher_MOT_exact=True,normalization_rebuilt=True,
        final_diagnostics_independently_rebuilt=diagnostics,late_thresholds_independently_sorted=thresholds,
        coefficient_and_normalization_bytes_unchanged=True,model_sha256=model_hash,new_optimizer_steps=0,
        Dev_rows=0,Test_rows=0,registration_sha256=sha256(own),seconds=time.monotonic()-started,
        not_a_fresh_blind_Test=True,goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_INDEPENDENT_FULL_TRAIN_LEDGER_FIT_REBUILD',thresholds=thresholds)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['audit']);p.parse_args();audit()
