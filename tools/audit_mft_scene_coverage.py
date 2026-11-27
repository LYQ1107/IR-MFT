#!/usr/bin/env python3
"""Independently recompute scalar diagnostics, not the visual producer or HOTA."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.native_pyramid_experiment import read,frozen_json
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity
from tools.fit_mft_proposal_quality import utc
from tools.native_pyramid_training_loop import guard_storage

W=Path('/var/tmp/liuyeqiang_ir_mft_perceptual_20261122')
B=W/'scene_coverage_diagnostic_20261127'
KINDS=['count','score','RGB','paired','duplicatedRGB','IR_roll','IR_missing']
THRESHOLDS=[.1,.25]


def derived(x,y):
    result={}
    for j,kind in enumerate(KINDS):
        a=x[:,j];target=y[:,-1];need=target>=.1;table={}
        for t in THRESHOLDS:
            gate=a>=t;tp=int(np.count_nonzero(gate&need));fp=int(np.count_nonzero(gate&~need));fn=int(np.count_nonzero(~gate&need));tn=int(np.count_nonzero(~gate&~need))
            table[str(t)]=dict(frames=len(a),selected_frames=tp+fp,needed_frames=tp+fn,TP=tp,FP=fp,FN=fn,TN=tn,
                precision=tp/(tp+fp)if tp+fp else None,recall=tp/(tp+fn)if tp+fn else None,
                selected_mean_target=float(target[gate].mean())if gate.any()else None,
                unselected_mean_target=float(target[~gate].mean())if (~gate).any()else None)
        result[kind]=dict(mean_deficit=float(a.mean()),mean_need_ratio=float(target.mean()),
            Pearson_with_need=float(np.corrcoef(a,target)[0,1])if len(a)>1 and a.std()>0 and target.std()>0 else None,thresholds=table)
    return result


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    r=read(B/'registered.json');f=read(B/'complete_Train_Dev.json');execution={}
    for name in ['engineering_r1','run_r1']:
        e=read(B/(name+'.execution.json'));assert e['actual_child_exit_code']==0 and e['source_unchanged']
        assert identity(e['identity']['PID'])is None
        for n,h in e['source_hashes'].items():assert sha256(ROOT/n)==h,n
        execution[name]=dict(identity=e['identity'],actual_exit=0,receipt_sha256=sha256(B/(name+'.execution.json')))
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in r['artifact_hashes'].items():assert sha256(n)==h,n
    additional=read(B/'additional_dependencies_registered.json')
    for n,h in additional['source_hashes'].items():assert sha256(ROOT/n)==h,n
    assert additional['registration_sha256']==sha256(B/'registered.json')==f['registration_sha256']
    s=f['private_scalar_table'];assert sha256(s['path'])==s['sha256']and Path(s['path']).stat().st_size==s['bytes']
    assert s['feature_columns']==KINDS
    with np.load(s['path'],allow_pickle=False)as z:arrays={k:z[k]for k in z.files}
    assert sum(a.nbytes for a in arrays.values())==s['payload_bytes']<16*2**20
    prior=read(W/'birth_only_20261127/complete_Train_diagnosis.json');checked=0;agreements={};summary={}
    for split in ['train','dev']:
        agreements[split]={};xs=[];ys=[]
        for sequence,row in f['rows'][split].items():
            x=arrays[split+'_'+sequence+'_features'];y=arrays[split+'_'+sequence+'_targets'];n=row['frames']
            assert x.shape==y.shape==(n,7)and x.dtype==y.dtype==np.float64
            assert np.isfinite(x).all()and np.isfinite(y).all()and ((x>=0)&(x<=1)).all()
            assert np.all(y[:,:-1]==y[:,:-1].astype(np.int64))and (y>=0).all()
            np.testing.assert_array_equal(y[:,1],y[:,2]+y[:,3]+y[:,4]);assert(y[:,5]<=y[:,2]).all()
            ratios=np.divide(y[:,5],y[:,0],out=np.zeros(n),where=y[:,0]>0);np.testing.assert_array_equal(ratios,y[:,-1])
            np.testing.assert_array_equal(x[:,2],x[:,6])
            assert derived(x,y)==row['full'];checked+=7*2*12
            cuts=[0,int(.5*n),int(.75*n),n]
            for phase,left,right in zip(['first50','middle25','last25'],cuts[:-1],cuts[1:]):
                assert derived(x[left:right],y[left:right])==row['phases'][phase];checked+=7*2*12
            for j,k in enumerate(s['target_columns'][:-1]):assert float(y[:,j].sum())==row['totals'][k]
            if split=='train':
                old=prior['coverage'][sequence];q=r['qualified'][split][sequence]
                assert row['inference_receipt']['original_zero_gate_MOT_sha256']==old['zero_gate_MOT_sha256']
                assert row['labels_sha256']==old['current_Train_label_packet_sha256']
                for phase,part in [('early',y[:int(.75*n)]),('late',y[int(.75*n):])]:
                    v=old['by_case'][phase]['BO_unconditional_c3'];summed=part.sum(axis=0)
                    assert summed[1]==v['opportunities']and summed[5]==v['known_positive_uncovered_by_zero_output_local_IoU05']
            agreements[split][sequence]={str(t):dict(paired_vs_RGB_different_frames=int(np.count_nonzero((x[:,3]>=t)!=(x[:,2]>=t))),
                paired_vs_IR_roll_different_frames=int(np.count_nonzero((x[:,3]>=t)!=(x[:,5]>=t))),
                paired_vs_count_different_frames=int(np.count_nonzero((x[:,3]>=t)!=(x[:,0]>=t))))for t in THRESHOLDS}
            xs.append(x);ys.append(y)
        assert sum(len(x)for x in xs)==(27251 if split=='train'else 4834)
        assert derived(np.concatenate(xs),np.concatenate(ys))==f['summaries'][split]['pooled']
        for kind in KINDS:
            for t in THRESHOLDS:
                values=[row['full'][kind]['thresholds'][str(t)]for row in f['rows'][split].values()]
                for key,v in f['summaries'][split]['equal_video'][kind][str(t)].items():
                    selected=[p[key]for p in values if p[key]is not None]
                    expected=float(np.mean(selected))if selected else None;assert v==expected
        summary[split]=f['summaries'][split]
    assert len(arrays)==16
    result=dict(status='COMPLETE_INDEPENDENT_SCALAR_COVERAGE_AUDIT',UTC=utc(),
        complete_report_sha256=sha256(B/'complete_Train_Dev.json'),registration_sha256=sha256(B/'registered.json'),
        private_scalar_sha256=s['sha256'],source_sha256=sha256(Path(__file__)),actual_producer_execution=execution,
        recomputed_threshold_fields_at_least=checked,all8_complete_sequence_tables_recomputed=True,
        all6_Train_existing_opportunity_counts_and_original_MOT_hashes_agree=True,
        target_count_conservation_exact=True,IR_missing_RGB_frame_deficits_exact=True,
        gate_disagreement_frames=agreements,budget=guard_storage(W),
        audit_scope='Recomputed scalar summaries, conservation, immutable-source and actual-exit provenance; not an independent replay of visual features, frame GT matching, tracker states or HOTA',
        Test_read=False,new_GPU_work=False,new_accuracy=False,goal_met=False,supported_innovations=0)
    frozen_json(B/'independent_scalar_audit.json',result)
    print(json.dumps(dict(status=result['status'],frames=32085,recomputed_fields=checked)),flush=True)


if __name__=='__main__':main()
