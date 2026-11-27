#!/usr/bin/env python3
"""Full-Train origin-clean ledger supervision; deterministic bounded-RAM fit."""
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
from tools.mft_identity_ledger import LedgerTracker,LedgerLogistic,FEATURE_NAMES,IR_FEATURES
from tools.mft_identity_ledger_labels import LedgerCollector
from tools.mft_candidate_risk_data import qualify_Train,load_arrays,load_GT
from tools.mft_candidate_risk_model import NewtonPass,class_video_weights
from tools.fit_mft_candidate_risk import locations as parent_locations,threshold_values
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_stream_tracks import IndependentByteTrack,mot_line
from tools.native_pyramid_training_loop import guard_storage
from tools.run_mft_candidate_train_bank import atomic_json
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity

SPEC=ROOT/'configs/perceptual_identity_ledger_20261127.json'
SOURCES=['tools/mft_identity_ledger.py','tools/mft_identity_ledger_labels.py',
    'tools/fit_mft_identity_ledger.py','tools/supervise_mft_identity_ledger.py',
    'tests/test_mft_identity_ledger.py',str(SPEC.relative_to(ROOT))]


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def locations():
    plan,protocol,work,parent=parent_locations()
    return plan,protocol,work,work/'identity_ledger_20261127',parent


def sources():
    plan,protocol,work,base,parent=locations()
    old=read(work/'anchor_exchange_20261127/registered.json')['source_hashes']
    for n,h in old.items():assert sha256(ROOT/n)==h,n
    return {**old,**{n:sha256(ROOT/n)for n in SOURCES}}


def engineer():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan,protocol,work,base,parent=locations();frozen=sources();qualified=qualify_Train(work,protocol)
    speed=threshold_values(work)['Train_unknown_speed_upper'];result={}
    for sequence,q in qualified.items():
        arrays=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256'])
        def replay(future=False,rename=False):
            packets=[];graph=[]
            collector=LedgerCollector(speed,lambda s,f,x,y:packets.append((f,hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest())))
            original=IndependentByteTrack();rows=[]
            for i in range(64):
                frame=i+1;lo,hi=map(int,arrays['offsets'][i:i+2]);left,right=map(int,bounds[i:i+2])
                d=arrays['detections'][lo:hi].copy();rgb=arrays['rgb'][lo:hi].copy();ir=arrays['ir'][lo:hi].copy();g=gt[left:right].copy()
                if future and frame>32:d[:,:4]+=777;rgb*=-1;ir*=-1
                if rename:g[:,1]+=100000
                text=''.join(map(mot_line,collector.step(sequence,frame,d,rgb,ir,g,split='train')))
                assert text==''.join(map(mot_line,original.update(d,frame)))
                rows.append(text)
                m=collector.tracker.ledger
                graph.append(hashlib.sha256(b''.join(a.RGB.tobytes()+a.box.tobytes()for _,a in sorted(m.memories.items())if a.ready)).hexdigest())
            return rows,graph,packets,collector.statistics
        normal=replay();future=replay(future=True);renamed=replay(rename=True)
        assert normal[:3]==renamed[:3]
        assert normal[0][:32]==future[0][:32]and normal[1][:32]==future[1][:32]
        assert [v for v in normal[2]if v[0]<=32]==[v for v in future[2]if v[0]<=32]
        result[sequence]=dict(frames=64,prefix_frames=32,original_MOT_exact=True,
            GT_renaming_features_targets_and_MOT_exact=True,future_suffix_prefix_exact=True,
            statistics=normal[3],bank_sha256=q['bank_sha256'],GT_sha256=q['GT_sha256'])
        guard_storage(work)
    assert sources()==frozen
    frozen_json(base/'real_Train_engineering.json',dict(status='COMPLETE_ALL6_REAL_TRAIN_LEDGER_ENGINEERING',
        UTC=utc(),source_hashes=frozen,qualified_Train=qualified,rows=result,sequences=6,frames=384,
        no_new_models=True,new_accuracy=False))
    print(json.dumps(dict(status='COMPLETE_ALL6_REAL_TRAIN_LEDGER_ENGINEERING',frames=384)),flush=True)


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan,protocol,work,base,parent=locations();s=sources();proof=read(base/'CPU_regressions_r2.json')
    assert proof['actual_tool_exit_code']==0 and proof['tests_passed']==6
    for n,h in proof['source_hashes'].items():assert sha256(ROOT/n)==h,n
    e=read(base/'engineering_r1.execution.json');p=read(base/'real_Train_engineering.json')
    assert e['actual_child_exit_code']==0 and e['source_unchanged']and identity(e['identity']['PID'])is None
    assert p['source_hashes']==s and p['sequences']==6 and p['frames']==384
    q=qualify_Train(work,protocol);assert q==p['qualified_Train'];spec=read(SPEC)
    files=[SPEC,base/'CPU_regressions_r2.json',base/'real_Train_engineering.json',base/'engineering_r1.execution.json',
        work/'anchor_exchange_20261127/complete_Train.json',parent/'Train_feature_statistics.json']
    v=dict(status='REGISTERED_FULL_TRAIN_IDENTITY_LEDGER_BEFORE_COLLECTION',UTC=utc(),source_hashes=s,
        artifact_hashes={str(p):sha256(p)for p in files},qualified_Train=q,feature_names=list(FEATURE_NAMES),
        specification_sha256=sha256(SPEC),Train_frames=27251,Dev_rows=0,Test_rows=0,seed=0,
        maximum_RAM_feature_payload_bytes=spec['fit']['maximum_RAM_feature_payload_bytes'],
        truncate_pairs=False,pair_disk_dump=False,teacher_costs_unchanged=True,
        original_short_gap_or_archive_coefficients_reused=False,historical_Test_previously_exposed=True,
        goal_met=False,supported_innovations=0)
    assert not(base/'fit_registered.json').exists();frozen_json(base/'fit_registered.json',v)
    print(json.dumps(dict(status=v['status'],sources=len(s),Train_frames=27251)),flush=True)


def registered():
    plan,protocol,work,base,parent=locations();r=read(base/'fit_registered.json')
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    for n,h in r['artifact_hashes'].items():assert sha256(n)==h,n
    return r


def fit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();plan,protocol,work,base,parent=locations();started=time.monotonic();spec=read(SPEC)
    assert not(base/'fit_complete.json').exists()and not(base/'private_models.json').exists()
    speed=threshold_values(work)['Train_unknown_speed_upper']
    reference=read(parent/'Train_feature_statistics.json')['full_teacher_references']
    coverage={};cache={};payload=0
    for sequence,q in reg['qualified_Train'].items():
        arrays=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256'])
        batches={phase:[]for phase in ['early','late']};digests={k:hashlib.sha256()for k in batches}
        counts={k:dict(known=0,positive=0,negative=0)for k in batches};cut=int(.75*q['frames'])
        def sink(s,f,x,y):
            nonlocal payload
            assert s==sequence
            payload+=x.nbytes+y.nbytes
            assert payload<=reg['maximum_RAM_feature_payload_bytes'],'Preserve failure; do not truncate any candidate'
            part='early'if f<=cut else'late';batches[part].append((x,y))
            c=counts[part];c['known']+=len(y);c['positive']+=int(y.sum());c['negative']+=int((y==0).sum())
            digests[part].update(np.asarray([f],np.int64).tobytes()+x.tobytes()+y.tobytes())
        collector=LedgerCollector(speed,sink);digest=hashlib.sha256()
        for i,f in enumerate(arrays['frame_ids']):
            frame=int(f);lo,hi=map(int,arrays['offsets'][i:i+2]);left,right=map(int,bounds[i:i+2])
            rows=collector.step(sequence,frame,arrays['detections'][lo:hi],arrays['rgb'][lo:hi],arrays['ir'][lo:hi],gt[left:right],split='train')
            for row in rows:digest.update(mot_line(row).encode())
            if frame%500==0 or frame==q['frames']:
                guard_storage(work);atomic_json(base/'progress.json',dict(phase='FULL_TRAIN_LEDGER_ORIGIN_CANDIDATES',UTC=utc(),
                    sequence=sequence,frame=frame,frames=q['frames'],RAM_payload_bytes=payload,seconds=time.monotonic()-started))
        assert digest.hexdigest()==reference[sequence]['original_teacher_MOT_sha256']
        assert collector.statistics['frames']==q['frames']and collector.statistics['proposals']==reference[sequence]['proposal_rows']
        cache[sequence]={k:(np.concatenate([b[0]for b in rows]),np.concatenate([b[1]for b in rows]))
            if rows else(np.empty((0,len(FEATURE_NAMES)),np.float32),np.empty(0,np.float32))for k,rows in batches.items()}
        coverage[sequence]=dict(frames=q['frames'],proposals=len(arrays['detections']),counts=counts,early_cut=cut,
            collector_statistics=collector.statistics,ledger_statistics=collector.tracker.statistics,
            known_origin_roots=sum(v is not None for v in collector.origins.labels.values()),
            unknown_or_mixed_origin_roots=sum(v is None for v in collector.origins.labels.values()),
            original_teacher_MOT_exact=True,original_teacher_MOT_sha256=digest.hexdigest(),
            feature_target_packet_sha256={k:d.hexdigest()for k,d in digests.items()},bank_sha256=q['bank_sha256'],GT_sha256=q['GT_sha256'])
        del batches,arrays,gt
        print(json.dumps(dict(sequence=sequence,counts=counts,RAM_payload_bytes=payload)),flush=True)
    frozen_json(base/'Train_coverage.json',dict(status='COMPLETE_ALL27251_TRAIN_LEDGER_CANDIDATES',UTC=utc(),rows=coverage,
        all_teacher_MOT_exact=True,unique_proposals=sum(r['proposals']for r in coverage.values()),Train_frames=27251,
        RAM_feature_payload_bytes=payload,pair_disk_dump=False,truncated_pairs=0,Dev_rows=0,Test_rows=0,
        registration_sha256=sha256(base/'fit_registered.json')))
    videos=[s for s in cache if len(cache[s]['early'][1])]
    count=sum(len(cache[s]['early'][1])for s in videos);total=np.zeros(len(FEATURE_NAMES));squares=total.copy()
    for s in videos:
        x=cache[s]['early'][0]
        for lo in range(0,len(x),4096):
            v=x[lo:lo+4096].astype(np.float64);total+=v.sum(axis=0);squares+=(v*v).sum(axis=0)
    mean=total/count;scale=np.maximum(np.sqrt(np.maximum(squares/count-mean*mean,0)),1e-4)
    models={k:LedgerLogistic(mean,scale,k)for k in ['RGB','paired']};history=[]
    def accumulate(m):
        acc=NewtonPass(m,regularization=spec['fit']['L2'])
        for s in videos:
            x,y=cache[s]['early'];c=coverage[s]['counts']['early']
            for lo in range(0,len(y),4096):
                yy=y[lo:lo+4096];acc.add(x[lo:lo+4096],yy,class_video_weights(yy,c['positive'],c['negative'],len(videos)))
        return acc
    for index in range(spec['fit']['fixed_Newton_updates']):
        row=dict(pass_index=index+1,models={k:accumulate(m).step()for k,m in models.items()});history.append(row)
        atomic_json(base/'fit_history.json',dict(rows=history));print(json.dumps(row),flush=True)
    final={}
    for k,m in models.items():
        loss,g,h=accumulate(m).objective();final[k]=dict(loss=loss,gradient_norm=float(np.linalg.norm(g)),
            next_undamped_Newton_step_norm=float(np.linalg.norm(np.linalg.solve(h+1e-8*np.eye(len(g)),g))))
    calibration={};thresholds={}
    for k,m in models.items():
        wrong=[];right=[]
        for s in cache:
            x,y=cache[s]['late'];p=m.probabilities(x)
            wrong.append(p[y==0]);right.append(p[y==1])
        bad=np.concatenate(wrong);good=np.concatenate(right);assert len(bad)>0 and len(good)>0
        thresholds[k]={name:float(np.quantile(bad,q))for name,q in spec['fit']['late_wrong_quantiles'].items()}
        thresholds[k]['veto']=float(np.quantile(good,.01))
        calibration[k]=dict(negative=len(bad),positive=len(good),negative_max=float(bad.max()),positive_median=float(np.median(good)),
            positive_passing={q:int((good>t).sum())for q,t in thresholds[k].items()})
    assert not models['RGB'].coefficients[list(IR_FEATURES)].any()
    private=dict(status='PRIVATE_LEDGER_HEADS_NOT_FOR_GIT',UTC=utc(),models={k:m.to_dict()for k,m in models.items()},
        thresholds=thresholds,history=history,final_diagnostics=final,registration_sha256=sha256(base/'fit_registered.json'),
        coverage_sha256=sha256(base/'Train_coverage.json'),seed=0,Train_only=True)
    registered();guard_storage(work);frozen_json(base/'private_models.json',private)
    frozen_json(base/'fit_complete.json',dict(status='COMPLETE_ALL6_TRAIN_ORIGIN_CLEAN_LEDGER_FIT',UTC=utc(),
        model_file=str(base/'private_models.json'),model_sha256=sha256(base/'private_models.json'),thresholds=thresholds,
        final_fit_diagnostics=final,calibration=calibration,history=history,Train_frames=27251,Train_sequences=6,
        early_videos_used=videos,early_known_edges=count,unique_Train_proposals=sum(r['proposals']for r in coverage.values()),
        RAM_feature_payload_bytes=payload,all_teacher_MOT_exact=True,truncated_pairs=0,pair_disk_dump=False,
        registration_sha256=sha256(base/'fit_registered.json'),coverage_sha256=sha256(base/'Train_coverage.json'),
        Dev_rows=0,Test_rows=0,seed=0,new_GPU_work=False,new_accuracy=False,seconds=time.monotonic()-started,
        goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_ALL6_TRAIN_ORIGIN_CLEAN_LEDGER_FIT',thresholds=thresholds,final=final)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['engineer','register','fit'])
    {'engineer':engineer,'register':register,'fit':fit}[p.parse_args().mode]()


if __name__=='__main__':main()
