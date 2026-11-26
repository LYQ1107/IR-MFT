#!/usr/bin/env python3
"""Train-only causal archive supervision and bounded-RAM logistic fitting."""
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
from tools.mft_archive_reasoning import (ArchiveLinker,ArchiveTrainCollector,ArchiveLogistic,
    FEATURE_NAMES,IR_FEATURES)
from tools.mft_candidate_risk_teacher import CandidateTeacher
from tools.mft_candidate_risk_data import qualify_Train,load_arrays,load_GT
from tools.mft_candidate_risk_model import NewtonPass,class_video_weights
from tools.fit_mft_candidate_risk import locations as parent_locations,threshold_values
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_stream_tracks import IndependentByteTrack,mot_line
from tools.native_pyramid_training_loop import guard_storage
from tools.run_mft_candidate_train_bank import atomic_json
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_dev_candidate_graph_diagnosis import registered as diagnosis_registered
from tools.run_mft_native_memory_priority_v3 import identity

SPEC=ROOT/'configs/perceptual_archive_reasoning_20261127.json'
SOURCES=['tools/mft_archive_reasoning.py','tools/fit_mft_archive_reasoning.py',
    'tests/test_mft_archive_reasoning.py',str(SPEC.relative_to(ROOT))]


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def locations():
    plan,protocol,work,parent=parent_locations()
    return plan,protocol,work,work/'archive_reasoning_20261127',parent


def current_sources():
    parent=diagnosis_registered();result=dict(parent['source_hashes'])
    result.update({name:sha256(ROOT/name)for name in SOURCES});return result


def require_sources(sources):
    for name,digest in sources.items():assert sha256(ROOT/name)==digest,name


def engineer():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan,protocol,work,base,parent=locations();sources=current_sources()
    proof=read(base/'CPU_regressions_r1.json')
    assert proof['actual_tool_exit_code']==0 and proof['tests_passed']==5
    for name,digest in proof['source_hashes'].items():assert sources[name]==digest
    qualified=qualify_Train(work,protocol);speed=threshold_values(work)['Train_unknown_speed_upper'];results={}
    for sequence,q in qualified.items():
        arrays=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);count=min(128,q['frames']);cut=min(64,count)
        def replay(changed=False):
            graph=[];t=ArchiveLinker(CandidateTeacher(speed),speed,observer=lambda x,e,*a:
                graph.append((a[-1],hashlib.sha256(x.tobytes()+e.tobytes()).hexdigest())))
            expected=IndependentByteTrack();rows=[]
            for i in range(count):
                lo,hi=map(int,arrays['offsets'][i:i+2]);f=i+1
                d=arrays['detections'][lo:hi].copy();rgb=arrays['rgb'][lo:hi].copy();ir=arrays['ir'][lo:hi].copy()
                if changed and f>cut:d[:,:4]+=777;rgb*=-1;ir*=-1
                text=''.join(map(mot_line,t.update(sequence,f,d,rgb,ir)))
                assert text==''.join(map(mot_line,expected.update(d,f)));rows.append(text)
            return rows,graph,t.statistics
        normal=replay();other=replay(True)
        assert normal[0][:cut]==other[0][:cut]
        assert [v for v in normal[1]if v[0]<=cut]==[v for v in other[1]if v[0]<=cut]
        results[sequence]=dict(frames=count,prefix_frames=cut,original_MOT_exact=True,
            future_suffix_corruption_prefix_and_features_exact=True,statistics=normal[2],
            bank_sha256=q['bank_sha256'],MOT_sha256=hashlib.sha256(''.join(normal[0]).encode()).hexdigest())
        guard_storage(work)
    require_sources(sources)
    frozen_json(base/'CPU_real_Train_engineering.json',dict(status='COMPLETE_ALL6_REAL_TRAIN_ARCHIVE_ENGINEERING',
        UTC=utc(),source_hashes=sources,qualified_Train=qualified,rows=results,sequences=6,
        frames=sum(v['frames']for v in results.values()),no_model_fit=True,no_new_accuracy=True,new_GPU_work=False))
    print(json.dumps(dict(status='COMPLETE_ALL6_REAL_TRAIN_ARCHIVE_ENGINEERING',frames=sum(v['frames']for v in results.values()))),flush=True)


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan,protocol,work,base,parent=locations();spec=read(SPEC);sources=current_sources()
    engineer=read(base/'CPU_real_Train_engineering.json');execution=read(base/'engineering_r1.execution.json')
    assert execution['actual_child_exit_code']==0 and execution['source_unchanged']
    assert identity(execution['identity']['PID'])is None and engineer['source_hashes']==sources
    assert engineer['sequences']==6 and engineer['frames']==768
    q=qualify_Train(work,protocol);assert q==engineer['qualified_Train']
    assert spec['feature_dimension']==len(FEATURE_NAMES)==26 and spec['training']['videos']==6
    paths=[base/'CPU_regressions_r1.json',base/'CPU_regressions_r1.stdout.txt',
        base/'CPU_real_Train_engineering.json',base/'engineering_r1.execution.json',SPEC,
        work/'dev_candidate_graph_diagnosis_20261127/complete.json',parent/'Train_feature_statistics.json']
    value=dict(status='REGISTERED_ALL6_TRAIN_ARCHIVE_FITTING_BEFORE_FEATURE_COLLECTION',UTC=utc(),
        source_hashes=sources,artifact_hashes={str(p):sha256(p)for p in paths},
        specification_sha256=sha256(SPEC),qualified_Train=q,feature_names=list(FEATURE_NAMES),
        Train_frames=27251,Dev_rows=0,Test_rows=0,random_seed=0,additional_GPU_work=False,
        memory_payload_cap_bytes=spec['training']['RAM_feature_payload_cap_bytes'],
        feature_pair_disk_dump=False,truncate_pairs=False,source_teacher='Unchanged original ByteTrack on all proposals',
        current_short_gap_coefficients_reused=False,historical_Test_previously_exposed=True,
        goal_met=False,supported_innovations=0)
    assert not(base/'fit_registered.json').exists();frozen_json(base/'fit_registered.json',value)
    print(json.dumps(dict(status=value['status'],sources=len(sources),Train_frames=27251)),flush=True)


def registered():
    plan,protocol,work,base,parent=locations();reg=read(base/'fit_registered.json')
    require_sources(reg['source_hashes'])
    for path,digest in reg['artifact_hashes'].items():assert sha256(path)==digest,path
    assert reg['qualified_Train']==qualify_Train(work,protocol)
    return reg


def fit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();plan,protocol,work,base,parent=locations();started=time.monotonic()
    assert not(base/'fit_complete.json').exists()and not(base/'private_models.json').exists()
    reference=read(parent/'Train_feature_statistics.json')['full_teacher_references']
    speed=threshold_values(work)['Train_unknown_speed_upper'];cache={};coverage={};payload=0
    for sequence,q in reg['qualified_Train'].items():
        arrays=load_arrays(q['bank_path'],q['frames'],q['bank_sha256'])
        gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256']);batches=[]
        digests={k:hashlib.sha256()for k in ['early','late']}
        counts={k:dict(known=0,positive=0,negative=0)for k in ['early','late']}
        def sink(batch):
            nonlocal payload
            payload+=batch.x.nbytes+batch.y.nbytes
            assert payload<=reg['memory_payload_cap_bytes'],'Keep failure; no edge truncation is permitted'
            batches.append(batch);part='early'if batch.frame<=int(.75*q['frames'])else'late'
            c=counts[part];c['known']+=len(batch.y);c['positive']+=int(batch.y.sum());c['negative']+=int((batch.y==0).sum())
            digests[part].update(np.asarray([batch.frame],np.int64).tobytes()+batch.x.tobytes()+batch.y.tobytes())
        collector=ArchiveTrainCollector(speed,sink);digest=hashlib.sha256()
        for i,f in enumerate(arrays['frame_ids']):
            frame=int(f);lo,hi=map(int,arrays['offsets'][i:i+2]);left,right=map(int,bounds[i:i+2])
            rows=collector.step(sequence,frame,arrays['detections'][lo:hi],arrays['rgb'][lo:hi],
                arrays['ir'][lo:hi],gt[left:right],split='train')
            for row in rows:digest.update(mot_line(row).encode())
            if frame%500==0 or frame==q['frames']:
                guard_storage(work);atomic_json(base/'progress/fit.json',dict(phase='COLLECT_ALL6_TRAIN_ARCHIVE_EDGES',
                    sequence=sequence,frame=frame,frames=q['frames'],RAM_payload_bytes=payload,
                    seconds=time.monotonic()-started,GPU_work=False,new_accuracy=False))
        expected=reference[sequence]
        assert digest.hexdigest()==expected['original_teacher_MOT_sha256']
        assert len(arrays['detections'])==expected['proposal_rows']and collector.statistics['frames']==q['frames']
        assert collector.statistics['proposals']==len(arrays['detections'])and collector.history.frame==q['frames']
        cache[sequence]=batches
        coverage[sequence]=dict(frames=q['frames'],proposals=len(arrays['detections']),counts=counts,
            collector_statistics=collector.statistics,archive_statistics=collector.linker.statistics,
            original_teacher_MOT_exact=True,original_teacher_MOT_sha256=digest.hexdigest(),
            feature_target_packet_sha256={k:v.hexdigest()for k,v in digests.items()},
            bank_sha256=q['bank_sha256'],GT_sha256=q['GT_sha256'],GT_rows=len(gt),
            early_cut=int(.75*q['frames']))
        print(json.dumps(dict(sequence=sequence,frames=q['frames'],counts=counts,RAM_payload_bytes=payload)),flush=True)
    frozen_json(base/'Train_archive_coverage.json',dict(status='COMPLETE_ALL27251_TRAIN_FRAMES_NO_MODEL_YET',
        UTC=utc(),rows=coverage,feature_payload_RAM_bytes=payload,feature_payload_cap_bytes=reg['memory_payload_cap_bytes'],
        all_teacher_MOT_exact=True,all_frames_and_proposals_used=True,edge_truncation=False,feature_pair_disk_dump=False,
        registration_sha256=sha256(base/'fit_registered.json'),Dev_rows=0,Test_rows=0,new_GPU_work=False))
    early_videos=[s for s,c in coverage.items()if c['counts']['early']['known']>0]
    early_counts={s:c['counts']['early']for s,c in coverage.items()}
    assert early_videos and sum(c['positive']for c in early_counts.values())>0
    assert sum(c['negative']for c in early_counts.values())>0,'Archive classifier needs actual known wrong edges'
    count=0;total=np.zeros(len(FEATURE_NAMES),np.float64);squares=total.copy()
    for sequence,batches in cache.items():
        for b in batches:
            if b.frame>coverage[sequence]['early_cut']:continue
            count+=len(b.x);total+=b.x.sum(axis=0,dtype=np.float64);squares+=np.square(b.x.astype(np.float64)).sum(axis=0)
    mean=total/count;scale=np.maximum(np.sqrt(np.maximum(squares/count-mean**2,0)),1e-4)
    models={k:ArchiveLogistic(mean,scale,k)for k in ['RGB','paired']};history=[]
    def accumulate(model):
        acc=NewtonPass(model)
        for sequence in early_videos:
            c=early_counts[sequence]
            for b in cache[sequence]:
                if b.frame<=coverage[sequence]['early_cut']:
                    acc.add(b.x,b.y,class_video_weights(b.y,c['positive'],c['negative'],len(early_videos)))
        return acc
    for iteration in range(6):
        row=dict(pass_index=iteration+1,models={k:accumulate(m).step()for k,m in models.items()})
        history.append(row);atomic_json(base/'fit_history.json',dict(status='SIX_FIXED_UPDATES_FROM_BOUNDED_RAM',rows=history))
        print(json.dumps(row),flush=True)
    final={}
    for k,m in models.items():
        acc=accumulate(m);loss,gradient,hessian=acc.objective()
        final[k]=dict(weighted_objective_after_final_step=loss,gradient_norm=float(np.linalg.norm(gradient)),
            rows=acc.rows,weight=acc.weight,optimizer_updated_during_diagnostic=False)
    wrong={k:[]for k in models};late={k:dict(known=0,positive=0,negative=0)for k in models}
    for sequence,batches in cache.items():
        for b in batches:
            if b.frame<=coverage[sequence]['early_cut']:continue
            for k,m in models.items():
                p=m.probabilities(b.x);wrong[k].append(p[b.y==0].copy())
                late[k]['known']+=len(b.y);late[k]['positive']+=int(b.y.sum());late[k]['negative']+=int((b.y==0).sum())
    thresholds={k:{}for k in models}
    for k in models:
        a=np.concatenate(wrong[k]);assert len(a)>0 and np.isfinite(a).all()
        thresholds[k]={'q99':float(np.quantile(a,.99)),'q999':float(np.quantile(a,.999))}
    assert not models['RGB'].coefficients[list(IR_FEATURES)].any()
    private=dict(status='COMPLETE_PRIVATE_ARCHIVE_MODELS_NOT_FOR_GIT',UTC=utc(),
        models={k:m.to_dict()for k,m in models.items()},thresholds=thresholds,history=history,
        coverage_sha256=sha256(base/'Train_archive_coverage.json'),registration_sha256=sha256(base/'fit_registered.json'),
        final_fit_diagnostics=final,Train_only=True,seed=0)
    registered();guard_storage(work);frozen_json(base/'private_models.json',private)
    frozen_json(base/'fit_complete.json',dict(status='COMPLETE_SEPARATE_ARCHIVE_TRAIN_MODELS_NO_NEW_ACCURACY',UTC=utc(),
        model_file=str(base/'private_models.json'),model_sha256=sha256(base/'private_models.json'),
        registration_sha256=sha256(base/'fit_registered.json'),coverage_sha256=sha256(base/'Train_archive_coverage.json'),
        feature_names=list(FEATURE_NAMES),thresholds=thresholds,late_calibration_counts=late,
        early_videos_used=early_videos,empty_early_videos=[s for s in coverage if s not in early_videos],
        early_known_edges=count,Train_frames=27251,Train_sequences=6,Dev_rows=0,Test_rows=0,
        unique_Train_proposals=sum(v['proposals']for v in coverage.values()),optimizer_passes=6,
        bounded_RAM_feature_payload_bytes=payload,feature_pair_disk_dump=False,truncated_pairs=0,
        all_teacher_MOT_exact=True,all_frames_and_proposals_used=True,seed=0,new_GPU_work=False,
        final_fit_diagnostics=final,seconds=time.monotonic()-started,new_tracking_accuracy=False,
        encoder_Train_identity_exposure=read(SPEC)['training']['encoder_exposure'],
        goal_met=False,supported_innovations=0))
    atomic_json(base/'progress/fit.json',dict(phase='COMPLETE_ARCHIVE_FIT_NO_DEV_YET',UTC=utc(),new_accuracy=False))
    print(json.dumps(dict(status='COMPLETE_SEPARATE_ARCHIVE_TRAIN_MODELS_NO_NEW_ACCURACY',thresholds=thresholds,
        seconds=time.monotonic()-started)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['engineer','register','fit'])
    {'engineer':engineer,'register':register,'fit':fit}[p.parse_args().mode]()


if __name__=='__main__':main()
