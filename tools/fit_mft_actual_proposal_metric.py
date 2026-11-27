#!/usr/bin/env python3
"""Train-only Fisher diagnostic on actual ReID detection crops, not GT ROIs.

This classical transform is an investigation tool, not a novelty claim. Every
frame/proposal is processed; only known positive reliable observations supervise
the transform. Held-video labels enter retrieval scoring, never projection fit.
"""
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
from tools.fit_mft_candidate_risk import locations as parent_locations
from tools.mft_candidate_risk_data import qualify_Train,load_arrays,load_GT
from tools.mft_candidate_risk_teacher import observations
from tools.mft_candidate_risk_labels import current_detection_labels
from tools.mft_identity_projection import fit_fisher,paired_features,transform,unit_rows
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.run_mft_candidate_train_bank import atomic_json
from tools.run_pseudo_ir_comparison import sha256

SPEC=ROOT/'configs/perceptual_actual_proposal_metric_20261127.json'


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def locations():
    plan,protocol,work,parent=parent_locations();return plan,protocol,work,work/'actual_proposal_metric_20261127'


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan,protocol,work,base=locations();q=qualify_Train(work,protocol)
    parent=read(work/'identity_ledger_20261127/fit_registered.json')
    sources=dict(parent['source_hashes'])
    for n in ['tools/fit_mft_actual_proposal_metric.py',str(SPEC.relative_to(ROOT)),'tools/mft_identity_projection.py']:sources[n]=sha256(ROOT/n)
    for n,h in sources.items():assert sha256(ROOT/n)==h,n
    spec=read(SPEC);assert spec['dimensions']==32 and spec['ridge_fraction']==.1
    v=dict(status='REGISTERED_ALL_TRAIN_ACTUAL_PROPOSAL_METRIC_BEFORE_COLLECTION',UTC=utc(),source_hashes=sources,
        qualified_Train=q,specification_sha256=sha256(SPEC),all_Train_frames=27251,Dev_rows=0,Test_rows=0,seed=0,
        GT_crop_or_feature_extraction=False,no_new_images_or_descriptor_bank=True,coefficient_or_normalization_publication=False,
        encoder_previously_seen_all_Train_identities=True,held_video_exclusion_scope='Projection fitting only',
        historical_Test_previously_exposed=True,goal_met=False,supported_innovations=0)
    assert not(base/'registered.json').exists();frozen_json(base/'registered.json',v)
    print(json.dumps(dict(status=v['status'],sources=len(sources))),flush=True)


def registered():
    plan,protocol,work,base=locations();r=read(base/'registered.json')
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    assert r['specification_sha256']==sha256(SPEC)
    return r


def retrieve(gallery,query,truth):
    if not len(truth):return dict(queries=0,correct=0,top1_percent=None)
    g=unit_rows(gallery);q=unit_rows(query);correct=0
    for lo in range(0,len(q),2048):correct+=int(((q[lo:lo+2048]@g.T).argmax(axis=1)==truth[lo:lo+2048]).sum())
    return dict(queries=len(truth),correct=correct,top1_percent=100*correct/len(truth))


def gallery_query(row):
    early,late=row['early'],row['late'];labels=np.unique(early['identity']);keep=[]
    for label in labels:
        indices=np.flatnonzero(early['identity']==label)
        if len(indices)>=3:keep.append((label,indices[:3]))
    names=[v[0]for v in keep];mapping={label:i for i,label in enumerate(names)}
    query_indices=np.asarray([i for i,label in enumerate(late['identity'])if label in mapping],np.int64)
    truth=np.asarray([mapping[int(late['identity'][i])]for i in query_indices],np.int64)
    g={cue:np.asarray([unit_rows(early[cue][indices]).mean(axis=0)for _,indices in keep],np.float32)for cue in ['RGB','IR']}
    for cue in ['RGB','IR']:g[cue+'_seeds']=np.asarray([early[cue][indices]for _,indices in keep],np.float32)
    q={cue:late[cue][query_indices]for cue in ['RGB','IR']}
    return g,q,truth,dict(gallery_identities=len(keep),gallery_observations=3*len(keep),
        late_reliable_positive_rows=len(late['identity']),unknown_identity_query_rows_excluded=len(late['identity'])-len(truth),
        gallery_definition='First3 reliable known positive actual detections per identity in first75% of held Train video;GT diagnostic only')


def fit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();plan,protocol,work,base=locations();spec=read(SPEC);started=time.monotonic();cache={};coverage={};payload=0
    assert not(base/'complete.json').exists()and not(base/'private_projections.npz').exists()
    for sequence,q in reg['qualified_Train'].items():
        arrays=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256'])
        rows={part:{k:[]for k in ['RGB','IR','identity','frame']}for part in ['early','late']}
        counts=dict(frames=0,proposals=0,known_positive=0,known_background=0,unknown=0,reliable_known_positive=0)
        signatures={k:hashlib.sha256()for k in rows};cut=int(.75*q['frames'])
        for i,f in enumerate(arrays['frame_ids']):
            frame=int(f);lo,hi=map(int,arrays['offsets'][i:i+2]);left,right=map(int,bounds[i:i+2]);d=arrays['detections'][lo:hi]
            labels=current_detection_labels(d,gt[left:right],frame=frame,split='train');obs=observations(d,arrays['rgb'][lo:hi],arrays['ir'][lo:hi])
            counts['frames']+=1;counts['proposals']+=len(d);counts['known_positive']+=int((labels>0).sum())
            counts['known_background']+=int((labels==0).sum());counts['unknown']+=int((labels<0).sum())
            indices=[j for j,o in enumerate(obs)if o['reliable']and labels[j]>0]
            counts['reliable_known_positive']+=len(indices);part='early'if frame<=cut else'late'
            if indices:
                batch={cue:np.asarray([obs[j][cue.lower()]for j in indices],np.float32)for cue in ['RGB','IR']}
                batch['identity']=labels[indices].copy();batch['frame']=np.full(len(indices),frame,np.int64)
                signatures[part].update(np.asarray([frame,len(indices)],np.int64).tobytes())
                for k,v in batch.items():rows[part][k].append(v);payload+=v.nbytes;signatures[part].update(v.tobytes())
                assert payload<=spec['maximum_RAM_input_payload_bytes'],'Never truncate proposals or qualified observations'
            if frame%1000==0 or frame==q['frames']:
                guard_storage(work);atomic_json(base/'progress.json',dict(status='COLLECTING_ALL_TRAIN_ACTUAL_ROI_METRIC',UTC=utc(),
                    sequence=sequence,frame=frame,frames=q['frames'],RAM_input_payload_bytes=payload,seconds=time.monotonic()-started))
        cache[sequence]={part:{k:np.concatenate(v)if v else np.empty((0,512),np.float32)if k in ['RGB','IR']else np.empty(0,np.int64)
            for k,v in values.items()}for part,values in rows.items()}
        assert counts['frames']==q['frames']and counts['proposals']==len(arrays['detections'])
        coverage[sequence]=dict(**counts,early_cut=cut,early_fit_qualified_rows=len(cache[sequence]['early']['identity']),
            late_retrieval_qualified_rows=len(cache[sequence]['late']['identity']),packet_sha256={k:v.hexdigest()for k,v in signatures.items()},
            bank_sha256=q['bank_sha256'],GT_sha256=q['GT_sha256'])
        print(json.dumps(dict(sequence=sequence,coverage=coverage[sequence])),flush=True)
        del arrays,gt,rows
    def fitting(videos,kind):
        blocks=[];video=[];identity=[];excluded={}
        for index,s in enumerate(videos):
            row=cache[s]['early'];labels,c=np.unique(row['identity'],return_counts=True);allowed=labels[c>=3]
            keep=np.isin(row['identity'],allowed);excluded[s]=dict(identities=int((c<3).sum()),observations=int(c[c<3].sum()))
            v=row['RGB'][keep]if kind=='RGB'else paired_features(row['RGB'][keep],row['IR'][keep]).astype(np.float32)
            blocks.append(v);video.append(np.full(len(v),index,np.int64));identity.append(row['identity'][keep])
        values=np.concatenate(blocks);ids=np.concatenate(identity);vid=np.concatenate(video)
        parameters,detail=fit_fisher(values,vid,ids,spec['dimensions'],spec['ridge_fraction'])
        return parameters,dict(**detail,fit_videos=videos,excluded_sparse_early_classes=excluded)
    held_rows=[];sequences=list(cache)
    for held in sequences:
        others=[s for s in sequences if s!=held];g,q,truth,ginfo=gallery_query(cache[held]);metrics={};fits={}
        for name in ['RGB','IR']:metrics['raw_'+name]=retrieve(g[name],q[name],truth)
        seed_RGB=g['RGB_seeds'].reshape(-1,512);seed_IR=g['IR_seeds'].reshape(-1,512)
        metrics['raw_paired']=retrieve(paired_features(seed_RGB,seed_IR).reshape(len(g['RGB']),3,-1).mean(axis=1),
            paired_features(q['RGB'],q['IR']),truth)
        for kind in ['RGB','paired']:
            projection,detail=fitting(others,kind);fits[kind]=detail
            gallery=seed_RGB if kind=='RGB'else paired_features(seed_RGB,seed_IR)
            queries=q['RGB']if kind=='RGB'else paired_features(q['RGB'],q['IR'])
            projected_gallery=transform(gallery,projection).reshape(len(g['RGB']),3,-1).mean(axis=1)
            metrics['projected_'+kind]=retrieve(projected_gallery,transform(queries,projection),truth)
        held_rows.append(dict(held_sequence=held,gallery_query=ginfo,metrics=metrics,fits=fits,
            held_video_features_and_labels_excluded_from_projection_fit=True,encoder_pretraining_had_held_Train_identities=True))
        atomic_json(base/'progress.json',dict(status='PROJECTION_ONLY_LEAVE_ONE_TRAIN_VIDEO_OUT',UTC=utc(),held_sequence=held,
            rows=held_rows,seconds=time.monotonic()-started));print(json.dumps(dict(held_sequence=held,metrics=metrics)),flush=True)
    names=['raw_RGB','raw_IR','raw_paired','projected_RGB','projected_paired']
    means={name:float(np.mean([r['metrics'][name]['top1_percent']for r in held_rows]))for name in names}
    rgb_gain=means['projected_RGB']-means['raw_RGB'];paired_gain=means['projected_paired']-means['projected_RGB']
    rgb_wins=sum(r['metrics']['projected_RGB']['top1_percent']>r['metrics']['raw_RGB']['top1_percent']for r in held_rows)
    paired_wins=sum(r['metrics']['projected_paired']['top1_percent']>r['metrics']['projected_RGB']['top1_percent']for r in held_rows)
    proceed_RGB=rgb_gain>=2 and rgb_wins>=4;proceed_paired=paired_gain>=1 and paired_wins>=4
    parameters={};details={}
    for kind in ['RGB','paired']:
        p,d=fitting(sequences,kind);details[kind]=d
        parameters.update({kind+'_'+key:v for key,v in p.items()})
    path=base/'private_projections.npz';np.savez_compressed(path,**parameters);assert path.stat().st_size<=2**21
    registered();budget=guard_storage(work)
    frozen_json(base/'complete.json',dict(status='COMPLETE_FULL_TRAIN_ACTUAL_PROPOSAL_FISHER_DIAGNOSTIC',UTC=utc(),coverage=coverage,
        Train_frames=sum(r['frames']for r in coverage.values()),unique_proposals=sum(r['proposals']for r in coverage.values()),
        RAM_input_payload_bytes=payload,leave_one_Train_video_out_rows=held_rows,equal_video_top1_percent=means,
        projected_RGB_minus_raw_RGB_pp=rgb_gain,projected_paired_minus_projected_RGB_pp=paired_gain,
        RGB_held_video_wins=rgb_wins,paired_held_video_wins=paired_wins,prospective_RGB_progress_gate=proceed_RGB,
        prospective_paired_progress_gate=proceed_paired,full_early_Train_projection_details=details,
        private_model_file=str(path),private_model_sha256=sha256(path),private_model_bytes=path.stat().st_size,
        registration_sha256=sha256(base/'registered.json'),Dev_rows=0,Test_rows=0,seed=0,new_GPU_work=False,
        new_images_GT_ROIs_or_descriptor_banks=False,only_coefficients_saved_privately=True,
        legacy_Fisher_difference='Actual detector proposals and independently trained512D ReID crops on all27251Train frames;old study used sampled GT-box320D teacher FPN pooled features.',
        projection_only_held_video_exclusion_not_whole_encoder_generalization=True,not_tracking_HOTA=True,
        not_a_new_algorithm_or_validated_multimodal_contribution=True,budget=budget,seconds=time.monotonic()-started,
        goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_FULL_TRAIN_ACTUAL_PROPOSAL_FISHER_DIAGNOSTIC',means=means,
        RGB_gate=proceed_RGB,paired_gate=proceed_paired,seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','fit']);a=p.parse_args()
    register()if a.mode=='register'else fit()
