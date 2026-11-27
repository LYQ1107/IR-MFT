#!/usr/bin/env python3
"""Matched pre/post-BN inference on fixed sparse actual Train proposals.

No retraining, GT crop or Test/Dev image. Producers see no GT; the separate CPU
consumer labels exactly the already produced detector boxes for diagnostics.
"""
import argparse
import fcntl
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
from tools.native_pyramid_experiment import read,frozen_json,gpu_settings
from tools.native_pyramid_training_loop import guard_storage
from tools.run_mft_candidate_train_bank import atomic_json
from tools.run_pseudo_ir_comparison import sha256
from tools.task_utility_queue import Queue
from tools.run_mft_native_memory_priority_v3 import identity

SPEC=ROOT/'configs/perceptual_descriptor_location_probe_20261127.json'


def locations():
    plan,protocol,work,parent=parent_locations();return plan,protocol,work,work/'descriptor_location_probe_20261127'


def utc():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def register():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    plan,protocol,work,base=locations();q=qualify_Train(work,protocol)
    parent=read(work/'actual_proposal_metric_20261127/registered.json');sources=dict(parent['source_hashes'])
    for n in ['tools/probe_mft_native_descriptor_location.py',str(SPEC.relative_to(ROOT)),
              'tools/extract_perceptual_mft_features.py','yolox/tracker/native_cue.py'] :sources[n]=sha256(ROOT/n)
    for n,h in sources.items():assert sha256(ROOT/n)==h,n
    source=read(work/'tracklet_permanence_20261127_r2/Train_thresholds.json')['sources']
    for cue,meta in source.items():assert sha256(meta['checkpoint'])==meta['checkpoint_sha256']
    selected={s:np.unique(np.linspace(1,v['frames'],64,dtype=int)).tolist()for s,v in q.items()}
    budget=guard_storage(work);assert budget['work_file_bytes']+int(.1*2**30)<64*2**30
    v=dict(status='REGISTERED_MATCHED_PRE_POST_BN_TRAIN_PROPOSAL_PROBE_BEFORE_IMAGES',UTC=utc(),source_hashes=sources,
        qualified_Train=q,selected_frame_ids=selected,frames=sum(map(len,selected.values())),encoder_sources=source,
        specification_sha256=sha256(SPEC),source_metric_fit_loss_uses_preBN=True,current_bank_uses_postBN=True,
        no_bug_or_accuracy_gain_concluded_from_difference=True,GT_free_GPU_producers=True,private_probe_payload_cap_bytes=64*2**20,
        inference_seed=0,new_training_seeds=0,new_NN_fits=0,physical_GPUs=['4','5'],maximum_cards=2,
        Dev_frames=0,Test_frames=0,historical_Test_previously_exposed=True,goal_met=False,supported_innovations=0)
    assert not(base/'registered.json').exists();frozen_json(base/'registered.json',v)
    print(json.dumps(dict(status=v['status'],frames=v['frames'],sources=len(sources))),flush=True)


def registered():
    plan,protocol,work,base=locations();r=read(base/'registered.json')
    for n,h in r['source_hashes'].items():assert sha256(ROOT/n)==h,n
    assert r['specification_sha256']==sha256(SPEC)
    for m in r['encoder_sources'].values():assert sha256(m['checkpoint'])==m['checkpoint_sha256']
    return r


def produce(sequence):
    import cv2
    import torch
    from torch.nn import functional as F
    from tools.extract_perceptual_mft_features import AppearanceEncoder
    from yolox.tracker.native_cue import native_crop,native_input
    from tools.probe_mft_native_memory_priority import tensor_digest
    reg=registered();plan,protocol,work,base=locations();q=reg['qualified_Train'][sequence]
    physical,uuid=gpu_settings();assert torch.cuda.mem_get_info()[0]>=16*2**30
    models={}
    for cue,meta in reg['encoder_sources'].items():
        with torch.random.fork_rng(devices=[]):torch.manual_seed(0);model=AppearanceEncoder()
        saved=torch.load(meta['checkpoint'],map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'],strict=True);models[cue]=model.cuda().float().eval().requires_grad_(False)
    before={k:tensor_digest(m)for k,m in models.items()}
    arrays=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);selected=reg['selected_frame_ids'][sequence]
    parts={k:[]for k in ['detections','RGB_pre','RGB_post','IR_pre','IR_post']};offsets=[0];images=[];parity={'RGB':0.,'IR':0.}
    for frame in selected:
        lo,hi=map(int,arrays['offsets'][frame-1:frame+1]);d=arrays['detections'][lo:hi]
        rgbpath=Path(plan['data_root'])/'train'/sequence/'visible'/f'{frame:06d}.jpg'
        irpath=Path(plan['generated_ir_root'])/'train'/sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
        rb,ib=rgbpath.read_bytes(),irpath.read_bytes();rgb=cv2.imdecode(np.frombuffer(rb,np.uint8),cv2.IMREAD_COLOR)
        ir=cv2.imdecode(np.frombuffer(ib,np.uint8),cv2.IMREAD_UNCHANGED)
        assert rgb is not None and ir is not None and rgb.shape[:2]==ir.shape and ir.ndim==2
        images.append(dict(frame=frame,RGB_path=str(rgbpath),IR_path=str(irpath),RGB_sha256=hashlib.sha256(rb).hexdigest(),IR_sha256=hashlib.sha256(ib).hexdigest()))
        for cue,pixels in [('RGB',rgb),('IR',ir)]:
            pre=[];post=[];model=models[cue.lower()]
            with torch.inference_mode():
                for begin in range(0,len(d),32):
                    pairs=[native_crop(pixels,box)for box in d[begin:begin+32,:4]]
                    pp=torch.from_numpy(np.stack([v[0]for v in pairs]));rect=torch.from_numpy(np.stack([v[1]for v in pairs]))
                    raw=model.projection(model.backbone(native_input(pp,rect,'cuda')))
                    pre.append(F.normalize(raw,dim=-1).cpu().numpy());post.append(F.normalize(model.neck(raw),dim=-1).cpu().numpy())
            pre=np.concatenate(pre)if pre else np.empty((0,512),np.float32)
            post=np.concatenate(post)if post else np.empty((0,512),np.float32)
            expected=arrays[cue.lower()][lo:hi].astype(np.float32)
            error=float(np.max(np.abs(post-expected)))if len(post)else 0.
            assert error<=1e-3,(sequence,frame,cue,error);parity[cue]=max(parity[cue],error)
            for where,value in [('pre',pre),('post',post)]:
                assert np.isfinite(value).all()and np.allclose(np.linalg.norm(value,axis=1),1,atol=1e-5)
                parts[cue+'_'+where].append(value.astype(np.float16))
        parts['detections'].append(d.copy());offsets.append(offsets[-1]+len(d))
    assert {k:tensor_digest(m)for k,m in models.items()}==before
    result={k:np.concatenate(v)for k,v in parts.items()};result['offsets']=np.asarray(offsets,np.int64);result['frame_ids']=np.asarray(selected,np.int64)
    payload=sum(v.nbytes for v in result.values());assert payload<=32*2**20
    folder=base/'private_cache';folder.mkdir(exist_ok=True);path=folder/(sequence+'.npz');assert not path.exists()
    np.savez_compressed(path,**result);registered();guard_storage(work)
    frozen_json(folder/(sequence+'.json'),dict(status='COMPLETE_MATCHED_PRE_POST_BN_ACTUAL_TRAIN_CROPS',sequence=sequence,
        frames=len(selected),selected_frame_ids=selected,proposals=len(result['detections']),uncompressed_bytes=payload,path=str(path),
        sha256=sha256(path),bytes=path.stat().st_size,image_sources=images,postBN_FP16_bank_max_abs_error=parity,
        all_weights_and_BN_unchanged=True,GT_read=False,oracle_crops=False,new_NN_fits=0,
        physical_gpu=physical,GPU_uuid=uuid,actual_identity=identity(os.getpid()),source_versions=reg['source_hashes'],
        registration_sha256=sha256(base/'registered.json')))
    print(json.dumps(dict(sequence=sequence,frames=len(selected),proposals=len(result['detections']),parity=parity,physical_gpu=physical)),flush=True)


def analyse():
    from tools.mft_candidate_risk_labels import current_detection_labels
    from tools.mft_candidate_risk_teacher import observations
    from tools.mft_identity_projection import unit_rows,paired_features
    reg=registered();plan,protocol,work,base=locations();rows=[];payload=0
    for sequence,q in reg['qualified_Train'].items():
        meta=read(base/'private_cache'/(sequence+'.json'));assert sha256(meta['path'])==meta['sha256']and not meta['GT_read']
        assert meta['all_weights_and_BN_unchanged']and meta['selected_frame_ids']==reg['selected_frame_ids'][sequence]
        with np.load(meta['path'],allow_pickle=False)as data:a={k:data[k]for k in data.files}
        payload+=sum(v.nbytes for v in a.values());assert payload<=reg['private_probe_payload_cap_bytes']
        gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256'])
        parts={k:[]for k in ['RGB_pre','RGB_post','IR_pre','IR_post','identity','frame']};counts=dict(all_proposals=0,positive=0,known_background=0,unknown=0,reliable_positive=0)
        for i,f in enumerate(a['frame_ids']):
            frame=int(f);lo,hi=map(int,a['offsets'][i:i+2]);left,right=map(int,bounds[frame-1:frame+1]);d=a['detections'][lo:hi]
            labels=current_detection_labels(d,gt[left:right],frame=frame,split='train')
            obs=observations(d,a['RGB_post'][lo:hi],a['IR_post'][lo:hi]);keep=np.asarray([j for j,o in enumerate(obs)if o['reliable']and labels[j]>0],np.int64)
            counts['all_proposals']+=len(d);counts['positive']+=int((labels>0).sum());counts['known_background']+=int((labels==0).sum());counts['unknown']+=int((labels<0).sum());counts['reliable_positive']+=len(keep)
            for k in ['RGB_pre','RGB_post','IR_pre','IR_post']:parts[k].append(a[k][lo:hi][keep].astype(np.float32))
            parts['identity'].append(labels[keep]);parts['frame'].append(np.full(len(keep),frame,np.int64))
        parts={k:np.concatenate(v)for k,v in parts.items()};early=parts['frame']<=int(.75*q['frames']);gallery=[];truth=[];queries=[]
        for label in np.unique(parts['identity'][early]):
            indices=np.flatnonzero(early&(parts['identity']==label))
            if len(indices)<3:continue
            query=np.flatnonzero(~early&(parts['identity']==label))
            if not len(query):continue
            gallery.append(indices[:3]);truth.extend([len(gallery)-1]*len(query));queries.extend(query.tolist())
        assert gallery and queries,(sequence,'No valid fixed-probe early/late template evidence')
        query=np.asarray(queries,np.int64);truth=np.asarray(truth,np.int64);metrics={}
        features={k:unit_rows(parts[k])for k in ['RGB_pre','RGB_post','IR_pre','IR_post']}
        for where in ['pre','post']:features['paired_'+where]=paired_features(features['RGB_'+where],features['IR_'+where])
        for k,x in features.items():
            g=unit_rows(np.asarray([x[indices].mean(axis=0)for indices in gallery]));correct=int(((x[query]@g.T).argmax(axis=1)==truth).sum())
            metrics[k]=dict(correct=correct,queries=len(query),top1_percent=100*correct/len(query))
        rows.append(dict(sequence=sequence,frames=len(a['frame_ids']),counts=counts,gallery_identities=len(gallery),
            late_queries=len(query),metrics=metrics,metadata_sha256=sha256(base/'private_cache'/(sequence+'.json'))))
    means={k:float(np.mean([r['metrics'][k]['top1_percent']for r in rows]))for k in rows[0]['metrics']}
    rgb_gain=means['RGB_pre']-means['RGB_post'];wins=sum(r['metrics']['RGB_pre']['top1_percent']>r['metrics']['RGB_post']['top1_percent']for r in rows)
    paired_gain=means['paired_pre']-means['RGB_pre'];paired_wins=sum(r['metrics']['paired_pre']['top1_percent']>r['metrics']['RGB_pre']['top1_percent']for r in rows)
    registered();frozen_json(base/'complete.json',dict(status='COMPLETE_MATCHED_SPARSE_TRAIN_PRE_POST_BN_DIAGNOSTIC',UTC=utc(),rows=rows,
        frames=384,proposals=sum(r['counts']['all_proposals']for r in rows),private_payload_bytes=payload,
        equal_video_top1_percent=means,RGB_pre_minus_post_pp=rgb_gain,RGB_pre_video_wins=wins,
        paired_pre_minus_RGB_pre_pp=paired_gain,paired_pre_video_wins=paired_wins,
        prospective_RGB_progress_gate=rgb_gain>=2 and wins>=4,prospective_paired_progress_gate=paired_gain>=1 and paired_wins>=4,
        postBN_parity_to_existing_bank=True,all_encoders_frozen=True,new_NN_fits=0,new_training_seeds=0,
        Dev_rows=0,Test_rows=0,not_fullTrain_or_tracking_HOTA=True,encoder_previously_seen_all_Train_identities=True,
        no_bug_concluded_merely_from_feature_location=True,goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_MATCHED_SPARSE_TRAIN_PRE_POST_BN_DIAGNOSTIC',means=means,RGB_gain=rgb_gain)),flush=True)


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    reg=registered();plan,protocol,work,base=locations()
    with(base/'state.lock').open('w')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(base/'state.json').exists()
        state=dict(status='STARTING_MATCHED_DESCRIPTOR_LOCATION_PROBE',controller_pid=os.getpid(),phases={},seed=0,goal_met=False)
        queue=Queue(work,base,state,plan['data_root'],registered);queue.save();cmd=[sys.executable,'-u',str(Path(__file__).relative_to(ROOT))]
        try:
            queue.jobs('DESCRIPTOR_LOCATION_GPU',[dict(id=s,command=cmd+['produce','--sequence',s])for s in reg['qualified_Train']],True,
                lambda j:read(base/'private_cache'/(j['id']+'.json'))['all_weights_and_BN_unchanged'])
            queue.jobs('DESCRIPTOR_LOCATION_TRAIN_ANALYSIS',[dict(id='all6',command=cmd+['analyse'])],False,
                lambda j:read(base/'complete.json')['frames']==384)
            state.update(status='COMPLETE_MATCHED_DESCRIPTOR_LOCATION_PROBE');queue.save()
        except Exception as error:
            state.update(status='FAILED_PRESERVE_DESCRIPTOR_LOCATION_PROBE',error=repr(error));queue.save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['register','run','produce','analyse']);p.add_argument('--sequence');a=p.parse_args()
    if a.mode=='produce':produce(a.sequence)
    else:{'register':register,'run':run,'analyse':analyse}[a.mode]()
