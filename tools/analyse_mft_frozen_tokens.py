#!/usr/bin/env python3
"""Separate Train-GT oracle-gallery diagnostic for frozen current-box tokens."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from tools.run_mft_frozen_tokens import registered,locations,SPEC,utc
from tools.mft_candidate_risk_data import load_arrays,load_GT
from tools.mft_candidate_risk_labels import current_detection_labels
from tools.mft_candidate_risk_teacher import observations
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.run_pseudo_ir_comparison import sha256
from tools.run_mft_native_memory_priority_v3 import identity
from tools.audit_mft_candidate_risk import require_closed_jobs


def unit(x):
    x=np.asarray(x,np.float64);return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def paired(a,b):return unit(np.concatenate([a,b],axis=1))


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    started=time.monotonic();reg=registered();plan,protocol,work,base,train=locations();spec=read(SPEC)
    execution=read(base/'run_r1.execution.json');assert execution['actual_child_exit_code']==0 and execution['source_unchanged']and identity(execution['identity']['PID'])is None
    state=read(base/'state.json');require_closed_jobs(state,list(state['phases']),2)
    gpu=[read(base/('GPU_'+g+'.json'))for g in ['4','5']]
    assert all(r['model_all_weights_and_buffers_unchanged']and not r['GT_labels_read']for r in gpu)
    manifests={s:m for r in gpu for s,m in r['sequences'].items()};assert set(manifests)==set(reg['qualified_Train'])
    sources={**reg['source_hashes'],'tools/analyse_mft_frozen_tokens.py':sha256(Path(__file__))}
    frozen_json(base/'analysis_registered.json',dict(status='FROZEN_ORACLE_GALLERY_DIAGNOSTIC_BEFORE_TRAIN_LABEL_JOIN',UTC=utc(),
        source_hashes=sources,artifact_hashes={str(p):sha256(p)for p in [base/'registered.json',base/'state.json',base/'run_r1.execution.json',
            base/'GPU_4.json',base/'GPU_5.json']},gallery='All first75percent identities with>=3reliable actual proposals;first3only,including IDs without late queries.',
        queries='Reliable knownpositive last25percent proposals with a gallery and>=25frame age. Oracle labels are unavailable to online inference.',
        primary_pool='mean',no_parameter_fit=True,Dev_or_Test_rows=0))
    rows=[]
    for sequence,q in reg['qualified_Train'].items():
        meta=manifests[sequence];assert sha256(meta['path'])==meta['sha256']
        with np.load(meta['path'],allow_pickle=False)as data:a={k:data[k].copy()for k in data.files}
        assert a['frame_ids'].tolist()==reg['selected_frame_ids'][sequence]
        bank=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256'])
        counts=dict(all_proposals=0,known_positive=0,known_background=0,unknown=0,reliable_positive=0)
        parts={k:[]for k in ['legacy_RGB','RGB_mean','IR_mean','gray_mean','RGB_cls','IR_cls','gray_cls','identity','frame']}
        packet=hashlib.sha256();images=hashlib.sha256()
        for index,fid in enumerate(a['frame_ids']):
            frame=int(fid);lo,hi=map(int,a['offsets'][index:index+2]);oldlo,oldhi=map(int,bank['offsets'][frame-1:frame+1]);d=a['detections'][lo:hi]
            np.testing.assert_array_equal(d,bank['detections'][oldlo:oldhi])
            packet.update(np.asarray([frame,len(d)],dtype='<i8').tobytes()+np.asarray(d,dtype='<f4').tobytes())
            rp=Path(plan['data_root'])/'train'/sequence/'visible'/f'{frame:06d}.jpg'
            ip=Path(plan['generated_ir_root'])/'train'/sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
            image=dict(frame=frame,RGB_path=str(rp),IR_path=str(ip),RGB_sha256=sha256(rp),IR_sha256=sha256(ip))
            images.update(json.dumps(image,separators=(',',':')).encode()+b'\n')
            left,right=map(int,bounds[frame-1:frame+1]);labels=current_detection_labels(d,gt[left:right],frame=frame,split='train')
            obs=observations(d,bank['rgb'][oldlo:oldhi],bank['ir'][oldlo:oldhi])
            keep=np.asarray([j for j,o in enumerate(obs)if o['reliable']and labels[j]>0],np.int64)
            counts['all_proposals']+=len(d);counts['known_positive']+=int((labels>0).sum());counts['known_background']+=int((labels==0).sum())
            counts['unknown']+=int((labels<0).sum());counts['reliable_positive']+=len(keep)
            parts['legacy_RGB'].append(bank['rgb'][oldlo:oldhi][keep].astype(np.float32))
            for key in ['RGB_mean','IR_mean','gray_mean','RGB_cls','IR_cls','gray_cls']:parts[key].append(a[key][lo:hi][keep].astype(np.float32))
            parts['identity'].append(labels[keep]);parts['frame'].append(np.full(len(keep),frame,np.int64))
        assert packet.hexdigest()==meta['original_selected_detector_packet_sha256']and images.hexdigest()==meta['delivered_image_packet_sha256']
        parts={k:np.concatenate(v)for k,v in parts.items()};early=parts['frame']<=int(.75*q['frames'])
        gallery=[];ids=[];last=[]
        for label in np.unique(parts['identity'][early]):
            take=np.flatnonzero(early&(parts['identity']==label))
            if len(take)>=3:
                gallery.append(take[:3]);ids.append(int(label));last.append(int(parts['frame'][take[2]]))
        assert gallery
        lookup={label:index for index,label in enumerate(ids)};queries=[];truth=[];without_gallery=0;too_close=0
        for j in np.flatnonzero(~early):
            label=int(parts['identity'][j])
            if label not in lookup:without_gallery+=1;continue
            column=lookup[label]
            if int(parts['frame'][j])-last[column]<25:too_close+=1;continue
            queries.append(int(j));truth.append(column)
        assert queries
        query=np.asarray(queries,np.int64);truth=np.asarray(truth,np.int64)
        features={'legacy_RGB':unit(parts['legacy_RGB'])}
        for pool in ['mean','cls']:
            rgb=unit(parts['RGB_'+pool]);ir=unit(parts['IR_'+pool]);gray=unit(parts['gray_'+pool])
            features['DINO_RGB_'+pool]=rgb;features['DINO_RGB_IR_'+pool]=paired(rgb,ir)
            features['DINO_RGB_gray_'+pool]=paired(rgb,gray);features['DINO_duplicateRGB_'+pool]=paired(rgb,rgb)
        metrics={};scores={};predictions={}
        for name,x in features.items():
            prototypes=unit(np.stack([x[g].mean(axis=0)for g in gallery]));score=x[query]@prototypes.T
            # Stable smallest-gallery-ID tie break within1e-12, same for all arms.
            pred=np.argmax(score>=score.max(axis=1,keepdims=True)-1e-12,axis=1)
            scores[name]=score;predictions[name]=pred
            correct=score[np.arange(len(query)),truth];wrong=score.copy();wrong[np.arange(len(query)),truth]=-np.inf
            metrics[name]=dict(queries=len(query),gallery_identities=len(ids),known_top1_accuracy=float(np.mean(pred==truth)),
                mean_true_cosine=float(correct.mean()),mean_true_minus_best_wrong=float((correct-wrong.max(1)).mean())if len(ids)>1 else None,
                feature_dimensions=x.shape[1])
        numerical={}
        for pool in ['mean','cls']:
            error=float(np.max(abs(scores['DINO_duplicateRGB_'+pool]-scores['DINO_RGB_'+pool])))
            assert error<1e-12
            numerical[pool]=dict(max_duplicate_cosine_error=error,duplicate_top1_predictions_exact=bool(np.array_equal(predictions['DINO_duplicateRGB_'+pool],predictions['DINO_RGB_'+pool])))
        rows.append(dict(sequence=sequence,counts=counts,queries=len(query),gallery_ids=ids,gallery_IDs_without_late_queries_retained=True,
            late_without_gallery=without_gallery,late_too_close=too_close,metrics=metrics,duplicate_numerics=numerical,
            original_detector_and_delivered_image_packets_rebuilt_exact=True,bank_sha256=meta['sha256']))
        guard_storage(work)
    aggregate={k:float(np.mean([r['metrics'][k]['known_top1_accuracy']for r in rows]))for k in rows[0]['metrics']}
    rep=100*(aggregate['DINO_RGB_mean']-aggregate['legacy_RGB']);ir=100*(aggregate['DINO_RGB_IR_mean']-aggregate['DINO_RGB_mean'])
    gray=100*(aggregate['DINO_RGB_IR_mean']-aggregate['DINO_RGB_gray_mean']);queries=sum(r['queries']for r in rows)
    positive_rep=sum(r['metrics']['DINO_RGB_mean']['known_top1_accuracy']>r['metrics']['legacy_RGB']['known_top1_accuracy']for r in rows)
    positive_ir=sum(r['metrics']['DINO_RGB_IR_mean']['known_top1_accuracy']>r['metrics']['DINO_RGB_mean']['known_top1_accuracy']for r in rows)
    rep_gate=queries>=100 and rep>=spec['representation_gate']['DINO_RGB_minus_legacy_RGB_top1_pp']and positive_rep>=spec['representation_gate']['positive_videos']
    ir_gate=queries>=100 and ir>=spec['IR_identity_gate']['paired_minus_DINO_RGB_top1_pp']and gray>=spec['IR_identity_gate']['paired_minus_RGB_gray_top1_pp']and positive_ir>=spec['IR_identity_gate']['positive_videos']
    registered()
    for n,h in sources.items():assert sha256(ROOT/n)==h
    frozen_json(base/'complete_oracle_feature_diagnostic.json',dict(status='COMPLETE_ACTUAL_SPARSE_TRAIN_DINO_ORACLE_GALLERY_DIAGNOSTIC',UTC=utc(),
        rows=rows,equal_video_known_top1_accuracy=aggregate,queries=queries,DINO_RGB_minus_legacy_RGB_pp=rep,
        DINO_paired_minus_DINO_RGB_pp=ir,DINO_paired_minus_DINO_gray_pp=gray,positive_representation_videos=positive_rep,
        positive_IR_videos=positive_ir,representation_gate_passed=bool(rep_gate),IR_identity_gate_passed=bool(ir_gate),
        all6_Train_videos=True,actual_Train_frames=384,actual_proposals=6752,source_hashes=sources,
        oracle_Train_identity_galleries_not_online_tracking=True,no_HOTA_or_fullTrain_heldout_domain_claim=True,
        unknowns_only_excluded_from_identity_metric_not_encoded_input=True,old_ReID_Train_identity_exposure=True,
        external_DINO_pretraining_overlap_not_established=True,new_fits_or_seeds=0,Dev_or_Test_rows=0,
        seconds=time.monotonic()-started,goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_DINO_ORACLE_FEATURE_DIAGNOSTIC',queries=queries,representation_gain_pp=rep,
        paired_gain_pp=ir,paired_minus_gray_pp=gray,representation_gate=bool(rep_gate),IR_identity_gate=bool(ir_gate))),flush=True)


if __name__=='__main__':main()
