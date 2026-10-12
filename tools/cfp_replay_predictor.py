#!/usr/bin/env python3
"""Replay fixed canonical-dev proposals with trained predictors or memory controls."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch

from cfp.predictive import LatentIdentityPredictor
from cfp.predictive_association import PredictiveAssociationTracker
from cfp.joint_association import JointAssociationTracker
from cfp.utility import UtilityMLP
from cfp.reliable_writes import ReliableWritesMixin
from cfp.provenance import sha256,write_json,snapshot_sources
from yolox.tracker.basetrack import BaseTrack


class ReliablePredictiveTracker(ReliableWritesMixin,PredictiveAssociationTracker):
    pass


class ReliableJointTracker(ReliableWritesMixin,JointAssociationTracker):
    pass


def main():
    p=argparse.ArgumentParser();p.add_argument('--detections',type=Path,required=True);p.add_argument('--features-root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--execution-config',type=Path,required=True);p.add_argument('--memory',choices=['predictor','last','ema','fifo','all_history_ema','constant_zero','constant_half','anchor_only'],required=True)
    p.add_argument('--utility-kind',choices=['cmue','ordinary','quality','fixed_matched','uncertainty']);p.add_argument('--utility-config',type=Path);p.add_argument('--utility-execution-config',type=Path);p.add_argument('--utility-checkpoint',type=Path)
    p.add_argument('--write-policy-config',type=Path);p.add_argument('--save-write-audit',action='store_true');p.add_argument('--checkpoint',type=Path);p.add_argument('--appearance-weight',type=float,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Predictive replay outputs are immutable')
    config=json.loads(args.config.read_text());execution=json.loads(args.execution_config.read_text());protocol=json.loads(args.protocol.read_text())
    if any(v['state']!='FROZEN' for v in [config,execution,protocol]) or execution['trial_sha256']!=sha256(args.config) or args.appearance_weight not in execution['appearance_weight_candidates']:raise ValueError('Frozen predictive trial and equal control budget required')
    dm=json.loads((args.detections/'manifest.json').read_text());fm=json.loads((args.features_root/'manifest.json').read_text())
    if dm['status']!='COMPLETE' or dm['probe'] or dm['protocol_sha256']!=sha256(args.protocol) or fm['status']!='COMPLETE' or fm['protocol_sha256']!=sha256(args.protocol) or fm['detection_manifest_sha256']!=sha256(args.detections/'manifest.json'):raise ValueError('Complete canonical detection/feature inputs required')
    if set(fm.get('selected_sequences',list(fm['videos'])))!=set(protocol['dev']) or fm.get('encoder_role','main_baseline')!='main_baseline' or fm['embedding_stage']!='post_neck' or fm.get('crop_variant','original')!='original':raise ValueError('This first replay permits frozen canonical-dev original/post-neck inputs only')
    if any(fm[k]['training_metadata']['protocol_sha256']!=sha256(args.protocol) for k in ['rgb_encoder','ir_encoder']):raise ValueError('Latent encoder split changed')
    policy=None
    if args.write_policy_config:
        policy=json.loads(args.write_policy_config.read_text())
        if policy['state']!='FROZEN' or policy['parent_execution_sha256']!=sha256(args.execution_config) or policy['appearance_weight']!=args.appearance_weight:raise ValueError('Write toggle needs its frozen identical-score comparison')
    model=None;model_metadata=None
    torch.set_num_threads(2)
    if args.memory=='predictor':
        if args.checkpoint is None:raise ValueError('Frozen trained predictor checkpoint required')
        path=args.checkpoint.parent/'manifest.json';model_metadata=json.loads(path.read_text())
        if model_metadata['status']!='COMPLETE' or sha256(args.checkpoint)!=model_metadata['checkpoint_sha256'] or model_metadata['config_sha256']!=sha256(args.config) or model_metadata['training_config_sha256']!=execution['training_config_sha256'] or model_metadata['protocol_sha256']!=sha256(args.protocol) or model_metadata['seed']!=fm['rgb_encoder']['seed']:raise ValueError('Matched complete fixed predictor required')
        training_features=json.loads((Path(model_metadata['features_root'])/'manifest.json').read_text())
        if sha256(Path(model_metadata['features_root'])/'manifest.json')!=model_metadata['feature_manifest_sha256'] or training_features['rgb_encoder']['checkpoint_sha256']!=fm['rgb_encoder']['checkpoint_sha256']:raise ValueError('Predictor target latent coordinate space differs from deployment encoder')
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        if checkpoint['epoch']!=8:raise ValueError('Fixed epoch8 checkpoint required')
        model=LatentIdentityPredictor(512,model_metadata['architecture'],time_encoding=model_metadata.get('time_encoding','linear_clipped'));model.load_state_dict(checkpoint['model'],strict=True);model.eval()
        if execution.get('history_view','recent')=='anchor_recent' and model_metadata['history']!='anchor_recent':raise ValueError('Anchor-context deployment requires matched anchor-context training')
    elif args.checkpoint is not None:raise ValueError('Memory controls do not use learned checkpoints')
    utility_kwargs={};utility_metadata=None
    if args.utility_kind:
        if args.utility_config is None or args.utility_execution_config is None:raise ValueError('Joint utility needs frozen target/execution configurations')
        uc=json.loads(args.utility_config.read_text());ue=json.loads(args.utility_execution_config.read_text())
        if uc['state']!='FROZEN' or ue['state']!='FROZEN' or ue['parent_target_config']!=args.utility_config.name:raise ValueError('Joint utility target/execution differs')
        gate=calibration=None
        if args.utility_kind in ['cmue','ordinary','quality']:
            if args.utility_checkpoint is None:raise ValueError('Joint learned utility needs its completed checkpoint')
            checkpoint=torch.load(args.utility_checkpoint,map_location='cpu',weights_only=False)
            if checkpoint['epoch']!=uc['epochs'] or checkpoint['kind']!=args.utility_kind or checkpoint['seed']!=fm['rgb_encoder']['seed'] or checkpoint['metadata']['config_sha256']!=sha256(args.utility_config) or checkpoint['metadata']['protocol_sha256']!=sha256(args.protocol):raise ValueError('Joint gate checkpoint kind/seed/protocol changed')
            gate=UtilityMLP(uc['mlp']['hidden'],quality_only=args.utility_kind=='quality').eval();gate.load_state_dict(checkpoint['model'],strict=True);calibration=checkpoint['calibration']
        elif args.utility_checkpoint is not None:raise ValueError('Parameter-free gate must not load a learned checkpoint')
        utility_kwargs=dict(utility_model=gate,utility_kind=args.utility_kind,calibration=calibration,utility_threshold=ue['threshold'],utility_context_version=uc.get('candidate_context_version',1),teacher_association_weight=uc.get('teacher_association_weight',.3))
        utility_metadata=dict(kind=args.utility_kind,target_config_sha256=sha256(args.utility_config),execution_sha256=sha256(args.utility_execution_config),checkpoint_sha256=sha256(args.utility_checkpoint) if args.utility_checkpoint else None,scope='Frozen utility uses trained EMA evidence; accepted IR EMA mixes with predicted RGB only on original utility subset. Not joint retraining or paired latent prediction.')
    elif any(v is not None for v in [args.utility_config,args.utility_execution_config,args.utility_checkpoint]):raise ValueError('Utility paths require a declared joint gate')
    args.output.mkdir(parents=True)
    snapshots=snapshot_sources(args.output,[__file__,'cfp/predictive.py','cfp/predictive_association.py','cfp/association.py','cfp/provenance.py',args.config,args.execution_config,args.protocol,'cfp/reliable_writes.py','cfp/history.py']+([args.write_policy_config] if args.write_policy_config else [])+(['cfp/joint_association.py','cfp/utility_association.py','cfp/utility.py','cfp/candidate_context.py',args.utility_config,args.utility_execution_config] if args.utility_kind else []))
    report=dict(status='RUNNING',partition='dev',memory=args.memory,appearance_weight=args.appearance_weight,seed=fm['rgb_encoder']['seed'],config=config,execution_config=execution,protocol_sha256=sha256(args.protocol),feature_manifest_sha256=sha256(args.features_root/'manifest.json'),rgb_encoder=fm['rgb_encoder'],ir_encoder=fm['ir_encoder'],predictor_checkpoint_sha256=sha256(args.checkpoint) if args.checkpoint else None,model_metadata=model_metadata,source_snapshots=snapshots,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),write_policy=policy,write_policy_config_sha256=sha256(args.write_policy_config) if args.write_policy_config else None,save_write_audit=args.save_write_audit,videos={},scope='Frozen canonical association dev, historical detector exposure disclosed. Shared geometry candidate graph for every predictor/history control; unchanged low-score IoU/lifecycle, score before writes. No GT loaded. Write policy independently toggled after geometry updates. No historical test or end-to-end speed claim.')
    write_json(args.output/'manifest.json',report)
    report['joint_utility']=utility_metadata
    for sequence in protocol['dev']:
        dp=args.detections/(sequence+'.npz');fp=args.features_root/(sequence+'.npz');prov=json.loads(fp.with_suffix('.provenance.json').read_text())
        if sha256(dp)!=dm['caches'][sequence]['sha256'] or sha256(dp)!=prov['detection_sha256'] or sha256(fp)!=prov['features_sha256']:raise ValueError('Frozen deployment arrays changed')
        with np.load(dp) as a:frames,offsets,boxes,scores=[a[k] for k in ['frame_ids','offsets','boxes','scores']]
        with np.load(fp) as a:rgb=a['rgb'];ir=a['ir'] if args.utility_kind else None
        if not np.array_equal(frames,np.arange(1,len(frames)+1)) or len(offsets)!=len(frames)+1 or offsets[0]!=0 or offsets[-1]!=len(boxes) or np.any(np.diff(offsets)<0) or rgb.shape!=(len(boxes),512) or len(scores)!=len(boxes) or not all(np.isfinite(v).all() for v in [boxes,scores,rgb]):raise ValueError('Invalid complete deployment alignment')
        BaseTrack._count=0;g=execution['geometry'];tracker_type=ReliableJointTracker if args.utility_kind else ReliablePredictiveTracker
        tracker=tracker_type(SimpleNamespace(track_thresh=g['track_thresh'],track_buffer=g['track_buffer'],match_thresh=g['match_thresh'],mot20=False),predictor=model,memory=args.memory,write_policy=policy,history_view=execution.get('history_view','recent'),appearance_weight=args.appearance_weight,frame_rate=g['frame_rate'],ema=execution['ema'],proximity_iou=execution['proximity_iou'],**utility_kwargs)
        rows=[];latencies=[];audit_rows=[]
        for i,frame in enumerate(frames):
            lo,hi=offsets[i:i+2];detections=np.concatenate([boxes[lo:hi].copy(),scores[lo:hi,None]],1);start=time.perf_counter()
            targets=tracker.update(detections,(1,1),(1,1),rgb[lo:hi],ir[lo:hi] if ir is not None else None,proposal_indices=np.arange(lo,hi));latencies.append(time.perf_counter()-start)
            if args.save_write_audit:
                if policy is not None:decisions=tracker.last_write_decisions
                else:decisions=[dict(frame=int(frame),track_id=int(t.track_id),proposal_index=t.current_proposal_index,accepted=True,written=True,reason='ordinary_write',confidence=float(t.score),row_margin=None,column_margin=None) for t in tracker.tracked_stracks if t.last_identity_frame==frame]
                audit_rows.extend(json.dumps(d,allow_nan=False)+'\n' for d in decisions)
            for t in targets:
                x,y,w,h=t.tlwh
                if w*h>g['min_box_area']:rows.append(f'{frame},{t.track_id},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{float(t.score):.6f},-1,-1,-1\n')
        output=args.output/(sequence+'.txt');output.write_text(''.join(rows));report['videos'][sequence]=dict(frames=len(frames),prediction_sha256=sha256(output),detection_sha256=sha256(dp),features_sha256=sha256(fp),association_seconds=sum(latencies),association_latency_p50_p95_ms=(np.quantile(latencies,[.5,.95])*1000).tolist(),prediction_counts=tracker.prediction_counts,decision_counts=tracker.decision_counts)
        if args.save_write_audit:
            audit=args.output/(sequence+'_writes.jsonl');audit.write_text(''.join(audit_rows));report['videos'][sequence]['write_audit_sha256']=sha256(audit)
        report['videos'][sequence]['write_counts']=tracker.write_counts
        if args.utility_kind:report['videos'][sequence].update(joint_counts=tracker.joint_counts,utility_counts=tracker.utility_scorer.utility_counts)
        write_json(args.output/'manifest.json',report);print(sequence,tracker.prediction_counts,flush=True)
    report.update(status='COMPLETE',latency_scope='CPU association/predictor/cache replay only, shared machine; decoding/generator/detector/encoder excluded. Not online FPS.');write_json(args.output/'manifest.json',report)


if __name__=='__main__':main()
