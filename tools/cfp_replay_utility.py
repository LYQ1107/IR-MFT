#!/usr/bin/env python3
"""Replay CMUE and matched gate controls without supervision/oracle inputs."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch

from cfp.provenance import sha256, write_json, snapshot_sources
from cfp.utility import UtilityMLP
from cfp.utility_association import UtilityAssociationTracker
from yolox.tracker.basetrack import BaseTrack


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--detections',type=Path,required=True)
    p.add_argument('--features-root',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--execution-config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--kind',choices=['cmue','ordinary','quality','fixed_matched','untrained','fixed_constant_ir','fixed_row_mean_ir','uncertainty'],required=True)
    p.add_argument('--appearance-weight',type=float,required=True)
    p.add_argument('--no-calibration',action='store_true')
    p.add_argument('--no-fallback',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Utility replay outputs are immutable')
    protocol=json.loads(args.protocol.read_text());config=json.loads(args.config.read_text());execution=json.loads(args.execution_config.read_text())
    if protocol['state']!='FROZEN' or config['state']!='FROZEN' or execution['state']!='FROZEN' or args.appearance_weight not in execution['appearance_weight_candidates']:
        raise ValueError('Frozen utility execution budget required')
    if execution.get('parent_target_config')!=args.config.name:raise ValueError('Utility execution targets another frozen trial')
    if (args.no_calibration or args.no_fallback) and args.kind!='cmue':raise ValueError('Calibration/fallback controls apply to CMUE only')
    if config['maximum_ir_weight']!=.5 or config['teacher_temperature']!=.1 or config['null_distance']!=.35:
        raise ValueError('This adapter requires its declared first-trial scale')
    det_meta=json.loads((args.detections/'manifest.json').read_text());feature_meta=json.loads((args.features_root/'manifest.json').read_text())
    if det_meta['status']!='COMPLETE' or det_meta['probe'] or det_meta['protocol_sha256']!=sha256(args.protocol) or feature_meta['status']!='COMPLETE' or feature_meta['protocol_sha256']!=sha256(args.protocol):
        raise ValueError('Complete matched proposal/feature coverage required')
    if feature_meta['detection_manifest_sha256']!=sha256(args.detections/'manifest.json') or feature_meta['embedding_stage']!='post_neck' or feature_meta.get('crop_variant','original')!='original':
        raise ValueError('CMUE trial requires original-crop post-neck fixed features')
    if feature_meta.get('encoder_role','main_baseline')!='main_baseline' or set(feature_meta.get('selected_sequences',list(feature_meta['videos'])))!=set(protocol['dev']):
        raise ValueError('This research replay is canonical dev only with heldout main encoders')
    for branch in ('rgb_encoder','ir_encoder'):
        if feature_meta[branch]['training_metadata']['protocol_sha256']!=sha256(args.protocol):raise ValueError('Encoder split changed')
    seed=feature_meta['rgb_encoder']['seed']
    if feature_meta['ir_encoder']['seed']!=seed:raise ValueError('Paired encoder seeds differ')
    torch.set_num_threads(1)
    model,checkpoint,calibration=None,None,None
    if args.kind not in ('fixed_matched','fixed_constant_ir','fixed_row_mean_ir','uncertainty'):
        if args.checkpoint is None:raise ValueError('Learned/untrained gate comparison requires checkpoint provenance')
        checkpoint=torch.load(args.checkpoint,map_location='cpu')
        expected_kind='ordinary' if args.kind=='untrained' else args.kind
        if checkpoint['epoch']!=config['epochs'] or checkpoint['kind']!=expected_kind or checkpoint['seed']!=seed or checkpoint['metadata']['config_sha256']!=sha256(args.config) or checkpoint['metadata']['protocol_sha256']!=sha256(args.protocol):
            raise ValueError('Gate checkpoint kind/seed/protocol differs from declared pipeline')
        torch.manual_seed(seed)
        model=UtilityMLP(config['mlp']['hidden'],quality_only=args.kind=='quality').eval()
        if args.kind!='untrained':model.load_state_dict(checkpoint['model'],strict=True)
        calibration=checkpoint['calibration']
    args.output.mkdir(parents=True)
    sources=snapshot_sources(args.output,[__file__,'cfp/association.py','cfp/utility.py',
                       'cfp/utility_association.py','cfp/candidate_context.py','cfp/provenance.py',args.config,args.execution_config,args.protocol])
    manifest=dict(status='RUNNING',kind=args.kind,seed=seed,partition='dev',protocol_sha256=sha256(args.protocol),
                  config=config,config_sha256=sha256(args.config),execution_config=execution,execution_config_sha256=sha256(args.execution_config),
                  appearance_weight=args.appearance_weight,threshold=execution['threshold'],
                  no_calibration=args.no_calibration,no_fallback=args.no_fallback,calibration=calibration,
                  gate_checkpoint_sha256=sha256(args.checkpoint) if args.checkpoint else None,
                  gate_metadata=checkpoint['metadata'] if checkpoint else None,
                  feature_manifest_sha256=sha256(args.features_root/'manifest.json'),
                  rgb_encoder=feature_meta['rgb_encoder'],ir_encoder=feature_meta['ir_encoder'],
                  source_sha256=sha256(__file__),tracker_source_sha256=sha256('cfp/association.py'),
                  source_snapshots=sources,
                  utility_tracker_source_sha256=sha256('cfp/utility_association.py'),utility_source_sha256=sha256('cfp/utility.py'),
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  scope='No annotations/GT labels are loaded. Same RGB-defined candidate graph across matched controls; native external trackers remain separate strongest association comparators.',videos={})
    write_json(args.output/'manifest.json',manifest)
    if args.kind=='untrained':
        torch.save(dict(model=model.state_dict(),seed=seed),args.output/'untrained_initial_model.pt')
        manifest['untrained_model_sha256']=sha256(args.output/'untrained_initial_model.pt')
    for sequence in protocol['dev']:
        det_path=args.detections/(sequence+'.npz');feature_path=args.features_root/(sequence+'.npz')
        provenance=json.loads(feature_path.with_suffix('.provenance.json').read_text())
        if sha256(det_path)!=det_meta['caches'][sequence]['sha256'] or sha256(det_path)!=provenance['detection_sha256'] or sha256(feature_path)!=provenance['features_sha256']:
            raise ValueError('Fixed proposal/feature content changed')
        with np.load(det_path) as bank:frames,offsets,boxes,scores=[bank[k] for k in ['frame_ids','offsets','boxes','scores']]
        with np.load(feature_path) as bank:rgb,ir=bank['rgb'],bank['ir']
        if len(rgb)!=len(boxes) or ir.shape!=rgb.shape or not np.array_equal(frames,np.arange(1,len(frames)+1)) or offsets[-1]!=len(boxes):
            raise ValueError('Complete aligned feature coverage required')
        BaseTrack._count=0
        tracker=UtilityAssociationTracker(SimpleNamespace(track_thresh=.58,track_buffer=45,match_thresh=.97,mot20=False),
                model=model,kind=args.kind,calibration=calibration,threshold=execution['threshold'],
                no_calibration=args.no_calibration,no_fallback=args.no_fallback,
                appearance_weight=args.appearance_weight,ema=.9,proximity_iou=.1,appearance_max_distance=.35)
        tracker.candidate_context_version=config.get('candidate_context_version',1)
        tracker.teacher_association_weight=config.get('teacher_association_weight',.3)
        rows=[];latencies=[]
        for i,frame in enumerate(frames):
            lo,hi=offsets[i:i+2];now=time.perf_counter()
            tracks=tracker.update(np.concatenate([boxes[lo:hi].copy(),scores[lo:hi,None]],1),(1,1),(1,1),rgb[lo:hi],ir[lo:hi],np.arange(lo,hi))
            latencies.append(time.perf_counter()-now)
            for track in tracks:
                x,y,w,h=track.tlwh
                if w*h>100:rows.append(f'{frame},{track.track_id},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{float(track.score):.6f},-1,-1,-1\n')
        output=args.output/(sequence+'.txt');output.write_text(''.join(rows))
        manifest['videos'][sequence]=dict(prediction_sha256=sha256(output),detection_sha256=sha256(det_path),features_sha256=sha256(feature_path),
                 frames=len(frames),association_seconds=sum(latencies),association_latency_p50_p95_ms=(np.quantile(latencies,[.5,.95])*1000).tolist(),
                 decision_counts=tracker.decision_counts,utility_counts=tracker.utility_counts)
        print(sequence,json.dumps(manifest['videos'][sequence]),flush=True)
    manifest.update(status='COMPLETE',latency_scope='Association plus CPU gate only; excludes encoders/generator/detector/decode. Shared CPU run is not a dedicated online speed benchmark.')
    write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
