#!/usr/bin/env python3
"""Build stopped-target future-latent examples with clean and actual past histories."""
import argparse
from collections import defaultdict,deque
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from cfp.association import AppearanceTracker
from cfp.provenance import sha256,write_json,snapshot_sources
from cfp.supervision import independent_proposal_labels
from yolox.tracker.basetrack import BaseTrack

spec=importlib.util.spec_from_file_location('cfp_gt_annotation_helper',Path(__file__).with_name('cfp_build_utility_queries.py'))
gt_helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(gt_helper)


class PredictionRecorder(AppearanceTracker):
    def __init__(self,args,labels,frame_of_proposal,config,seed):
        super().__init__(args,mode='rgb',appearance_weight=.1,ema=.9)
        self.labels=labels;self.frame_of_proposal=frame_of_proposal;self.config=config
        self.rng=np.random.default_rng(seed);self.anchors={}
        self.accepted=defaultdict(lambda:deque(maxlen=64));self.clean=defaultdict(lambda:deque(maxlen=64))
        self.ambiguous=set();self.rows=[];self.exclusions=defaultdict(int)

    def association_cost(self,tracks,detections):
        result=super().association_cost(tracks,detections)
        if (self.frame_id-1)%self.config['query_frame_stride'] or not tracks or not detections:return result
        chosen=[d.current_proposal_index for d in detections if d.current_rgb is not None and self.labels[d.current_proposal_index]>=0 and self.labels[d.current_proposal_index] not in self.ambiguous]
        if len(chosen)<2:
            self.exclusions['fewer_than_two_same_frame_physical_ids']+=len(tracks);return result
        candidate_labels=self.labels[chosen]
        if len(np.unique(candidate_labels))!=len(chosen):raise ValueError('Nonunique physical hard-negative identities')
        for track in tracks:
            if track.track_id not in self.anchors:continue
            identity=self.anchors[track.track_id]
            if identity in self.ambiguous:continue
            positive=np.flatnonzero(candidate_labels==identity)
            if not len(positive):
                self.exclusions['unavailable_stopped_current_target']+=1;continue
            gap=int(self.rng.choice(self.config['gap_training_candidates']))
            cutoff=self.frame_id-gap
            actual=[v for v in self.accepted[track.track_id] if v[0]<=cutoff][-self.config['history_length']:]
            clean=[v for v in self.clean[identity] if v[0]<=cutoff][-self.config['history_length']:]
            if not actual or not clean:
                self.exclusions['missing_clean_or_actual_gap_history']+=1;continue
            if any(t>=self.frame_id or self.frame_of_proposal[index]!=t for t,index in actual+clean):
                raise ValueError('Current/future latent observation in predictor inputs')
            purity=float(np.mean([self.labels[index]==identity for _,index in actual]))
            self.rows.append(dict(frame=self.frame_id,track_id=track.track_id,anchor_id=identity,requested_gap=gap,
                              positive_index=int(positive[0]),candidate_indices=np.array(chosen,np.int64),
                              actual=actual,clean=clean,purity=purity))
        return result

    def collect_writes(self,frame,lo,hi,valid_rgb):
        # Called after all association stages/writes; labels annotate outputs only.
        for track in self.tracked_stracks:
            index=track.current_proposal_index
            if track.last_identity_frame==frame and index is not None and track.current_rgb is not None:
                if not lo<=index<hi:raise ValueError('Recorded write is not from current proposal frame')
                self.accepted[track.track_id].append((frame,index))
                if track.track_id not in self.anchors and self.labels[index]>=0 and self.labels[index] not in self.ambiguous:self.anchors[track.track_id]=int(self.labels[index])
        for index in range(lo,hi):
            if self.labels[index]>=0 and valid_rgb[index] and self.labels[index] not in self.ambiguous:self.clean[int(self.labels[index])].append((frame,index))


def save_rows(path,rows,length):
    values={k:np.array([r[k] for r in rows],np.int64) for k in ['frame','track_id','anchor_id','requested_gap','positive_index']}
    values['history_purity']=np.array([r['purity'] for r in rows],np.float32)
    for key in ['actual','clean']:
        indices=np.full((len(rows),length),-1,np.int64);times=np.zeros((len(rows),length),np.int64)
        for i,r in enumerate(rows):
            n=len(r[key]);times[i,:n]=[v[0] for v in r[key]];indices[i,:n]=[v[1] for v in r[key]]
        values[key+'_history_indices']=indices;values[key+'_history_frames']=times
    values['candidate_offsets']=np.concatenate([[0],np.cumsum([len(r['candidate_indices']) for r in rows])]).astype(np.int64)
    values['candidate_indices']=np.concatenate([r['candidate_indices'] for r in rows])
    np.savez_compressed(path,**values)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--detections',type=Path,required=True)
    p.add_argument('--features-root',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--partition',choices=['train','dev'],required=True)
    p.add_argument('--query-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Predictive query data is immutable')
    config=json.loads(args.config.read_text());protocol=json.loads(args.protocol.read_text())
    query_config=json.loads(args.query_config.read_text())
    if query_config['state']!='FROZEN' or query_config['trial_sha256']!=sha256(args.config):raise ValueError('Frozen predictive sampling policy required')
    config={**config,**query_config}
    if config['state']!='FROZEN' or protocol['state']!='FROZEN' or config['history_length']!=8:raise ValueError('Declared bounded eight-observation first predictive trial required')
    if sorted(config['fit_videos']+config['calibration_videos'])!=sorted(protocol['train']):raise ValueError('Predictor fit/calibration must partition association train videos')
    if sha256(args.data_root/'annotations/train.json')!=protocol['train_annotation_sha256']:raise ValueError('Prediction physical labels changed')
    dm=json.loads((args.detections/'manifest.json').read_text());fm=json.loads((args.features_root/'manifest.json').read_text())
    if dm['status']!='COMPLETE' or dm['probe'] or dm['protocol_sha256']!=sha256(args.protocol) or fm['status']!='COMPLETE' or fm['protocol_sha256']!=sha256(args.protocol):raise ValueError('Complete canonical proposal/feature inputs required')
    if fm['detection_manifest_sha256']!=sha256(args.detections/'manifest.json') or fm.get('encoder_role','main_baseline')!='main_baseline' or fm['embedding_stage']!='post_neck' or fm.get('crop_variant','original')!='original':raise ValueError('First predictive trial needs consistent frozen main post-neck coordinate space')
    if set(fm.get('selected_sequences',list(fm['videos'])))!=set(protocol[args.partition]):raise ValueError('Full selected video partition required')
    for name in ['rgb_encoder','ir_encoder']:
        if fm[name]['training_metadata']['protocol_sha256']!=sha256(args.protocol):raise ValueError('Frozen latent target encoder uses another split')
    seed=fm['rgb_encoder']['seed']
    if fm['ir_encoder']['seed']!=seed:raise ValueError('Matched latent encoder seeds required')
    gt=gt_helper.gt_by_video(json.loads((args.data_root/'annotations/train.json').read_text()),protocol)
    policy=defaultdict(set)
    for r in protocol['ambiguous_identity_keys']:policy[(r['sequence'],r['frame_id'])].add(r['track_id'])
    args.output.mkdir(parents=True)
    sources=snapshot_sources(args.output,[__file__,'tools/cfp_build_utility_queries.py','cfp/association.py','cfp/supervision.py','cfp/provenance.py',args.config,args.query_config,args.protocol])
    report=dict(status='RUNNING',partition=args.partition,encoder_seed=seed,config_sha256=sha256(args.config),query_config_sha256=sha256(args.query_config),protocol_sha256=sha256(args.protocol),config=config,
                features_root=str(args.features_root),feature_manifest_sha256=sha256(args.features_root/'manifest.json'),source_snapshots=sources,
                source_sha256=sha256(__file__),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),videos={},
                scope='Frozen main encoder stopped current targets; clean and actual accepted histories strictly before sampled gap. Main fish encoder saw all six association-train videos; only temporal query fitting holds out SN-015. GT/purity fields are retrospective training diagnostics, never model inference inputs.')
    write_json(args.output/'manifest.json',report)
    for sequence in protocol[args.partition]:
        dp=args.detections/(sequence+'.npz');fp=args.features_root/(sequence+'.npz');prov=json.loads(fp.with_suffix('.provenance.json').read_text())
        if sha256(dp)!=dm['caches'][sequence]['sha256'] or sha256(dp)!=prov['detection_sha256'] or sha256(fp)!=prov['features_sha256']:raise ValueError('Frozen latent/proposal content changed')
        with np.load(dp) as a:frames,offsets,boxes,scores=[a[k] for k in ['frame_ids','offsets','boxes','scores']]
        with np.load(fp) as a:bank={k:a[k] for k in a.files}
        if not np.array_equal(frames,np.arange(1,len(frames)+1)) or len(offsets)!=len(frames)+1 or offsets[0]!=0 or offsets[-1]!=len(boxes) or np.any(np.diff(offsets)<0):raise ValueError('Invalid proposal frame indexing')
        if bank['rgb'].shape!=(len(boxes),512) or bank['ir'].shape!=(len(boxes),512) or bank['valid_rgb'].shape!=(len(boxes),) or len(scores)!=len(boxes):raise ValueError('Unaligned stopped feature bank')
        if not all(np.isfinite(v).all() for v in (boxes,scores,bank['rgb'],bank['ir'])):raise ValueError('Nonfinite proposal/features')
        labels=np.full(len(boxes),-1,np.int64);ambiguity={};frame_of_proposal=np.repeat(frames,np.diff(offsets))
        for i,frame in enumerate(frames):
            identities,gb=zip(*gt[sequence][int(frame)]) if gt[sequence][int(frame)] else ([],[])
            lo,hi=offsets[i:i+2];labels[lo:hi],ambiguity[int(frame)]=independent_proposal_labels(boxes[lo:hi],gb,identities,minimum_iou=config['physical_matching_min_iou'],minimum_gap=config['physical_matching_ambiguity_gap'])
            ambiguity[int(frame)].update(policy[(sequence,int(frame))])
        BaseTrack._count=0;tracker=PredictionRecorder(SimpleNamespace(track_thresh=.58,track_buffer=45,match_thresh=.97,mot20=False),labels,frame_of_proposal,config,seed)
        for i,frame in enumerate(frames):
            lo,hi=offsets[i:i+2];tracker.ambiguous=ambiguity[int(frame)]
            tracker.update(np.concatenate([boxes[lo:hi].copy(),scores[lo:hi,None]],1),(1,1),(1,1),bank['rgb'][lo:hi],bank['ir'][lo:hi],np.arange(lo,hi))
            tracker.collect_writes(int(frame),int(lo),int(hi),bank['valid_rgb'])
        if not tracker.rows:raise ValueError('No physically supervised causal latent targets')
        out=args.output/(sequence+'.npz');save_rows(out,tracker.rows,config['history_length'])
        purity=np.array([r['purity'] for r in tracker.rows]);entry=dict(queries=len(tracker.rows),sha256=sha256(out),features_sha256=sha256(fp),
                clean_available_queries=len(tracker.rows),actual_history_mean_purity=float(purity.mean()),fully_contaminated_history_fraction=float(np.mean(purity==0)),
                requested_gap_counts={str(g):sum(r['requested_gap']==g for r in tracker.rows) for g in config['gap_training_candidates']},exclusions=dict(tracker.exclusions))
        report['videos'][sequence]=entry;write_json(args.output/'manifest.json',report);print(sequence,json.dumps(entry),flush=True)
    report['status']='COMPLETE';write_json(args.output/'manifest.json',report)


if __name__=='__main__':main()
