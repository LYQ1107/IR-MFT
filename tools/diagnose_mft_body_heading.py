#!/usr/bin/env python3
"""Train-only cached principal-axis versus oracle motion-heading diagnostic."""
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
from tools.run_mft_birth_only import registered as parent_registered,locations
from tools.mft_candidate_risk_data import load_arrays,load_GT
from tools.mft_candidate_risk_labels import current_detection_labels
from tools.native_pyramid_experiment import read,frozen_json
from tools.native_pyramid_training_loop import guard_storage
from tools.run_pseudo_ir_comparison import sha256

SPEC=ROOT/'configs/perceptual_body_heading_probe_20261127.json'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    started=time.monotonic();parent=parent_registered();plan,protocol,work,peer,train=locations()
    base=work/'body_heading_probe_20261127';old=work/'IR_body_geometry_probe_20261127';spec=read(SPEC)
    cached=read(old/'complete.json');selection=read(old/'selection_before_images.json')
    assert cached['frames']==30 and cached['sequences']==6 and cached['source_sha256']==sha256(ROOT/'tools/probe_mft_IR_body_geometry.py')
    assert selection['source_sha256']==cached['source_sha256']
    sources={**parent['source_hashes'],'tools/diagnose_mft_body_heading.py':sha256(Path(__file__)),str(SPEC.relative_to(ROOT)):sha256(SPEC)}
    for n,h in sources.items():assert sha256(ROOT/n)==h,n
    original_inputs=cached['original_image_inputs']
    for x in original_inputs:assert sha256(x['path'])==x['sha256']
    frozen_json(base/'registered.json',dict(status='REGISTERED_ORACLE_MOTION_TARGET_TRAIN_HEADING_PROBE',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_hashes=sources,
        artifact_hashes={str(p):sha256(p)for p in [SPEC,old/'complete.json',old/'selection_before_images.json']},
        original_image_hashes_rechecked=len(original_inputs),same30_Train_frames=selection['frames'],
        labels_for_targets_only=True,GT_crops=False,Dev_or_Test_rows=0,new_GPU_work=False,new_models=False))
    observations=[];r=spec['strict_reliability'];qualified=selection['qualified_Train']
    def reliable(m):return bool(m['valid']and m['anisotropy']>=r['anisotropy_min']and m['dominance']>=r['dominance_min']and r['area_fraction_min']<=m['area_fraction']<=r['area_fraction_max'])
    for sequence,q in qualified.items():
        a=load_arrays(q['bank_path'],q['frames'],q['bank_sha256']);gt,bounds=load_GT(q['GT_path'],q['frames'],q['GT_sha256'])
        own=[x for x in cached['observations']if x['sequence']==sequence]
        for frame in selection['frames'][sequence]:
            lo,hi=map(int,a['offsets'][frame-1:frame+1]);left,right=map(int,bounds[frame-1:frame+1]);d=a['detections'][lo:hi];current=gt[left:right]
            labels=current_detection_labels(d,current,frame=frame,split='train')
            rows=sorted([x for x in own if x['frame']==frame],key=lambda x:x['proposal_index'])
            expected=np.flatnonzero((labels>0)&(d[:,4]>.6)).tolist()
            assert [x['proposal_index']for x in rows]==expected
            for index,x in enumerate(rows):
                label=int(labels[x['proposal_index']]);truth=current[(current[:,1]==label)&(current[:,6]>0)&(current[:,7]==1)];assert len(truth)==1
                center=truth[0,2:4]+truth[0,4:6]/2
                shifted=rows[(index-1)%len(rows)]
                for lag in spec['lags']:
                    if frame<=lag:continue
                    pleft,pright=map(int,bounds[frame-lag-1:frame-lag+1]);past=gt[pleft:pright]
                    past=past[(past[:,1]==label)&(past[:,6]>0)&(past[:,7]==1)]
                    if not len(past):continue
                    assert len(past)==1;move=center-(past[0,2:4]+past[0,4:6]/2);distance=float(np.linalg.norm(move))
                    if distance<spec['minimum_GT_displacement_pixels']:continue
                    angle=float(np.arctan2(move[1],move[0]));direction=np.array([np.cos(2*angle),np.sin(2*angle)])
                    features={}
                    for cue in ['RGB_gray','IR']:
                        m=x[cue];rolled=shifted[cue]
                        axis=np.array([m['axis_cos2'],m['axis_sin2']]);control=np.array([rolled['axis_cos2'],rolled['axis_sin2']])
                        features[cue]=dict(valid=m['valid'],strict=reliable(m),axis_motion_dot=float(axis@direction)if m['valid']else None,
                            rolled_valid=rolled['valid']and len(rows)>1,rolled_axis_motion_dot=float(control@direction)if rolled['valid']and len(rows)>1 else None)
                    observations.append(dict(sequence=sequence,frame=frame,proposal_index=x['proposal_index'],lag=lag,
                        oracle_GT_displacement_pixels=distance,RGB_gray=features['RGB_gray'],IR=features['IR']))
        guard_storage(work)
    summaries={};aggregate={}
    for lag in spec['lags']:
        summaries[str(lag)]={};aggregate[str(lag)]={}
        for mode in ['all_covalid','strict_covalid']:
            videos={}
            for sequence in qualified:
                rows=[x for x in observations if x['sequence']==sequence and x['lag']==lag and all(x[c]['valid']for c in ['RGB_gray','IR'])
                    and(mode=='all_covalid'or all(x[c]['strict']for c in ['RGB_gray','IR']))]
                if not rows:
                    videos[sequence]=dict(rows=0,IR_minus_gray=None,IR_minus_rolled_IR=None,rolled_rows=0);continue
                gray=float(np.mean([x['RGB_gray']['axis_motion_dot']for x in rows]));ir=float(np.mean([x['IR']['axis_motion_dot']for x in rows]))
                rolled=[x for x in rows if x['IR']['rolled_valid']]
                difference=float(np.mean([x['IR']['axis_motion_dot']-x['IR']['rolled_axis_motion_dot']for x in rolled]))if rolled else None
                videos[sequence]=dict(rows=len(rows),RGB_gray_mean_axis_motion_dot=gray,IR_mean_axis_motion_dot=ir,IR_minus_gray=ir-gray,
                    IR_minus_rolled_IR=difference,rolled_rows=len(rolled))
            summaries[str(lag)][mode]=videos;present=[v for v in videos.values()if v['rows']]
            control=[v for v in videos.values()if v['IR_minus_rolled_IR']is not None]
            aggregate[str(lag)][mode]=dict(rows=sum(v['rows']for v in videos.values()),videos=len(present),
                equal_video_IR_minus_gray=float(np.mean([v['IR_minus_gray']for v in present]))if present else None,
                equal_video_IR_minus_rolled_IR=float(np.mean([v['IR_minus_rolled_IR']for v in control]))if control else None,
                positive_videos=sum(v['IR_minus_gray']>0 for v in present),rolled_control_videos=len(control))
    primary=aggregate[str(spec['primary_lag'])]['strict_covalid'];gate=spec['pilot_gate']
    passed=bool(primary['rows']>=gate['minimum_strict_moving_rows_at_primary_lag']and primary['positive_videos']>=gate['minimum_positive_videos']
        and primary['equal_video_IR_minus_gray']is not None and primary['equal_video_IR_minus_gray']>=gate['minimum_equal_video_IR_minus_gray']
        and primary['equal_video_IR_minus_rolled_IR']is not None and primary['equal_video_IR_minus_rolled_IR']>=gate['minimum_equal_video_IR_minus_permuted_IR'])
    for n,h in sources.items():assert sha256(ROOT/n)==h,n
    frozen_json(base/'complete.json',dict(status='COMPLETE_CACHED30TRAIN_ORACLE_HEADING_FEATURE_DIAGNOSTIC',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),per_sequence=summaries,equal_video_aggregate=aggregate,
        preregistered_primary=primary,pilot_gate_passed=passed,observations=observations,source_hashes=sources,
        no_identity_discrimination_or_tracking_accuracy_claim=True,oracle_GT_motion_targets_not_inference_inputs=True,
        original_images_hash_checked_without_regeneration=True,new_models=False,new_GPU_work=False,Dev_or_Test_rows=0,
        new_image_crop_or_feature_disk_bank=False,seconds=time.monotonic()-started,goal_met=False,supported_innovations=0))
    print(json.dumps(dict(status='COMPLETE_CACHED30TRAIN_ORACLE_HEADING_FEATURE_DIAGNOSTIC',primary=primary,pilot_gate_passed=passed)),flush=True)


if __name__=='__main__':main()
