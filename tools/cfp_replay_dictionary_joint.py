#!/usr/bin/env python3
"""Complete-video new-coordinate identity with actual native memories."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch

from cfp.dictionary_separated_joint import DictionarySeparatedJoint
from cfp.native_dictionary_joint import install_dictionary_joint
from cfp.native_local_identity import RecentIdentityControl, install_native_local_identity
from cfp.predictive_association import PredictiveAssociationTracker
from cfp.provenance import sha256, snapshot_sources, write_json
from yolox.tracker.basetrack import BaseTrack


def main():
    ap = argparse.ArgumentParser()
    for name in ('config', 'features', 'detections', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--memory', choices=('last', 'fifo', 'ema'), default='fifo')
    ap.add_argument('--model', type=Path)
    ap.add_argument('--context-identity', action='store_true')
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    if (config['state']!='FROZEN' or args.video not in config['videos'] or args.output.exists()
            or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or args.context_identity and args.model is None):
        raise ValueError('Declared full video, CPU-only and immutable new output required')
    if shutil.disk_usage(args.output.parent).free<40*2**30:
        raise OSError('40GiBreserve')
    fm = json.loads((args.features/'manifest.json').read_text())
    dm = json.loads((args.detections/'manifest.json').read_text())
    dp = args.detections/(args.video+'.npz')
    if (fm['status']!='COMPLETE_NATIVE_CURRENT_DINO_FEATURES' or fm['video']!=args.video
            or sha256(dp)!=fm['detection_sha256'] or sha256(dp)!=dm['caches'][args.video]['sha256']
            or fm['detection_manifest_sha256']!=sha256(args.detections/'manifest.json')):
        raise ValueError('Complete native current-only features and unchanged proposals required')
    with np.load(dp, allow_pickle=False) as a:
        frames, offsets, boxes, scores = [a[k].copy() for k in ('frame_ids', 'offsets', 'boxes', 'scores')]
    if sha256(args.features/'frame_ids.npy')!=fm['frame_ids_sha256'] or sha256(args.features/'offsets.npy')!=fm['offsets_sha256']:
        raise ValueError('Native feature timestamps changed')
    if not np.array_equal(frames,np.load(args.features/'frame_ids.npy')) or not np.array_equal(offsets,np.load(args.features/'offsets.npy')):
        raise ValueError('Native feature/proposal full-frame indexing differs')
    bank = {}
    for name in ('embedding', 'raw_cls', 'local4', 'visibility4', 'valid'):
        path = args.features/(name+'.npy')
        if sha256(path)!=fm['artifacts'][name]['sha256']:
            raise ValueError('Current feature producer drift')
        bank[name] = np.load(path, mmap_mode='r')
    bank['frame_of_proposal'] = np.repeat(frames,np.diff(offsets))
    if (len(bank['embedding'])!=len(boxes) or bank['embedding'].shape[1]!=384
            or fm['completed_frames']!=len(frames) or not np.array_equal(frames,np.arange(1,len(frames)+1))):
        raise ValueError('Every native frame/proposal/empty state must be preserved')
    if args.model is None or args.context_identity or args.memory!='fifo':
        raise ValueError('Exact registered full-set model/frozenFIFO reference required')
    model_manifest=json.loads((args.model/'manifest.json').read_text());guard=model_manifest['guard'];trained=model_manifest['config']
    if (model_manifest['status']!='COMPLETE' or args.video not in trained['inner_videos']+trained['outer_videos']
            or args.video in trained['fit_videos'] or fm['encoder']['trained_encoder'] is not None
            or fm['encoder']['coordinate_kind']!='frozen_DINO_CLS384'
            or sha256(args.model/'final.pt')!=model_manifest['checkpoint_sha256']
            or any(sha256(p)!=h for p,h in guard['source_sha256'].items())):
        raise ValueError('Exact matched-source/excluded deployment/head/data guards required')
    spec=trained['model_specs'][guard['mode']]
    model=DictionarySeparatedJoint(trained['dimension'],trained['hidden'],trained['heads'],trained['capacity'],**spec).eval()
    model.load_state_dict(torch.load(args.model/'final.pt',map_location='cpu',weights_only=False)['model'])
    torch.set_num_threads(2)
    BaseTrack._count = 0
    ordinary = RecentIdentityControl(args.memory, config['history_length'], config['maximum_history_age_frames'])
    tracker = PredictiveAssociationTracker(SimpleNamespace(track_thresh=config['track_thresh'],
        track_buffer=config['track_buffer'],match_thresh=config['match_thresh'],mot20=config['mot20']),
        frame_rate=config['frame_rate'],predictor=ordinary,memory='predictor',history_view='recent',
        appearance_weight=config['appearance_weight'],proximity_iou=config['proximity_iou'],ema=.9)
    tracker=install_dictionary_joint(tracker,model,seed=0)
    args.output.mkdir(parents=True)
    report = dict(status='RUNNING',config=config,config_sha256=sha256(args.config),video=args.video,
        memory=args.memory,context_identity=args.context_identity,feature_manifest_sha256=sha256(args.features/'manifest.json'),
        model_manifest_sha256=sha256(args.model/'manifest.json') if args.model else None,
        model_checkpoint_sha256=model_manifest['checkpoint_sha256'] if model_manifest else None,
        detection_sha256=sha256(dp),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources=snapshot_sources(args.output/'sources',[__file__,args.config,'cfp/native_local_identity.py',
            'cfp/dictionary_separated_joint.py','cfp/native_dictionary_joint.py','cfp/joint_identity_decoder.py','cfp/centered_joint_identity.py','cfp/predictive_association.py',
            'cfp/association.py','cfp/predictive.py','cfp/provenance.py']),scope=trained['scope']+' Fullroute scoresallconfirmed/lost highstage sets withprivateNULL; unchangedlow/unconfirmed/lifecycle/output/writes. No GT,trainingtarget/subsetaugmentation,teacher or future enters inference. CacheProtocolA existingV3 globalunpad3-tail proposal source; not strictcurrentpad8 rawFPS.')
    write_json(args.output/'manifest.json',report)
    timings, rows = [], 0
    output_path = args.output/(args.video+'.txt')
    try:
        with output_path.open('w') as stream:
            for index, frame in enumerate(frames):
                lo,hi = offsets[index:index+2]
                start=time.perf_counter()
                targets=tracker.update(np.c_[boxes[lo:hi].copy(),scores[lo:hi]],(1,1),(1,1),
                    np.array(bank['embedding'][lo:hi],copy=True),proposal_indices=np.arange(lo,hi))
                for track in targets:
                    if np.prod(track.tlwh[2:])<=100:
                        continue
                    proposal=track.current_proposal_index
                    if track.frame_id!=frame or not lo<=proposal<hi:
                        raise ValueError('Confirmed output must correspond to actual current assigned measurement')
                    measured=boxes[proposal].copy();measured[2:]-=measured[:2]
                    stream.write(f'{frame},{int(track.track_id)},{measured[0]:.3f},{measured[1]:.3f},{measured[2]:.3f},{measured[3]:.3f},{float(track.score):.6f},-1,-1,-1\n')
                    rows+=1
                timings.append(time.perf_counter()-start)
                if (index+1)%256==0:
                    report.update(completed_frames=index+1,output_rows=rows)
                    write_json(args.output/'manifest.json',report)
                    if shutil.disk_usage(args.output).free<40*2**30:
                        raise OSError('40GiBreserve')
        report.update(status='COMPLETE_NATIVE_DICTIONARY_JOINT_REPLAY',frames=len(timings),completed_frames=len(timings),
            output_rows=rows,prediction_sha256=sha256(output_path),association_seconds=sum(timings),
            association_latency_p50_p95_ms=(1000*np.percentile(timings,[50,95])).tolist(),
            identity_counts=tracker.decision_counts,model_counts=getattr(tracker,'joint_counts',None),
            actual_native_history_source=True,online_GT_teacher_future_used=False)
        write_json(args.output/'manifest.json',report)
        print(json.dumps(dict(video=args.video,frames=len(timings),rows=rows,seconds=sum(timings))),flush=True)
    except Exception as error:
        report.update(status='FAILED_ACTUAL_EXECUTION',error=str(error))
        write_json(args.output/'manifest.json',report)
        raise


if __name__=='__main__':
    main()
