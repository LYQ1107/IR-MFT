#!/usr/bin/env python3
"""Separate unlabeled point production from posthoc two-video Dev diagnosis."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.tracker.foreground_transport import frame_correspondences
from yolox.tracker.semantic_transport import semantic_groups,semantic_votes,VARIANTS
from yolox.tracker.matching import ious,linear_assignment

ALL_VARIANTS=('rgb','bbox','original_joint')+VARIANTS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sequence',choices=['BT-005','SN-013'])
    p.add_argument('--audit',action='store_true')
    a=p.parse_args()
    main_path=ROOT/'configs/perceptual_mft_20261122.json'
    plan=json.loads(main_path.read_text())
    config_path=ROOT/'configs/perceptual_semantic_transport_20261123.json'
    config=json.loads(config_path.read_text())
    work=Path(plan['work_root'])
    protocol=json.loads((ROOT/plan['protocol']).read_text())
    base=work/'semantic_point_audit'
    if a.audit:
        evaluate(base,plan,protocol,config_path)
        return
    if a.sequence is None:
        raise ValueError('Choose one unlabeled producer or the complete audit')
    cv2.setNumThreads(1)
    out=base/a.sequence
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        raise ValueError('Semantic point producer already complete')
    streams={}
    for front,prefix in [('new','new_diffv2ir'),('rgb','rgb')]:
        path=Path(plan['frozen_detection_sources'][prefix+'_dev'])/'detections'/(a.sequence+'.npz')
        with np.load(path) as z:
            streams[front]={k:z[k] for k in ['frame_ids','offsets','boxes','scores']}
    length=len(streams['new']['frame_ids'])
    if not np.array_equal(streams['new']['frame_ids'],streams['rgb']['frame_ids']):
        raise ValueError('Control frame coverage mismatch')
    source_frames=np.arange(1,length-max(config['lags'])+1,config['source_frame_step'])
    arrays={front:{key:[] for key in ALL_VARIANTS} for front in streams}
    index={front:dict(source_frame=[],lag=[],edge_offsets=[0],n_source=[],n_target=[]) for front in streams}
    stats={front:{key:dict(native_points=0,IR_fallback_points=0,RGB_point_successes=0,IR_point_successes=0) for key in VARIANTS} for front in streams}
    def load(frame):
        rgb_path=Path(plan['data_root'])/'train'/a.sequence/'visible'/f'{frame:06d}.jpg'
        ir_path=Path(plan['generated_ir_root'])/'dev'/a.sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
        rgb,ir=cv2.imread(str(rgb_path)),cv2.imread(str(ir_path),-1)
        if rgb is None or ir is None or ir.ndim!=2 or ir.dtype!=np.uint8 or rgb.shape[:2]!=ir.shape:
            raise ValueError('Actual RGB/November IR required')
        scale=np.array([960/rgb.shape[1],540/rgb.shape[0]]*2,np.float32)
        rgb=cv2.resize(rgb,(960,540),interpolation=cv2.INTER_AREA)
        ir=cv2.resize(ir,(960,540),interpolation=cv2.INTER_AREA)
        regions={}
        for front,data in streams.items():
            lo,hi=data['offsets'][frame-1:frame+1]
            regions[front]=semantic_groups(rgb,ir,data['boxes'][lo:hi]*scale,
                                           maximum=config['maximum_points_per_body'])
        return dict(rgb=rgb,ir=ir,regions=regions)
    started=time.perf_counter()
    for position,frame in enumerate(source_frames):
        past=load(int(frame))
        for lag in config['lags']:
            current=load(int(frame)+lag)
            original=frame_correspondences(past['rgb'],current['rgb'],past['regions'],current['regions'])
            for front in streams:
                native=semantic_votes(past['rgb'],current['rgb'],past['ir'],current['ir'],
                                      past['regions'][front],current['regions'][front])
                values={key:original[(front,mode,'rgb')][0] for key,mode in
                        [('rgb','rgb'),('bbox','bbox'),('original_joint','joint')]}
                values.update({key:native[key]['matrix'] for key in VARIANTS})
                n,m=values['rgb'].shape
                for key in ALL_VARIANTS:
                    arrays[front][key].append(values[key].astype(np.float32).ravel())
                for key in VARIANTS:
                    for field in stats[front][key]:stats[front][key][field]+=native[key][field]
                meta=index[front]
                meta['source_frame'].append(int(frame));meta['lag'].append(lag)
                meta['n_source'].append(n);meta['n_target'].append(m)
                meta['edge_offsets'].append(meta['edge_offsets'][-1]+n*m)
        if (position+1)%8==0 or position+1==len(source_frames):
            progress=dict(source_frames=position+1,total_source_frames=len(source_frames),seconds=time.perf_counter()-started,
                          GT_read=False,Test_read=False)
            dump(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    outputs={}
    for front in streams:
        path=out/(front+'.npz')
        np.savez_compressed(path,**{key:np.concatenate(values) for key,values in arrays[front].items()},
                            **{key:np.array(value,np.int64) for key,value in index[front].items()})
        outputs[front]=dict(path=str(path),sha256=sha256(path))
    dump(out/'complete.json',dict(status='COMPLETE_UNLABELED_SPARSE_POINT_PRODUCER',sequence=a.sequence,GT_read=False,
        Test_read=False,config_sha256=sha256(config_path),main_plan_sha256=sha256(main_path),outputs=outputs,
        statistics=stats,source_frames=len(source_frames),pairs=len(source_frames)*len(config['lags']),
        source_sha256=sha256(__file__),semantic_source_sha256=sha256(ROOT/'yolox/tracker/semantic_transport.py'),
        seconds=time.perf_counter()-started,CPU_only=True,future_frame_input=False,
        causality='Each pair consumes declared past image t-lag and current image t; no earlier prediction is rewritten.'))


def evaluate(base,plan,protocol,config_path):
    manifests={sequence:json.loads((base/sequence/'complete.json').read_text()) for sequence in protocol['dev']}
    for manifest in manifests.values():
        if (manifest['status']!='COMPLETE_UNLABELED_SPARSE_POINT_PRODUCER' or manifest['GT_read']
                or manifest['config_sha256']!=sha256(config_path)):
            raise ValueError('Both unlabeled producers must finish before any Dev GT is read')
    rows=[]
    for sequence in protocol['dev']:
        gt_path=Path(plan['evaluation']['GT_root'])/'train'/sequence/'gt/gt.txt'
        truth=np.loadtxt(gt_path,delimiter=',',ndmin=2)
        for front,prefix in [('new','new_diffv2ir'),('rgb','rgb')]:
            source=Path(plan['frozen_detection_sources'][prefix+'_dev'])/'detections'/(sequence+'.npz')
            with np.load(source) as z:boxes,offsets=z['boxes'],z['offsets']
            labels={}
            def frame_labels(frame):
                if frame in labels:return labels[frame]
                lo,hi=offsets[frame-1:frame+1];b=boxes[lo:hi]
                gt=truth[(truth[:,0]==frame)&(truth[:,6]>0)&(truth[:,7]==1)]
                gb=gt[:,2:6].copy();gb[:,2:]+=gb[:,:2]
                label=np.full(len(b),-1,np.int64)
                matches,_,_=linear_assignment(1-ious(gb,b),.5)
                for gi,pi in matches:label[pi]=int(gt[gi,1])
                ignore=truth[(truth[:,0]==frame)&(truth[:,7]==8),2:6].copy();ignore[:,2:]+=ignore[:,:2]
                if len(ignore):label[(label<0)&(ious(b,ignore).max(1)>.2)]=-2
                labels[frame]=label;return label
            info=manifests[sequence]['outputs'][front];path=Path(info['path'])
            if sha256(path)!=info['sha256']:raise ValueError('Produced point data changed')
            with np.load(path) as z:data={key:z[key] for key in z.files}
            for lag in [1,2,4,8]:
                evaluation={key:dict(y=[],score=[],nonnull=0,correct=0,accepted=0,null=0,null_accept=0,
                                     low_RGB=0,low_RGB_correct=0) for key in ALL_VARIANTS}
                for position in np.flatnonzero(data['lag']==lag):
                    frame=int(data['source_frame'][position]);n,m=int(data['n_source'][position]),int(data['n_target'][position])
                    lo,hi=data['edge_offsets'][position:position+2]
                    past,current=frame_labels(frame),frame_labels(frame+lag)
                    keep=past>=0
                    y=(past[:,None]==current[None])&(past[:,None]>=0)&(current[None]>=0)
                    valid=np.broadcast_to(keep[:,None],(n,m))&(current[None]!=-2)
                    has=y.any(1)&keep
                    reference=data['rgb'][lo:hi].reshape(n,m)
                    low=(reference.max(1)<.3)&has if m else has
                    for key in ALL_VARIANTS:
                        score=data[key][lo:hi].reshape(n,m);e=evaluation[key]
                        e['y'].append(y[valid]);e['score'].append(score[valid])
                        e['nonnull']+=int(has.sum());e['null']+=int((keep&~has).sum());e['low_RGB']+=int(low.sum())
                        if not m:continue
                        predicted=score.argmax(1);accepted=(score.max(1)>=.05)&keep
                        correct=y[np.arange(n),predicted]&accepted
                        e['correct']+=int((correct&has).sum());e['accepted']+=int(accepted.sum())
                        e['null_accept']+=int((accepted&~has).sum());e['low_RGB_correct']+=int((correct&low).sum())
                for key,e in evaluation.items():
                    y=np.concatenate(e.pop('y'));score=np.concatenate(e.pop('score'))
                    rows.append(dict(sequence=sequence,frontend=front,lag=lag,variant=key,**e,
                        edge_AUC=float(roc_auc_score(y,score)),top1_recall=e['correct']/max(e['nonnull'],1),
                        accepted_precision=e['correct']/max(e['accepted'],1),
                        null_false_acceptance=e['null_accept']/max(e['null'],1),
                        low_RGB_top1_recall=e['low_RGB_correct']/max(e['low_RGB'],1)))
    report=dict(status='COMPLETE_SPARSE_DEV_MECHANISM_AUDIT_NOT_HOTA',config_sha256=sha256(config_path),
        rows=rows,producers=manifests,GT_inference_input=False,Test_read=False,goal_met=False,
        supported_multimodal_innovations=0,
        scope='Sparse declared source grid across both Dev videos, all fixed detector candidates. No parameter fitting or full tracking HOTA. Both unlabeled producers completed before posthoc GT assignment.')
    dump(base/'audit.json',report)
    dump(ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_SEMANTIC_POINT_AUDIT.json',report)
    print(json.dumps(dict(status=report['status'],rows=len(rows),Test_read=False)))


if __name__=='__main__':main()
