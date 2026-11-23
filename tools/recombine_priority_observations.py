#!/usr/bin/env python3
"""Use exact frozen ROI observations; preserve global coordinates and IDs."""
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

from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.tracker.foveated_observations import attention_windows,uniform_windows,coarse_attention as plain_attention
from yolox.tracker.foveated_regions import working_region,coarse_attention as region_attention
from yolox.tracker.foveated_causal_attention import CausalAttention
from yolox.tracker.priority_observations import priority_union
from yolox.tracker.matching import ious


def main():
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['plain','regions','causal'],required=True)
    p.add_argument('--sequence',choices=['BT-005','SN-013'],required=True);p.add_argument('--max-frames',type=int,default=0)
    a=p.parse_args();cv2.setNumThreads(1)
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(plan['work_root'])
    config_path=ROOT/'configs/perceptual_foveated_priority_20261123.json';config=json.loads(config_path.read_text())
    parent=config['parents'][a.family];parent_path=ROOT/parent['plan']
    if sha256(parent_path)!=parent['sha256']:raise ValueError('Parent attention plan changed')
    attention_config=json.loads(parent_path.read_text())
    source_dir=work/parent['directory']/'dev'/a.sequence
    manifest=json.loads((source_dir/'complete.json').read_text())
    raw_path=source_dir/'raw_roi_observations.npz'
    if (manifest['status']!='COMPLETE_FOVEATED_CURRENT_OBSERVATIONS' or manifest['GT_read'] or manifest['future_frame_input']
            or manifest['config_sha256']!=parent['sha256'] or manifest['raw_roi_sha256']!=sha256(raw_path)):
        raise ValueError('Exact complete parent raw observations required')
    with np.load(raw_path)as z:raw={key:z[key] for key in z.files}
    data={};sources={}
    for arm,prefix in [('rgb','rgb'),('new','new_diffv2ir')]:
        path=Path(plan['frozen_detection_sources'][prefix+'_dev'])/'detections'/(a.sequence+'.npz')
        with np.load(path)as z:data[arm]={key:z[key]for key in ['frame_ids','offsets','boxes','scores']}
        sources[arm]=dict(path=str(path),sha256=sha256(path))
    np.testing.assert_array_equal(data['rgb']['frame_ids'],data['new']['frame_ids'])
    limit=min(len(data['rgb']['frame_ids']),a.max_frames or len(data['rgb']['frame_ids']))
    out=work/('foveated_priority_'+a.family+'_proposals')/'dev'/a.sequence
    if a.max_frames:out=out/f'smoke_{a.max_frames}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise ValueError('Completed priority observations exist')
    scales=attention_config['attention_scales'];modes=attention_config.get('attention_modes',['rgb','ir'])
    policies=[(cue+'_'+scale,count)for cue in modes for scale in scales for count in [2,4]]+[('uniform',4)]
    buffers={f'{arm}_{cue}_k{count}_priority':[]for arm in ['rgb','new']for cue,count in policies}
    indices={key:[]for key in buffers};statistics={key:dict(promoted_weak=0,added=0,original_observations=0)for key in buffers}
    state=CausalAttention()if a.family=='causal'else None;started=time.perf_counter()
    for frame in range(1,limit+1):
        rgb_path=Path(plan['data_root'])/'train'/a.sequence/'visible'/f'{frame:06d}.jpg'
        ir_path=Path(plan['generated_ir_root'])/'dev'/a.sequence/'infrared_diffv2ir'/f'{frame:06d}.png'
        rgb,ir=cv2.imread(str(rgb_path)),cv2.imread(str(ir_path),-1)
        if rgb is None or ir is None or ir.ndim!=2 or rgb.shape[:2]!=ir.shape:raise ValueError('Current actual pair missing')
        lo,hi=data['rgb']['offsets'][frame-1:frame+1];b,s=data['rgb']['boxes'][lo:hi],data['rgb']['scores'][lo:hi]
        if state is not None:heats=state.update(frame,rgb,ir,b,s)
        elif a.family=='regions':
            support=working_region(rgb,b,s);rh=region_attention(rgb,b,s,rgb.shape,support=support)
            heats=dict(rgb=rh,ir=rh if float(ir.std())<1e-6 else region_attention(ir,b,s,rgb.shape,support=support))
        else:
            rh=plain_attention(rgb,b,s,rgb.shape)
            heats=dict(rgb=rh,ir=rh if float(ir.std())<1e-6 else plain_attention(ir,b,s,rgb.shape))
        selected={cue+'_'+scale:attention_windows(heat,rgb.shape,4,fraction=fraction)
                  for cue,heat in heats.items()for scale,fraction in scales.items()}
        selected['uniform']=uniform_windows(rgb.shape)
        first,last=np.searchsorted(raw['windows'][:,0],[frame,frame+1])
        lookup={tuple(raw['windows'][index,1:]):index for index in range(first,last)}
        for arm in ['rgb','new']:
            start,stop=data[arm]['offsets'][frame-1:frame+1]
            original=np.c_[data[arm]['boxes'][start:stop],data[arm]['scores'][start:stop]].astype(np.float32)
            for cue,count in policies:
                views=[]
                for window in selected[cue][:count]:
                    if window not in lookup:raise ValueError('Reconstructed attention window absent from exact frozen raw inference')
                    index=lookup[window];left,right=raw['offsets'][index:index+2]
                    det=np.c_[raw['boxes'][left:right],raw['scores'][left:right]].astype(np.float32)
                    overlap=np.zeros(len(det),bool)
                    if len(det)and len(original):overlap=ious(original[:,:4],det[:,:4]).max(0)>=.5
                    det=det[overlap|(det[:,4]>=.6)]
                    views.append((det,rgb.shape[1]/(window[2]-window[0])))
                combined,mapping=priority_union(original,views)
                np.testing.assert_array_equal(combined[:len(original),:4],original[:,:4])
                high=original[:,4]>=.6
                np.testing.assert_array_equal(combined[:len(original),4][high],original[high,4])
                key=f'{arm}_{cue}_k{count}_priority';buffers[key].append(combined)
                indices[key].append(np.where(mapping<0,-1,mapping+start))
                stats=statistics[key];stats['original_observations']+=len(original);stats['added']+=len(combined)-len(original)
                stats['promoted_weak']+=int((combined[:len(original),4]!=original[:,4]).sum())
        if frame%100==0 or frame==limit:
            progress=dict(frame=frame,frames=limit,seconds=time.perf_counter()-started,GT_read=False,new_detector_forwards=0)
            dump(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    outputs={}
    for key,values in buffers.items():
        merged=np.concatenate(values).reshape(-1,5);path=out/(key+'.npz');temporary=out/(key+'.tmp.npz')
        np.savez_compressed(temporary,frame_ids=np.arange(1,limit+1),offsets=np.r_[0,np.cumsum([len(value)for value in values])],
                            boxes=merged[:,:4],scores=merged[:,4],original_index=np.concatenate(indices[key]))
        os.replace(temporary,path)
        outputs[key]=dict(path=str(path),sha256=sha256(path),frames=limit,detections=len(merged),bytes=path.stat().st_size)
    dump(out/'complete.json',dict(status='SMOKE_COMPLETE'if a.max_frames else'COMPLETE_FOVEATED_CURRENT_OBSERVATIONS',
        family=a.family,sequence=a.sequence,split='dev',frames=limit,outputs=outputs,sources=sources,statistics=statistics,
        config_sha256=sha256(config_path),parent_config_sha256=parent['sha256'],raw_parent_sha256=manifest['raw_roi_sha256'],
        source_sha256=sha256(__file__),priority_source_sha256=sha256(ROOT/'yolox/tracker/priority_observations.py'),
        GT_read=False,future_frame_input=False,new_detector_forwards=0,high_global_coordinates_and_scores_immutable=True,
        original_global_distinct_objects_preserved=True,seconds=time.perf_counter()-started))


if __name__=='__main__':main()
