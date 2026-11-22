#!/usr/bin/env python3
"""Recover exact proposal indices for the existing immutable Train pair bank."""
import argparse
import configparser
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.extract_association_train_pairs import proposals
from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.tracker.matching import ious,linear_assignment


def main():
    p=argparse.ArgumentParser();p.add_argument('--sequence',required=True);a=p.parse_args()
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root'])
    if a.sequence not in protocol['train']:raise ValueError('Train-only reconstruction')
    config_path=ROOT/'configs/perceptual_native_candidates_20261123.json';source_plan=ROOT/'configs/perceptual_pair_calibration_20261123.json';config=json.loads(source_plan.read_text())
    directory=w/'native_candidate_index'/a.sequence;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():raise ValueError('Index already reconstructed')
    source=w/'association_train'/a.sequence/'pairs.npz';meta=json.loads((source.parent/'complete.json').read_text())
    if meta['status']!='COMPLETE_TRAIN_ONLY' or meta['Dev_rows'] or meta['Test_rows'] or sha256(source)!=meta['bank_sha256'] or meta['config_sha256']!=sha256(source_plan):raise ValueError('Immutable pair source mismatch')
    gt_root=Path(plan['evaluation']['GT_root'])/'train'/a.sequence;gt_path=gt_root/'gt/gt.txt'
    if sha256(gt_path)!=protocol['derived_gt_sha256']['train/'+a.sequence]:raise ValueError('Train GT changed')
    gt=np.loadtxt(gt_path,delimiter=',',ndmin=2);gt=gt[(gt[:,6]>0)&(gt[:,7]==1)]
    ini=configparser.ConfigParser();ini.read(gt_root/'seqinfo.ini');length=ini.getint('Sequence','seqLength');width=ini.getint('Sequence','imWidth');height=ini.getint('Sequence','imHeight')
    frames=np.unique(np.linspace(1,length-8,config['source_frames_per_train_video'],dtype=int));rng=np.random.default_rng(20261123+protocol['train'].index(a.sequence))
    observations={k:[] for k in ['boxes','frame','label','stream']};edges={k:[] for k in ['source','target','query','label','lag']};n_observations=0;n_queries=0
    def load(frame,dropout):
        nonlocal n_observations
        result={};group=gt[gt[:,0]==frame];truth=group[:,2:6].copy();truth[:,2:]+=truth[:,:2]
        for stream,sigma in enumerate([.03,.08]):
            boxes=proposals(group,rng,sigma,width,height,dropout);labels=np.full(len(boxes),-1,np.int32)
            matches,_,_=linear_assignment(1-ious(truth,boxes),.5)
            for gi,pi in matches:labels[pi]=int(group[gi,1])
            indices=np.arange(n_observations,n_observations+len(boxes),dtype=np.int32);n_observations+=len(boxes)
            for key,value in [('boxes',boxes),('frame',np.full(len(boxes),frame,np.int32)),('label',labels),('stream',np.full(len(boxes),stream,np.uint8))]:observations[key].append(value)
            result[stream]=dict(boxes=boxes,label=labels,index=indices)
        return result
    for frame in frames:
        previous=load(int(frame),0.)
        for lag in config['lags']:
            current=load(int(frame)+lag,config['target_dropout'])
            for stream,past in previous.items():
                target=current[stream];keep=past['label']>=0;n,m=int(keep.sum()),len(target['boxes'])
                label=(past['label'][keep,None]==target['label'][None])&(target['label'][None]>=0)
                query=np.repeat(np.arange(n_queries,n_queries+n,dtype=np.int64),m);n_queries+=n
                for key,value in [('source',np.repeat(past['index'][keep],m)),('target',np.tile(target['index'],n)),('query',query),('label',label.ravel()),('lag',np.full(n*m,lag,np.uint8))]:edges[key].append(value)
    obs={key:np.concatenate(value) for key,value in observations.items()};pairs={key:np.concatenate(value) for key,value in edges.items()}
    with np.load(source) as old:
        for key in ['query','label','lag']:np.testing.assert_array_equal(pairs[key],old[key])
        # Check every saved geometric input, not merely matching label counts.
        for lo in range(0,len(pairs['source']),65536):
            first=obs['boxes'][pairs['source'][lo:lo+65536]].astype(np.float32);second=obs['boxes'][pairs['target'][lo:lo+65536]].astype(np.float32)
            wa=np.maximum(first[:,2:]-first[:,:2],1);wb=np.maximum(second[:,2:]-second[:,:2],1)
            relative=((second[:,:2]+second[:,2:])/2-(first[:,:2]+first[:,2:])/2)/np.maximum((wa+wb)/2,1)
            size=np.log(wb/wa).clip(-3,3);intersection=np.maximum(np.minimum(first[:,2:],second[:,2:])-np.maximum(first[:,:2],second[:,:2]),0).prod(-1)
            overlap=intersection/np.maximum(wa.prod(-1)+wb.prod(-1)-intersection,1)
            geometry=np.c_[overlap,relative[:,0].clip(-10,10),relative[:,1].clip(-10,10),size,pairs['lag'][lo:lo+65536]/8].astype(np.float16)
            np.testing.assert_array_equal(geometry,old['base'][lo:lo+65536,:6])
    saved={**{key:value for key,value in obs.items()},**{'edge_'+key:value for key,value in pairs.items()}}
    path=directory/'index.npz';temporary=directory/'index.tmp.npz';np.savez_compressed(temporary,**saved);os.replace(temporary,path)
    dump(directory/'complete.json',dict(status='COMPLETE_EXACT_TRAIN_INDEX',sequence=a.sequence,observations=n_observations,queries=n_queries,pair_rows=len(pairs['label']),positive_rows=int(pairs['label'].sum()),query_label_lag_exact=True,every_saved_geometry_field_exact=True,source_bank_sha256=sha256(source),source_plan_sha256=sha256(source_plan),config_sha256=sha256(config_path),GT_sha256=sha256(gt_path),index_sha256=sha256(path),bytes=path.stat().st_size,raw_candidate_pixel_bytes=n_observations*128*256*4,Dev_rows=0,Test_rows=0,source_sha256=sha256(__file__),proposal_source_sha256=sha256(ROOT/'tools/extract_association_train_pairs.py')))
    print(json.dumps(dict(sequence=a.sequence,observations=n_observations,pairs=len(pairs['label']),all_query_label_lag_geometry_exact=True)),flush=True)


if __name__=='__main__':main()
