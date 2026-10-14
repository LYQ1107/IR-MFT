#!/usr/bin/env python3
"""Actual full-model/optimizer/scaler/three sampler and global RNG comparison."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser()
    for name in ['continuous','restarted','launch-proof','output']:ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise ValueError('New actualrestart proof required')
    runs=[torch.load(p/'.scratch-last.pt',map_location='cpu',weights_only=False) for p in [a.continuous,a.restarted]];reports=[json.loads((p/'manifest.json').read_text()) for p in [a.continuous,a.restarted]]
    if any(r['status']!='COMPLETE_UNLABELED_VISIBLE_TOKEN_PRETRAINING' for r in reports) or runs[0]['guard']!=runs[1]['guard'] or runs[0]['guard']['mode']!='masked_visible_tokens' or runs[0]['guard']['seed']!=0 or runs[0]['epoch']!=8 or runs[1]['epoch']!=8:raise ValueError('Sameexactsource/config/GPU actualmaskedseed0 final8 runs required')
    launch=json.loads(a.launch_proof.read_text());events=[[e for e in launch['launches'] if e['output']==str(p)] for p in [a.continuous,a.restarted]]
    if [e['phase'] for e in events[0]]!=['continuous'] or [e['phase'] for e in events[1]]!=['stop','resume'] or len({e['pid'] for es in events for e in es})!=3 or any(e['exit_code'] or sha256(e['log'])!=e['log_sha256'] for es in events for e in es):raise ValueError('Distinctsuccessful realprocesses/logs required')
    counts=dict(tensor_leaves=0,array_leaves=0,scalar_leaves=0);differences=[]
    def compare(x,y,path):
        if type(x) is not type(y):differences.append(path+':type')
        elif isinstance(x,torch.Tensor):
            counts['tensor_leaves']+=1
            if x.shape!=y.shape or x.dtype!=y.dtype or not torch.equal(x,y):differences.append(path+':tensor')
        elif isinstance(x,np.ndarray):
            counts['array_leaves']+=1
            if x.dtype!=y.dtype or not np.array_equal(x,y):differences.append(path+':array')
        elif isinstance(x,dict):
            if x.keys()!=y.keys():differences.append(path+':keys')
            for k in x.keys()&y.keys():compare(x[k],y[k],path+'/'+str(k))
        elif isinstance(x,(list,tuple)):
            if len(x)!=len(y):differences.append(path+':len')
            for i,(v,w) in enumerate(zip(x,y)):compare(v,w,path+'/'+str(i))
        else:
            counts['scalar_leaves']+=1
            if x!=y:differences.append(path+':value')
    fields=['model','optimizer','scheduler','scaler','rng_sampler','rng_augmentation','rng_mask','rng_python','rng_numpy','rng_torch','rng_cuda']
    for field in fields:compare(runs[0][field],runs[1][field],field)
    for index,(x,y) in enumerate(zip(reports[0]['epochs'],reports[1]['epochs'])):
        for field in ['epoch','loss_mean','sampled_input_and_queries_sha256','attempted_steps','amp_overflow_skips','optimizer_updates','visible_encoder_patch_min_median_max']:compare(x[field],y[field],f'epoch{index}/{field}')
    result=dict(status='EXACT_ACTUAL_VISIBLE_TOKEN_PRETRAIN_RESTART_VERIFIED' if not differences else 'ACTUAL_PRETRAIN_RESTART_DIFFERENCES',counts=counts,differences=differences,compared_fields=fields,continuous=str(a.continuous),restarted=str(a.restarted),launch_proof_sha256=sha256(a.launch_proof),actual_launches=events,guard=runs[0]['guard'],sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__]),scope='Realmaskedseed0 continuous8epochs versusdeliberateepoch2terminal/newprocessresume; fullmodel/decoder/optimizers/scaler/sampler/augmentation/mask/globalRNG/alllosses compared. Nototherhardware oridentity/MOT/novelty claim.');write_json(a.output,result);print(json.dumps(dict(status=result['status'],counts=counts)))
    if differences:raise ValueError('Actualfreshprocessrestart notexact')

if __name__=='__main__':main()
