#!/usr/bin/env python3
"""Real new-process centered joint-head restart, all train/state/metrics leaves."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from cfp.provenance import sha256,snapshot_sources,write_json


def main():
    ap=argparse.ArgumentParser()
    for name in ['continuous','restarted','stop-log','output']:ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists() or a.continuous.resolve()==a.restarted.resolve():raise ValueError('Distinct actual runs/new proof required')
    stops=[]
    for line in a.stop_log.read_text().splitlines():
        try:v=json.loads(line)
        except json.JSONDecodeError:continue
        if isinstance(v,dict) and v.get('status')=='CHECKPOINTED_EARLY_STOP':stops.append(v)
    if stops!=[dict(status='CHECKPOINTED_EARLY_STOP',epoch=2)]:raise ValueError('Actual epoch2terminal checkpoint required')
    reports=[json.loads((p/'manifest.json').read_text()) for p in [a.continuous,a.restarted]];states=[torch.load(p/'last.pt',map_location='cpu',weights_only=False) for p in [a.continuous,a.restarted]]
    if any(r['status']!='COMPLETE' for r in reports) or states[0]['guard']!=states[1]['guard'] or states[0]['epoch']!=states[1]['epoch'] or states[0]['epoch']!=8 or states[0]['guard']['architecture']!='joint' or states[0]['guard']['seed']!=1 or states[0]['guard']['device']!='cpu' or not states[0]['guard']['deterministic_algorithms']:raise ValueError('Same-source complete CPU jointseed1 protocol required')
    counts=dict(tensors=0,arrays=0,scalars=0);diff=[]
    def compare(x,y,path):
        if type(x)!=type(y):diff.append(path+':type')
        elif isinstance(x,torch.Tensor):
            counts['tensors']+=1
            if x.dtype!=y.dtype or x.shape!=y.shape or not torch.equal(x,y):diff.append(path+':tensor')
        elif isinstance(x,np.ndarray):
            counts['arrays']+=1
            if x.dtype!=y.dtype or x.shape!=y.shape or not np.array_equal(x,y):diff.append(path+':array')
        elif isinstance(x,dict):
            if x.keys()!=y.keys():diff.append(path+':keys')
            for k in x.keys()&y.keys():compare(x[k],y[k],path+'/'+str(k))
        elif isinstance(x,(list,tuple)):
            if len(x)!=len(y):diff.append(path+':length')
            for i,(v,w) in enumerate(zip(x,y)):compare(v,w,path+'/'+str(i))
        else:
            counts['scalars']+=1
            if x!=y:diff.append(path+':value')
    fields=['model','optimizer','scheduler','rng_numpy_generator','rng_numpy','rng_python','rng_torch','rng_cuda']
    for field in fields:compare(states[0][field],states[1][field],field)
    for i,(x,y) in enumerate(zip(reports[0]['epochs'],reports[1]['epochs'])):
        for key in ['epoch','mean_loss','sampled_frames','supervised_detections','optimizer_updates','sampled_input_sha256','objective_means']:compare(x[key],y[key],f'epoch{i+1}/{key}')
    for key in ['evaluations','first_gradient_norms','first_gradient_parameters']:compare(reports[0][key],reports[1][key],key)
    write_json(a.output,dict(status='EXACT_CENTERED_JOINT_RESTART_VERIFIED' if not diff else 'ACTUAL_CENTERED_JOINT_RESTART_DIFFERENCES',differences=diff,counts=counts,guard=states[0]['guard'],continuous=str(a.continuous),restarted=str(a.restarted),manifest_sha256=[sha256(p/'manifest.json') for p in [a.continuous,a.restarted]],stop_log_sha256=sha256(a.stop_log),sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__]),scope='ActualCPU jointseed1 continuous8epochs versus epoch2exit/freshresume; model/opt/sched/allRNG/digests/losses/gradients/allfixedstate identityNULLmetrics. Excludes walltime/filepaths/containerbytes; no nativegain/crosshardware guarantee.'))
    print(json.dumps(dict(differences=len(diff),counts=counts)),flush=True)
    if diff:raise RuntimeError('Actual restart not equivalent')

if __name__=='__main__':main()
