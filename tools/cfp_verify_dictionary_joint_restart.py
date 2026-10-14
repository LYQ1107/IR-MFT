#!/usr/bin/env python3
"""Actual epoch2 exit/fresh-process resume equality of source-matched C."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    ap=argparse.ArgumentParser()
    for name in ['continuous','restarted','stop-log','resume-log','output']:ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists() or a.continuous==a.restarted:raise ValueError('New proof of distinct actual runs required')
    if 'CHECKPOINTED_EARLY_STOP' not in a.stop_log.read_text() or '"epoch": 3' not in a.resume_log.read_text():raise ValueError('Actual epoch2exit then freshresume log required')
    saved=[torch.load(p/'last.pt',map_location='cpu',weights_only=False) for p in [a.continuous,a.restarted]]
    reports=[json.loads((p/'manifest.json').read_text()) for p in [a.continuous,a.restarted]]
    if any(d['status']!='COMPLETE' for d in reports) or saved[0]['guard']!=saved[1]['guard'] or saved[0]['guard']['mode']!='joint' or saved[0]['guard']['seed']!=1 or saved[0]['epoch']!=8 or saved[1]['epoch']!=8:
        raise ValueError('Exact registered completedjointseed1/source/runtime/finalepoch required')
    counts=dict(tensor_leaves=0,array_leaves=0,scalar_leaves=0);differences=[]
    def compare(x,y,path):
        if type(x)!=type(y):differences.append(path+':type')
        elif isinstance(x,torch.Tensor):
            counts['tensor_leaves']+=1
            if x.dtype!=y.dtype or x.shape!=y.shape or not torch.equal(x,y):differences.append(path+':tensor')
        elif isinstance(x,np.ndarray):
            counts['array_leaves']+=1
            if x.dtype!=y.dtype or not np.array_equal(x,y):differences.append(path+':array')
        elif isinstance(x,dict):
            if x.keys()!=y.keys():differences.append(path+':keys')
            for k in x.keys() & y.keys():compare(x[k],y[k],path+'/'+str(k))
        elif isinstance(x,(tuple,list)):
            if len(x)!=len(y):differences.append(path+':length')
            for i,(u,v) in enumerate(zip(x,y)):compare(u,v,path+'/'+str(i))
        else:
            counts['scalar_leaves']+=1
            if x!=y:differences.append(path+':scalar')
    fields=['model','optimizer','scheduler','rng_sampler','rng_codes','rng_subset','rng_numpy','rng_python','rng_torch']
    for field in fields:compare(saved[0][field],saved[1][field],field)
    for i,(x,y) in enumerate(zip(reports[0]['epochs'],reports[1]['epochs'])):
        for field in ['epoch','attempted_steps','optimizer_updates','mean_loss','objective_means','augmentation_counts','sampled_input_sha256']:compare(x[field],y[field],f'epoch{i}/{field}')
    write_json(a.output,dict(status='EXACT_DICTIONARY_JOINT_FRESH_RESTART_VERIFIED' if not differences else 'DICTIONARY_JOINT_RESTART_DIFFERS',counts=counts,differences=differences,
        compared_fields=fields,continuous=str(a.continuous),restarted=str(a.restarted),guard=saved[0]['guard'],
        stop_log_sha256=sha256(a.stop_log),resume_log_sha256=sha256(a.resume_log),
        sources=snapshot_sources(a.output.with_suffix('.sources'),[__file__]),
        scope='Actual separate CLIprocess epoch2exit/freshresume versus continuousjointseed1 eightCPUepochs. Fullmodel/optimizer/scheduler/sampler/codes/subset/globalNumPy/Python/Torch RNG andallloss/augmentation/inputdigests compared bitexact. Walltime/PID/outputpaths excluded; no crosshardware/allmodel proof.'))
    print(json.dumps(dict(differences=len(differences),**counts)))
    if differences:raise ValueError('Actual restarted state differs')


if __name__=='__main__':main()
