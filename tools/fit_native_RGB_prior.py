#!/usr/bin/env python3
"""Fit only four ordinary RGB real/null parameters with a convex objective."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.optimize import minimize
from tools.native_query_data import load_query_group,frozen_RGB_model,RGB_logits
from tools.run_pseudo_ir_comparison import dump,sha256
from yolox.tracker.native_query_prior import query_prior_objective


def main():
    p=argparse.ArgumentParser();p.add_argument('--frontend',choices=['new','rgb'],required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();torch.set_num_threads(2)
    config_path=ROOT/'configs/perceptual_native_query_v2_20261123.json';plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());protocol=json.loads((ROOT/plan['protocol']).read_text());w=Path(plan['work_root'])
    if json.loads((w/'native_query_bank_state.json').read_text())['status']!='COMPLETE':raise ValueError('Every Train native query packet required')
    output=w/'native_RGB_prior'/f'{a.frontend}_s{a.seed}';output.mkdir(parents=True,exist_ok=True)
    if (output/'complete.json').exists():raise ValueError('Completed RGB prior fit exists')
    model,core=frozen_RGB_model(a.frontend,a.seed);groups=[];sources={};cache={};started=time.perf_counter()
    for regime in ['local','long']:
        for video,sequence in enumerate(protocol['train']):
            data=load_query_group(w,sequence,regime);original=RGB_logits(model,data);q=data['query'];counts=data['counts'];offsets=data['offsets'];target=data['target'];labels=data['label'].astype(np.float64)
            dropped=np.zeros(len(counts),bool);valid=np.ones(len(q),bool);calibrated=original
            if regime=='long':
                if np.any(data['base'][:,14:21]!=0):raise ValueError('Only absent-motion queries allow post-extraction removal')
                rng=np.random.default_rng(20261123+video);dropped=(target>=0)&(rng.random(len(target))<.15)
                rows=np.flatnonzero(dropped);removed=offsets[rows]+target[rows];valid[removed]=False;labels[removed]=0
                modified=data['base'][:,:23].copy();modified[:,22]=np.repeat(np.log1p(counts-dropped.astype(int))/4,counts)
                calibrated=RGB_logits(model,data,modified)
            objective_data=dict(core_logits=calibrated.astype(np.float64),cosine=data['base'][:,23].astype(np.float64),query=q,offsets=offsets,valid=valid,label=labels,effective_counts=counts-dropped.astype(int));groups.append(objective_data)
            name=sequence+'_'+regime;path=output/(name+'.npz');temporary=output/(name+'.tmp.npz');np.savez_compressed(temporary,core_logits=original,calibration_core_logits=calibrated,calibration_dropped_queries=dropped);os.replace(temporary,path)
            cache[name]=dict(path=str(path),sha256=sha256(path),queries=len(counts),rows=len(q),null_queries=int((np.add.reduceat(labels,offsets[:-1])==0).sum()))
            sources[name]=data['sources'];print(json.dumps(dict(frontend=a.frontend,seed=a.seed,group=name,rows=len(q),queries=len(counts),dropped=int(dropped.sum()))),flush=True)
    if len(groups)!=12:raise ValueError('Both regimes and every Train video required')
    def objective(theta):
        value=0.;gradient=np.zeros(4)
        for group in groups:
            loss,grad=query_prior_objective(theta,group);value+=loss/12;gradient+=grad/12
        return value+.0001*np.square(theta).sum(),gradient+.0002*theta
    bounds=[(0,10),(0,50),(-20,20),(-20,20)];initial=np.array([1.,10.,0.,0.]);initial_loss=objective(initial)[0]
    result=minimize(objective,initial,method='L-BFGS-B',jac=True,bounds=bounds,options=dict(maxiter=200,ftol=1e-11,gtol=1e-6,maxls=40))
    loss,gradient=objective(result.x);projected=gradient.copy()
    for i,(lo,hi) in enumerate(bounds):
        if result.x[i]<=lo+1e-7 and projected[i]>0:projected[i]=0
        if result.x[i]>=hi-1e-7 and projected[i]<0:projected[i]=0
    if not np.isfinite(loss) or loss>initial_loss+1e-6 or (not result.success and np.max(abs(projected))>1e-4):raise RuntimeError('RGB real/null calibration did not converge: '+str(result.message))
    report=dict(status='COMPLETE_TRAIN_ONLY_CONVEX_RGB_PRIOR',frontend=a.frontend,seed=a.seed,config_sha256=sha256(config_path),parameters=dict(zip(['t','a','b','c'],map(float,result.x))),initial_loss=float(initial_loss),final_loss=float(loss),projected_gradient_inf=float(abs(projected).max()),optimizer_success=bool(result.success),message=str(result.message),iterations=int(result.nit),function_evaluations=int(result.nfev),core=core,cache=cache,sources=sources,fit_sequences=protocol['train'],Dev_fit_rows=0,Test_fit_rows=0,source_sha256=sha256(__file__),objective_source_sha256=sha256(ROOT/'yolox/tracker/native_query_prior.py'),mask_policy='Original physically recomputed local dropout only; extra15% removal solely in appearance-only long queries with changed RGB candidate counts and recomputed frozen logits.',seconds=time.perf_counter()-started)
    dump(output/'complete.json',report);print(json.dumps(dict(status=report['status'],frontend=a.frontend,seed=a.seed,parameters=report['parameters'],loss=loss,gradient=report['projected_gradient_inf'])),flush=True)


if __name__=='__main__':main()
