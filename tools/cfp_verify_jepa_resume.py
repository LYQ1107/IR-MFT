#!/usr/bin/env python3
"""Interrupt only our own fresh trainer and verify epoch-eight exact state."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from cfp.provenance import sha256, write_json, snapshot_sources


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous proof')
    root=Path('/data3/liuyeqiang/IR-MFT_assets/cfp/phase5')
    command=[sys.executable,'tools/cfp_train_jepa.py','--queries',str(root/'recovery_train_queries_v1'),'--config','configs/cfp/phase_v_jepa_trial_v1.json','--variant','J3','--seed','0','--output',str(args.output)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH='.',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    with args.output.with_name(args.output.name+'.interrupted.log').open('w') as stream:
        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
        started=time.monotonic()
        while True:
            if child.poll() is not None:raise RuntimeError('Child exited before intended interruption')
            manifest=args.output/'manifest.json'
            if manifest.exists():
                try:m=json.loads(manifest.read_text())
                except json.JSONDecodeError:m={}
                if len(m.get('epochs',[]))>=3:
                    child.terminate();child.wait(timeout=30);break
            if time.monotonic()-started>1200:
                child.terminate();child.wait();raise TimeoutError('Only own child terminated')
            time.sleep(.05)
    interrupted=torch.load(args.output/'last.pt',map_location='cpu',weights_only=False)
    epoch=interrupted['epoch']
    with args.output.with_name(args.output.name+'.resumed.log').open('w') as stream:
        result=subprocess.run(command+['--resume',str(args.output/'last.pt')],stdout=stream,stderr=subprocess.STDOUT,env=env)
    if result.returncode:raise RuntimeError('Preserved resume failure')
    resumed=torch.load(args.output/'epoch8.pt',map_location='cpu',weights_only=False)
    reference=root/'jepa_v1_J3_seed0/epoch8.pt'
    original=torch.load(reference,map_location='cpu',weights_only=False)
    keys=['guard','model','teacher','optimizer','scaler','numpy_rng','cpu_rng','cuda_rng','python_rng','ema_updates']
    checks={key:bool(equal(original[key],resumed[key])) for key in keys}
    checks['epoch_loss_and_query_order']=all(equal({k:v for k,v in a.items() if k!='seconds'},{k:v for k,v in b.items() if k!='seconds'}) for a,b in zip(original['epochs'],resumed['epochs']))
    write_json(args.output.with_name(args.output.name+'.proof.json'),dict(status='COMPLETE' if all(checks.values()) else 'FAILED',interrupted_after_epoch=epoch,terminated_pid=child.pid,
      resumed_checkpoint_sha256=sha256(args.output/'epoch8.pt'),reference_checkpoint_sha256=sha256(reference),checks=checks,
      source_snapshots=snapshot_sources(args.output/'proof_sources',[__file__]),scope='Actual SIGTERM of our own fresh training child; full-state exact equality excludes elapsed training seconds only.'))
    if not all(checks.values()):raise AssertionError(checks)
    print('Exact resume verified',epoch,checks,flush=True)


if __name__=='__main__':main()
