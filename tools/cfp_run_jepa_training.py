#!/usr/bin/env python3
"""Bounded explicit-GPU training dispatcher; preserve failed/incomplete runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys

from cfp.provenance import sha256, snapshot_sources, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan['state'] != 'FROZEN' or not 1 <= len(plan['physical_gpus']) <= 4 or len(set(plan['physical_gpus'])) != len(plan['physical_gpus']):
        raise ValueError('Explicit bounded physical GPU plan required')
    record = dict(status='RUNNING', plan_sha256=sha256(args.plan), runs={},
                  source_snapshots=snapshot_sources(args.output.parent / (args.output.stem+'_sources'), [__file__,args.plan]+sorted({spec.get('trainer','tools/cfp_train_jepa.py') for spec in plan['jobs']})))
    if args.output.exists() and json.loads(args.output.read_text())['plan_sha256'] != record['plan_sha256']:
        raise ValueError('Resume plan drift')
    write_json(args.output,record)
    free_gpus=queue.Queue()
    for gpu in plan['physical_gpus']:free_gpus.put(gpu)

    def job(spec):
        gpu=free_gpus.get()
        try:
            output=Path(spec['output'])
            trainer=spec.get('trainer','tools/cfp_train_jepa.py')
            if trainer not in ('tools/cfp_train_jepa.py','tools/cfp_train_jepa_v3.py'):raise ValueError('Explicit audited trainer required')
            command=[sys.executable,trainer,'--queries',spec['queries'],'--config',spec['config'],'--variant',spec['variant'],'--seed',str(spec['seed']),'--output',str(output)]
            if not output.exists():
                if shutil.disk_usage(output.parent).free < 40*2**30:raise OSError('40GiB reserve')
                with output.with_name(output.name+'.log').open('w') as log:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH='.',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'))
                if result.returncode:raise RuntimeError('Preserved failed training log: '+str(output))
            run=json.loads((output/'manifest.json').read_text())
            if run['status']!='COMPLETE' or run['guard']['config_sha256']!=sha256(spec['config']) or run['guard']['query_manifest_sha256']!=sha256(Path(spec['queries'])/'manifest.json') or run['guard']['variant']!=spec['variant'] or run['guard']['seed']!=spec['seed']:
                raise ValueError('Incomplete/incompatible model; do notoverwrite, repair/resume explicitly: '+str(output))
            if sha256(output/'inference_epoch8.pt')!=run['inference_checkpoint_sha256']:raise ValueError('Studentexport changed')
            trainer_key=str(Path(trainer).resolve())
            if run['guard']['sources'][trainer_key]!=sha256(trainer):raise ValueError('Trainer source drift')
            return output.name,dict(command=command,gpu=gpu,manifest_sha256=sha256(output/'manifest.json'),inference_checkpoint_sha256=run['inference_checkpoint_sha256'],diagnostics=run['diagnostics'],ema_updates=run['ema_updates'])
        finally:free_gpus.put(gpu)
    try:
        with ThreadPoolExecutor(max_workers=len(plan['physical_gpus'])) as pool:
            futures=[pool.submit(job,spec) for spec in plan['jobs']]
            for future in as_completed(futures):
                name,value=future.result();record['runs'][name]=value;write_json(args.output,record)
                print('Complete',len(record['runs']),'/',len(plan['jobs']),name,'EMAupdates',value['ema_updates'],flush=True)
        record['status']='COMPLETE';write_json(args.output,record)
    except Exception as error:
        record.update(status='FAILED',error=str(error));write_json(args.output,record);raise


if __name__=='__main__':main()
