#!/usr/bin/env python3
"""Wait for every pilot, produce all12 snapshots, then full official Dev."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.run_pseudo_ir_comparison import dump,sha256

ROOT=Path(__file__).resolve().parents[1]


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(plan['work_root'])
    fusion_path=ROOT/'configs/perceptual_correlation_recoverability_20261123.json'
    spec_path=ROOT/'configs/perceptual_recoverable_correlation_ablation_eval_20261123.json';spec=json.loads(spec_path.read_text())
    path=work/'correlation_recoverability_ablation_dev_state.json';lock=path.with_suffix('.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if path.exists():raise ValueError('Pilot Dev already scheduled')
    producer_sha256=sha256(ROOT/'tools/extract_recoverable_correlation_ablations.py')
    for seq,gpu in [('BT-005','2'),('SN-013','3')]:
        check=json.loads((work/'correlation_recoverability_ablation_predictions/dev'/seq/'source_check_8/complete.json').read_text())
        if (check['status']!='SMOKE_COMPLETE'or check['config_sha256']!=sha256(fusion_path)or check['physical_gpu']!=gpu
                or check['source_sha256']!=producer_sha256 or len(check['outputs'])!=7
                or not all(row['array_exact']for row in check['fresh_RGB_parity'])):
            raise ValueError('Both exact frozen-RGB and zero-repair source checks required')
    folder=ROOT/'configs/perceptual_recoverable_correlation_ablation_dev_20261123';folder.mkdir(exist_ok=False)
    rows=[]
    for item in spec['models']:
        rows.append(dict(item,output_policy='priority'))
        rows.append(dict(item,id=item['id']+'_raw',output_policy='raw'))
    jobs={}
    for row in rows:
        config=dict(row,proposal_key=row['id'],producer_directory='correlation_recoverability_ablation_predictions',fusion_plan_sha256=sha256(fusion_path),specification_sha256=sha256(spec_path),
            registration='All registered seed0 IR removals and6/12/18/24 snapshots, full two-video Dev, no Test selection or contribution claim.')
        out=folder/(row['id']+'.json');dump(out,config)
        jobs[row['id']]=dict(status='PENDING',config=str(out),sha256=sha256(out))
    state=dict(status='WAITING_FOR_ALL_MATCHED_PILOT_FITS',jobs=jobs,producers={},Test_read=False,
        fusion_plan_sha256=sha256(fusion_path),specification_sha256=sha256(spec_path),producer_source_sha256=producer_sha256)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MFT26IR_ROOT=plan['data_root'])
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps',
        '/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    while True:
        fits=json.loads((work/'correlation_recoverability_ablation_fit_state.json').read_text())
        if fits['status']in{'FAILED','DEPENDENCY_FAILED'}:state['status']='DEPENDENCY_FAILED';dump(path,state);return
        if fits['status']=='COMPLETE':break
        dump(path,state);time.sleep(5)
    if sha256(fusion_path)!=state['fusion_plan_sha256']or sha256(spec_path)!=state['specification_sha256']or sha256(ROOT/'tools/extract_recoverable_correlation_ablations.py')!=producer_sha256:
        raise ValueError('Frozen evaluation source changed while waiting')
    state['status']='PRODUCING_ALL_MATCHED_DEV_SNAPSHOTS';active={}
    for seq,gpu in [('BT-005','2'),('SN-013','3')]:
        command=[sys.executable,'-u','tools/extract_recoverable_correlation_ablations.py','--sequence',seq,'--specification',str(spec_path)]
        log=work/(seq+'.correlation_recoverability_ablation_dev.log')
        with log.open('ab',buffering=0)as handle:
            proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=gpu),stdin=subprocess.DEVNULL,
                stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        active[seq]=proc;state['producers'][seq]=dict(status='RUNNING',pid=proc.pid,physical_gpu=gpu,command=command,log=str(log))
    while active:
        for key,proc in list(active.items()):
            code=proc.poll()
            if code is not None:state['producers'][key].update(status='COMPLETE'if code==0 else'FAILED',exit_code=code);del active[key]
        dump(path,state);time.sleep(5)
    if any(j['status']=='FAILED'for j in state['producers'].values()):state['status']='FAILED';dump(path,state);return
    state['status']='RUNNING'
    while True:
        for key,proc in list(active.items()):
            code=proc.poll()
            if code is not None:jobs[key].update(status='COMPLETE'if code==0 else'FAILED',exit_code=code);del active[key]
        for key,job in jobs.items():
            if len(active)>=2 or any(j['status']=='FAILED'for j in jobs.values()):break
            if job['status']!='PENDING':continue
            if sha256(job['config'])!=job['sha256']:raise ValueError('Registered Dev replay changed')
            command=[sys.executable,'-u','tools/replay_recoverable_correlation.py','--config',job['config'],
                '--output',str(work/'correlation_recoverability_ablation_dev'/key)]
            log=work/(key+'.correlation_recoverability_dev_replay.log')
            with log.open('ab',buffering=0)as handle:
                proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,
                    stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[key]=proc;job.update(status='RUNNING',pid=proc.pid,command=command,log=str(log))
        if not active:
            if all(j['status']=='COMPLETE'for j in jobs.values()):state['status']='COMPLETE'
            elif any(j['status']=='FAILED'for j in jobs.values()):state['status']='FAILED'
        dump(path,state)
        if state['status']!='RUNNING':break
        time.sleep(5)


if __name__=='__main__':main()
