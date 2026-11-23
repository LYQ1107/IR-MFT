#!/usr/bin/env python3
"""Require full primary parity before the matched ROI rescue factorial."""
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
    main=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(main['work_root'])
    plan_path=ROOT/'configs/perceptual_roi_confirmed_recovery_20261123.json'
    path=work/'roi_confirmed_recovery_dev_state.json';lock=path.with_suffix('.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if path.exists():raise ValueError('ROI rescue Dev already registered')
    selection=json.loads((ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_PAIR_SELECTION.json').read_text())
    selected={(r['frontend'],r['seed']):r['selected']['Dev_metrics']for r in selection['rows']if r['cue']=='rgb'}
    folder=ROOT/'configs/perceptual_roi_confirmed_recovery_dev_20261123';folder.mkdir(exist_ok=False)
    jobs={}
    def register(front,primary,cue,seed,zero):
        config=json.loads((ROOT/f'configs/perceptual_pair_selected_20261123/K_{front}_rgb_s{seed}.json').read_text())
        aux_seed=(seed+1)%3
        auxiliary=json.loads((ROOT/f'configs/perceptual_pair_selected_20261123/K_{front}_{cue}_s{aux_seed}.json').read_text())
        name=f'RC_{front}_{primary}_{cue}_s{seed}'+('_zero'if zero else'')
        config.update(id=name,backend='ROI_confirmed_recovery',primary_association=primary,auxiliary_cue=cue,
            auxiliary_seed=aux_seed,auxiliary_checkpoint=auxiliary['association_checkpoint'],recovery_plan_sha256=sha256(plan_path),
            weak_recovery=True,weak_recovery_calibrated=True,weak_allow_appearance=False,weak_minimum_score=.1,
            weak_recovery_minimum=.8,weak_minimum_margin=.1,disable_ROI_confirmation=zero,
            registration='Complete original graph and scores; candidate ROI confidence is post-primary confirmation only. Independent cyclic-seed RGB/IR extra heads have matched fitting budget; no Test selection.')
        if primary=='byte':config.update(association_weight=0.,global_birth_threshold=.7,preserve_distinct_objects=False)
        output=folder/(name+'.json');dump(output,config)
        parent=(work/'screen'/('C01_new_ByteTrack'if front=='new'else'C00_rgb_ByteTrack')/'metrics.json'
                if primary=='byte'else Path(selected[(front,seed)]))
        jobs[name]=dict(status='PENDING',config=str(output),sha256=sha256(output),zero=zero,
                        primary_reference=str(parent)if zero else None)
    for front in ['new','rgb']:
        register(front,'byte','rgb',0,True)
        for seed in range(3):register(front,'fitted','rgb',seed,True)
    for front in ['new','rgb']:
        for primary in ['byte','fitted']:
            for cue in ['rgb','ir']:
                for seed in range(3):register(front,primary,cue,seed,False)
    state=dict(status='VERIFYING_ALL_PRIMARY_PARITIES',jobs=jobs,Test_read=False,plan_sha256=sha256(plan_path))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MFT26IR_ROOT=main['data_root'])
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps',
        '/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages','/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    active={};parity_checked=False
    while True:
        for key,proc in list(active.items()):
            code=proc.poll()
            if code is not None:jobs[key].update(status='COMPLETE'if code==0 else'FAILED',exit_code=code);del active[key]
        references=[job for job in jobs.values()if job['zero']]
        if not parity_checked and all(job['status']=='COMPLETE'for job in references):
            proof=[]
            for key,job in jobs.items():
                if not job['zero']:continue
                parent=Path(job['primary_reference']);current=work/'roi_confirmed_recovery_dev'/key/'metrics.json'
                expected=json.loads(parent.read_text());actual=json.loads(current.read_text())
                if actual['trackeval']!=expected['trackeval']:raise ValueError('All primary metric fields must be exact before new recovery')
                for sequence in ['BT-005','SN-013']:
                    first=current.parent/'track_results'/(sequence+'.txt');second=parent.parent/'track_results'/(sequence+'.txt')
                    if sha256(first)!=sha256(second):raise ValueError('Original primary track files differ')
                proof.append(dict(id=key,all39_metrics_exact=True,both_track_files_exact=True,parent_metrics_sha256=sha256(parent),metrics_sha256=sha256(current)))
            parity_checked=True;state['status']='RUNNING_MATCHED_RECOVERY_FACTORIAL'
            dump(work/'roi_confirmed_recovery_primary_parity.json',dict(status='ALL8_PRIMARY_CONTROLS_EXACT',rows=proof,Test_read=False))
        for key,job in jobs.items():
            if len(active)>=2 or any(j['status']=='FAILED'for j in jobs.values()):break
            if job['status']!='PENDING'or(not job['zero']and not parity_checked):continue
            if sha256(job['config'])!=job['sha256']or sha256(plan_path)!=state['plan_sha256']:raise ValueError('Registered recovery changed')
            command=[sys.executable,'-u','tools/replay_roi_confirmed_recovery.py','--config',job['config'],
                '--output',str(work/'roi_confirmed_recovery_dev'/key)]
            log=work/(key+'.roi_confirmed_recovery.log')
            with log.open('ab',buffering=0)as handle:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[key]=proc;job.update(status='RUNNING',pid=proc.pid,command=command,log=str(log))
        if not active:
            if all(j['status']=='COMPLETE'for j in jobs.values()):state['status']='COMPLETE'
            elif any(j['status']=='FAILED'for j in jobs.values()):state['status']='FAILED'
        dump(path,state)
        if state['status']in{'COMPLETE','FAILED'}:break
        time.sleep(5)


if __name__=='__main__':main()
