#!/usr/bin/env python3
"""Frozen post-training temporal controls on spare physical2, full Dev only."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tools.run_pseudo_ir_comparison import dump,sha256

ROOT=Path(__file__).resolve().parents[1]


def main():
    main=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(main['work_root'])
    spec_path=ROOT/'configs/perceptual_temporal_correction_20261123.json';spec=json.loads(spec_path.read_text())
    source=ROOT/'tools/extract_temporal_correction.py';path=work/'temporal_correction_dev_state.json';lock=path.with_suffix('.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if path.exists():raise ValueError('Temporal controls already scheduled')
    sources={str(p.relative_to(ROOT)):sha256(p)for p in [source,ROOT/'yolox/models/temporal_correction.py',ROOT/'yolox/tracker/physical_motion.py']}
    for seq in ['BT-005','SN-013']:
        check=json.loads((work/'temporal_correction_predictions/dev'/seq/'source_check_32/complete.json').read_text())
        if(check['status']!='SMOKE_COMPLETE'or check['frames']!=32 or not check['no_memory_reference_all_exact']
                or check['source_sha256']!=sha256(source)or check['specification_sha256']!=sha256(spec_path)
                or check['temporal_source_sha256']!=sources['yolox/models/temporal_correction.py']
                or check['motion_source_sha256']!=sources['yolox/tracker/physical_motion.py']):
            raise ValueError('All3 trained models/raw-priority real32frame exact no-memory proof required')
    folder=ROOT/'configs/perceptual_temporal_correction_dev_20261123';folder.mkdir(exist_ok=False);jobs={}
    for row in spec['models']:
        for temporal in spec['policies']:
            for output_policy in ['priority','raw']:
                key=row['id']+'_'+temporal+('_raw'if output_policy=='raw'else'')
                config=dict(row,id=key,temporal_policy=temporal,output_policy=output_policy,proposal_key=key,
                    producer_directory='temporal_correction_predictions',fusion_plan_sha256=spec['config_sha256'],specification_sha256=sha256(spec_path),
                    registration='Frozen global-best fullcue checkpoint from all50 completeCGA/CGR Dev cases; identical four temporal and two output policies acrossall3cues. No refit, Test, or contribution claim.')
                target=folder/(key+'.json');dump(target,config);jobs[key]=dict(status='PENDING',config=str(target),sha256=sha256(target))
    state=dict(status='WAITING_FOR_SPARE_PHYSICAL_GPU2',jobs=jobs,producers={},source_sha256=sources,specification_sha256=sha256(spec_path),Test_read=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MFT26IR_ROOT=main['data_root'])
    while True:
        fitting=json.loads((work/'correlation_recoverability_ablation_fit_state.json').read_text())
        if fitting['status']=='FAILED':state['status']='FIT_DEPENDENCY_FAILED';dump(path,state);return
        if fitting['status']=='COMPLETE':break
        dump(path,state);time.sleep(5)
    physical_bytes=sum(p.lstat().st_size for p in work.rglob('*')if not p.is_symlink()and p.is_file())
    if physical_bytes>64*2**30 or shutil.disk_usage(work).free<40*2**30:raise RuntimeError('User storage budget/reserve exceeded')
    state['physical_work_GiB_at_start']=physical_bytes/2**30;state['status']='PRODUCING_ALL_TEMPORAL_DEV_POLICIES'
    for seq in ['BT-005','SN-013']:
        if sha256(spec_path)!=state['specification_sha256']or any(sha256(ROOT/n)!=value for n,value in sources.items()):
            raise ValueError('Frozen temporal experiment changed')
        command=[sys.executable,'-u',str(source),'--sequence',seq];log=work/(seq+'.temporal_full_dev.log')
        with log.open('ab',buffering=0)as handle:
            proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES='2'),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        state['producers'][seq]=dict(status='RUNNING',pid=proc.pid,physical_gpu='2',command=command,log=str(log));dump(path,state)
        while proc.poll()is None:time.sleep(5)
        code=proc.returncode;state['producers'][seq].update(status='COMPLETE'if code==0 else'FAILED',exit_code=code)
        if code:state['status']='FAILED';dump(path,state);return
        check=json.loads((work/'temporal_correction_predictions/dev'/seq/'complete.json').read_text())
        if not check['no_memory_reference_all_exact']:state['status']='FULL_NO_MEMORY_PARITY_FAILED';dump(path,state);return
        dump(path,state)
    state['status']='RUNNING';active={}
    while True:
        for key,proc in list(active.items()):
            code=proc.poll()
            if code is not None:jobs[key].update(status='COMPLETE'if code==0 else'FAILED',exit_code=code);del active[key]
        for key,job in jobs.items():
            if len(active)>=2 or any(j['status']=='FAILED'for j in jobs.values()):break
            if job['status']!='PENDING':continue
            if sha256(job['config'])!=job['sha256']:raise ValueError('Frozen temporal replay config changed')
            command=[sys.executable,'-u','tools/replay_correlation_observations.py','--config',job['config'],
                '--output',str(work/'temporal_correction_dev'/key)]
            log=work/(key+'.temporal_dev_replay.log')
            with log.open('ab',buffering=0)as handle:
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[key]=proc;job.update(status='RUNNING',pid=proc.pid,command=command,log=str(log))
        if not active:
            if all(j['status']=='COMPLETE'for j in jobs.values()):state['status']='COMPLETE'
            elif any(j['status']=='FAILED'for j in jobs.values()):state['status']='FAILED'
        dump(path,state)
        if state['status']!='RUNNING':break
        time.sleep(5)


if __name__=='__main__':main()
