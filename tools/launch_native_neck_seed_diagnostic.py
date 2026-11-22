#!/usr/bin/env python3
"""Fixed seed0 full-Dev diagnosis while the other matched fits continue."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main():
    work=Path(json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())['work_root'])
    path=work/'native_neck_seed0_dev_state.json'
    lock=path.with_suffix('.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if path.exists():raise ValueError('Fixed-seed diagnostic already scheduled')
    jobs={f'{cue}_{sequence}':dict(status='PENDING',cue=cue,sequence=sequence)
          for sequence in ['BT-005','SN-013'] for cue in ['ir','rgb']}
    state=dict(status='RUNNING',jobs=jobs,scope='Fixed seed0 only, full Dev, no Test. All other fits continue.')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=':'.join([str(ROOT),str(ROOT/'exps'),'/data3/liuyeqiang/.codex/pydeps',
        '/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages',
        '/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT'])
    active={}
    while True:
        for gpu,(name,proc) in list(active.items()):
            code=proc.poll()
            if code is not None:
                jobs[name].update(status='COMPLETE' if code==0 else 'FAILED',exit_code=code);del active[gpu]
        for gpu in [2,3]:
            if gpu in active or any(j['status']=='FAILED' for j in jobs.values()):continue
            for name,job in jobs.items():
                if job['status']!='PENDING' or not (work/'native_neck_metric'/f"{job['cue']}_s0/complete.json").exists():continue
                command=[sys.executable,'-u','tools/extract_native_neck_eval_features.py','--split','dev',
                         '--sequence',job['sequence'],'--cue',job['cue'],'--seed-only','0']
                log=work/(name+'.native_neck_seed0_dev.log')
                with log.open('ab',buffering=0) as handle:
                    proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),
                        stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=(name,proc);job.update(status='RUNNING',pid=proc.pid,physical_gpu=gpu,command=command,log=str(log));break
        if not active:
            if all(j['status']=='COMPLETE' for j in jobs.values()):state['status']='FEATURES_COMPLETE'
            elif any(j['status']=='FAILED' for j in jobs.values()):state['status']='FAILED'
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,path)
        if state['status']!='RUNNING':break
        time.sleep(5)
    if state['status']=='FEATURES_COMPLETE':
        log=work/'native_neck_seed0_identity_audit.log'
        with log.open('wb') as handle:
            result=subprocess.run([sys.executable,'-u','tools/audit_native_neck_seed0_dev.py'],cwd=ROOT,
                env=dict(env,CUDA_VISIBLE_DEVICES=''),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
        state.update(status='COMPLETE' if result.returncode==0 else 'FAILED',audit_exit_code=result.returncode,audit_log=str(log))
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,path)


if __name__=='__main__':main()
