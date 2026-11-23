#!/usr/bin/env python3
"""Complete the missing S/C/R single-component/static fits after current Test NN."""
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
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(plan['work_root'])
    spec_path=ROOT/'configs/perceptual_structural_factorial_20261123.json';spec=json.loads(spec_path.read_text())
    path=work/'structural_factorial_fit_state.json';lock=path.with_suffix('.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not path.exists()
    jobs={f"{r['cue']}_{r['variant']}_s0":dict(r,status='PENDING')for r in spec['fits']}
    assert len(jobs)==4 and all(j['seed']==0 for j in jobs.values())
    source_names=['tools/train_structural_factorial.py','tools/launch_structural_factorial_fits.py',
        'tools/train_structural_relative.py','yolox/models/structural_factorial_repair.py',
        'yolox/models/structural_relative_repair.py','yolox/models/structural_relative_losses.py',
        'yolox/models/frozen_head_repair.py','yolox/models/recoverable_curriculum.py',
        'tools/train_correlation_repair.py']
    sources={n:sha256(ROOT/n)for n in source_names}
    state=dict(status='WAITING_FOR_ALL7_NO_CYCLE_TEST_NN_PRODUCERS',jobs=jobs,
        controller_pid=os.getpid(),sources=sources,factorial_plan_sha256=sha256(spec_path),
        fixed_seed=0,additional_seed_work=False,goal_met=False,supported_innovations=0)
    dump(path,state);active={}
    original=json.loads((work/'structural_relative_fit/ir_no_cycle_s0/complete.json').read_text())
    expected_draws=[r['sample_digest']for r in original['history']]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MFT26IR_ROOT=plan['data_root'])
    while True:
        dependency=json.loads((work/'structural_nocycle_test_state.json').read_text())
        assert 'FAIL'not in dependency['status'],'Retain upstream failure; do not overlap GPU work'
        released=len(dependency['producers'])==7 and all(r['status']=='COMPLETE'for r in dependency['producers'].values())
        for physical,(key,phase,proc)in list(active.items()):
            code=proc.poll()
            if code is None:continue
            del active[physical];job=jobs[key];job[phase+'_exit_code']=code
            assert code==0,str(job['log'])
            folder=work/'structural_factorial_fit'/key
            if phase=='smoke':folder=folder/'smoke_2'
            complete=json.loads((folder/'complete.json').read_text())
            assert complete['initial_raw_RGB_head_bit_exact'] and complete['trainable_parameters']==145600
            assert complete['factorial_plan_sha256']==state['factorial_plan_sha256']
            assert (complete['cue'],complete['variant'],complete['seed'])==(job['cue'],job['variant'],0)
            if phase=='smoke':
                assert complete['status']=='SMOKE_COMPLETE' and complete['successful_steps_this_invocation']==2
                digest=complete['history'][0]['sample_digest']
                if 'shared_smoke_sample_digest'in state:assert digest==state['shared_smoke_sample_digest']
                state['shared_smoke_sample_digest']=digest;job['status']='SMOKE_PASSED'
            else:
                assert complete['status']=='COMPLETE_TRAIN_STRUCTURAL_RELATIVE'
                assert complete['epochs']==24 and complete['successful_steps_this_invocation']==6912
                assert [r['sample_digest']for r in complete['history']]==expected_draws
                job['status']='COMPLETE'
            job[phase+'_complete_sha256']=sha256(folder/'complete.json')
        for physical in ['4','5']:
            if physical in active or not released:continue
            candidates=[(key,j)for key,j in jobs.items()if j['physical_gpu']==physical and j['status']in ['PENDING','SMOKE_PASSED']]
            if not candidates:continue
            key,job=candidates[0];phase='smoke'if job['status']=='PENDING'else'fit'
            assert sha256(spec_path)==state['factorial_plan_sha256']
            assert all(sha256(ROOT/n)==h for n,h in sources.items())
            assert shutil.disk_usage(work).free>=40*2**30
            used=0
            for parent,dirs,files in os.walk(work,followlinks=False):
                dirs[:]=[d for d in dirs if not Path(parent,d).is_symlink()]
                used+=sum(Path(parent,n).stat().st_size for n in files if not Path(parent,n).is_symlink())
            assert used<64*2**30
            command=[sys.executable,'-u','tools/train_structural_factorial.py','--cue',job['cue'],
                     '--variant',job['variant'],'--seed','0']
            if phase=='smoke':command+=['--smoke-steps','2']
            log=work/(key+'.structural_factorial_'+phase+'.log')
            with log.open('ab',buffering=0)as handle:
                proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=physical),
                    stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[physical]=(key,phase,proc);job.update(status=phase.upper()+'_RUNNING',pid=proc.pid,command=command,log=str(log))
        state['status']='FITTING_FOUR_MISSING_COMPONENT_CASES'if active else'WAITING_FOR_ALL7_NO_CYCLE_TEST_NN_PRODUCERS'
        if all(j['status']=='COMPLETE'for j in jobs.values()):state['status']='COMPLETE'
        dump(path,state)
        if state['status']=='COMPLETE':break
        time.sleep(5)


if __name__=='__main__':
    try:main()
    except Exception as error:
        plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
        path=Path(plan['work_root'])/'structural_factorial_fit_state.json'
        if path.exists():
            state=json.loads(path.read_text())
            if state.get('controller_pid')==os.getpid():
                state.update(status='FAILED',error=repr(error));dump(path,state)
        raise
