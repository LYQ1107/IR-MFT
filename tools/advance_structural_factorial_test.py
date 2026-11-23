#!/usr/bin/env python3
"""Run all new formal component Test cases after complete Dev and profiling."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_pseudo_ir_comparison import dump,sha256


def main():
    plan=json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text());work=Path(plan['work_root'])
    path=work/'structural_factorial_next_stage_state.json'
    lock=path.with_suffix('.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not path.exists()
    names=['tools/register_structural_factorial_test.py','tools/extract_structural_factorial.py',
           'tools/replay_structural_factorial_test.py','tools/launch_structural_factorial_test.py',
           'tools/export_structural_factorial_test.py','tools/advance_structural_factorial_test.py',
           'tools/summarize_structural_factorial.py','tools/structural_gate_observer.py',
           'tools/perceptual_gpu_handoff.py','configs/perceptual_gpu_handoff_20261123.json',
           'configs/perceptual_structural_factorial_20261123.json']
    sources={name:sha256(ROOT/name)for name in names}
    state=dict(status='WAITING_FOR_COMPLETE32DEV_AND_ANALYSIS_AND_RUNTIME',controller_pid=os.getpid(),
        sources=sources,fixed_seed=0,additional_seed_work=False,goal_met=False,supported_innovations=0)
    dump(path,state)
    try:
        report=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_STRUCTURAL_FACTORIAL_ANALYSIS.json'
        parent_report=ROOT/'docs/fusion_research/PERCEPTUAL_MFT_20261123_STRUCTURAL_NOCYCLE_TEST.json'
        while True:
            dev=json.loads((work/'structural_factorial_dev_state.json').read_text())
            if 'FAIL'in dev['status']:raise RuntimeError('Retain Dev failure before new Test')
            parent=json.loads((work/'structural_nocycle_test_state.json').read_text())
            if 'FAIL'in parent['status']:raise RuntimeError('Preserve parent Test failure before reusing sources')
            runtime_path=work/'structural_factorial_runtime_schedule.json'
            runtime=json.loads(runtime_path.read_text())if runtime_path.exists()else{}
            if 'FAIL'in runtime.get('status',''):raise RuntimeError('Retain component runtime failure; resolve before Test')
            if (dev['status']=='COMPLETE'and report.exists()and parent['status']=='COMPLETE'and parent_report.exists()
                    and runtime.get('status')=='COMPLETE_BOTH_RUNTIME_CONDITIONS_NO_GPU_CONFLICT'):break
            time.sleep(5)
        assert all(sha256(ROOT/name)==digest for name,digest in sources.items())
        subprocess.run([sys.executable,'tools/register_structural_factorial_test.py'],cwd=ROOT,check=True)
        log=work/'launch_structural_factorial_test.log'
        with log.open('ab',buffering=0)as handle:
            proc=subprocess.Popen([sys.executable,'-u','tools/launch_structural_factorial_test.py'],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(status='RUNNING_ALL_REQUIRED_COMPONENT_TEST_CASES',test_controller_pid=proc.pid,log=str(log))
        dump(path,state);assert proc.wait()==0
        subprocess.run([sys.executable,'tools/export_structural_factorial_test.py'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'tools/export_perceptual_official_tables.py'],cwd=ROOT,check=True)
        state['status']='COMPLETE_FROZEN_TEST_REQUIRES_SCIENTIFIC_ASSESSMENT';dump(path,state)
    except Exception as error:
        state.update(status='FAILED',error=repr(error));dump(path,state);raise


if __name__=='__main__':main()
