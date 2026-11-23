#!/usr/bin/env python3
"""Use GPU2 after its Dev worker exits, while GPU3 finishes the longer video."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_pseudo_ir_comparison import dump, sha256


def main():
    plan = json.loads((ROOT/'configs/perceptual_mft_20261122.json').read_text())
    work = Path(plan['work_root'])
    path = work/'structural_nocycle_runtime_schedule.json'
    lock = path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not path.exists()
    state = dict(status='WAITING_FOR_OWN_BT_DEV_PRODUCER_TO_RELEASE_GPU2',controller_pid=os.getpid(),
        benchmark_physical_gpu='2',fixed_seed=0,additional_seed_work=False,
        benchmark_source_sha256=sha256(ROOT/'tools/benchmark_structural_nocycle.py'),
        source_sha256=sha256(__file__),goal_met=False,supported_innovations=0)
    dump(path,state)
    try:
        while True:
            dev = json.loads((work/'structural_nocycle_dev_state.json').read_text())
            if 'FAIL' in dev['status']:
                raise RuntimeError('Retain Dev failure; no new benchmark GPU work')
            if dev.get('producers',{}).get('BT-005',{}).get('status') == 'COMPLETE':
                break
            time.sleep(5)
        assert sha256(ROOT/'tools/benchmark_structural_nocycle.py') == state['benchmark_source_sha256']
        env = dict(os.environ,CUDA_VISIBLE_DEVICES='2',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        for mode in ['live_file','memory_inputs']:
            command = [sys.executable,'-u','tools/benchmark_structural_nocycle.py']
            if mode == 'memory_inputs':
                command += ['--memory-inputs-and-components']
            log = work/('benchmark_structural_nocycle_'+mode+'.log')
            with log.open('ab',buffering=0)as handle:
                proc = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                    stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status='RUNNING_'+mode.upper(),benchmark_pid=proc.pid,log=str(log))
            dump(path,state)
            assert proc.wait() == 0,str(log)
        docs = ROOT/'docs/fusion_research'
        stem = 'PERCEPTUAL_MFT_20261123_STRUCTURAL_NOCYCLE_SINGLE_METHOD_RUNTIME'
        live = json.loads((docs/(stem+'.json')).read_text())
        memory = json.loads((docs/(stem+'_MEMORY_COMPONENTS.json')).read_text())
        assert len(live['rows']) == len(memory['rows']) == 5
        rows = {r['id']:r for r in memory['rows']}
        lines = ['# Structural IR: verified per-method runtime','',memory['scope'],'',
            '| Method at epoch6 | Adapter params | Live-file FPS | RAM-input pipeline FPS | NN/NMS/output-transfer FPS | Peak allocated GiB |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for row in live['rows']:
            m = rows[row['id']]
            lines.append(f"| {row['id']} | {row['adapter_parameters']} | {row['FPS']:.3f} | "
                f"{m['FPS']:.3f} | {m['NN_NMS_FPS']:.3f} | {m['peak_own_allocated_GPU_GiB']:.3f} |")
        lines += ['', 'Both runs check320frame/method comparisons each on64uniqueDevframes. '
            'These are two input conditions, not additional model seeds or independent accuracy trials. '
            'Full JSON retains source packet/checkpoint/code hashes, preparation and tracking time. '
            'Dataset reading and IR generation are excluded only where explicitly stated; generation is excluded in both. '
            'B4throughput is not B1online latency. Shared hardware load is uncontrolled.']
        (docs/(stem+'.md')).write_text('\n'.join(lines)+'\n')
        state['status'] = 'COMPLETE_BOTH_RUNTIME_CONDITIONS_NO_GPU_CONFLICT'
        dump(path,state)
    except Exception as error:
        state.update(status='FAILED',error=repr(error))
        dump(path,state)
        raise


if __name__ == '__main__':
    main()
