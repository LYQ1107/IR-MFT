#!/usr/bin/python3
"""One read-only CPU diagnostic after the original temporal-IR nine-epoch fit retires."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/data3/liuyeqiang/IR-MFT')
WORK = Path('/var/tmp/liuyeqiang_ir_mft_perceptual_20261122')
PREFIX = WORK/'native_temporal_ir_nine_epoch_learning_20261126_r1'
SCRIPT = ROOT/'tools/audit_mft_native_learning_state.py'
EXPECTED_SOURCE = 'aab4f92fbb9d608f3cab56251dc0436c7cb7fd395be366c9baebe5e88ed5cfde'
PYTHON = '/data3/liuyeqiang/.venvs/trackocd-a100/bin/python'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def identity(pid):
    folder=Path('/proc')/str(pid)
    fields=(folder/'stat').read_text().rsplit(')',1)[1].split()
    argv=(folder/'cmdline').read_bytes()
    return dict(pid=pid, start_ticks=int(fields[19]),
        raw_argv_sha256=hashlib.sha256(argv).hexdigest(), argv=argv)

def save(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)

state=json.loads((WORK/'native_pyramid_study_state.json').read_text())
jobs=state['phases']['NORMAL_TRAIN_09']
job=jobs['temporal_ir']
assert job['status']=='COMPLETE' and type(job['exit_code']) is int and job['exit_code']==0
assert all(j['status']=='COMPLETE' and type(j.get('exit_code')) is int and j['exit_code']==0
           for j in jobs.values())
try:
    previous=identity(job['pid'])
except FileNotFoundError:
    previous=None
assert previous is None or previous['start_ticks']!=94463963 or previous['raw_argv_sha256']!='66a0f63ac580910fbc9c6ffb500f3320ee3fabad4d22aa4fcfdb5269a075c026'
assert sha(SCRIPT)==EXPECTED_SOURCE
receipt=PREFIX.with_suffix('.execution.json')
log=PREFIX.with_suffix('.log')
assert not PREFIX.exists() and not receipt.exists() and not log.exists()
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
    MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
    MFT26IR_ROOT='/data3/liuyeqiang/datasets/mft26-ir',
    PYTHONPATH=':'.join([str(ROOT),str(ROOT/'exps'),
        '/data3/liuyeqiang/.codex/pydeps',
        '/data3/liuyeqiang/.venvs/cfp-fishmot/lib/python3.10/site-packages',
        '/data3/liuyeqiang/CFP-FishMOT_references/HybridSORT']))
command=[PYTHON,'-u',str(SCRIPT),'--fit-id','temporal_ir','--through-epoch','9',
         '--output-directory',str(PREFIX)]
started=time.monotonic()
with log.open('xb') as output:
    child=subprocess.Popen(command,cwd=str(ROOT),env=env,stdout=output,stderr=subprocess.STDOUT)
    expected_argv=b''.join(os.fsencode(part)+b'\0' for part in command)
    actual=None
    for _ in range(300):
        try:
            observed=identity(child.pid)
            if observed['argv']==expected_argv:
                actual={key:value for key,value in observed.items() if key!='argv'}
                break
        except FileNotFoundError:
            pass
        if child.poll() is not None:
            break
        time.sleep(.01)
    details=dict(status='ACTUAL_CPU_LEARNING_DIAGNOSTIC_RUNNING',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), command=command,
        identity=actual, source_sha256=EXPECTED_SOURCE, log=str(log),
        output_directory=str(PREFIX), exit_code=None, new_GPU_work=False,
        original_model_process_already_retired=True,
        optimizer_latest_read_only_after_original_actual_exit_zero=True,
        original_job=job)
    save(receipt,details)
    result=child.wait()
details.update(status='ACTUAL_CPU_LEARNING_DIAGNOSTIC_EXITED',
    finished_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    exit_code=result, wall_seconds=time.monotonic()-started,
    log_sha256=sha(log), complete_report_exists=(PREFIX/'complete.json').exists(),
    source_unchanged=sha(SCRIPT)==EXPECTED_SOURCE)
save(receipt,details)
print(json.dumps(details,ensure_ascii=False),flush=True)
sys.exit(result if result else (0 if actual and details['source_unchanged'] and details['complete_report_exists'] else 2))
